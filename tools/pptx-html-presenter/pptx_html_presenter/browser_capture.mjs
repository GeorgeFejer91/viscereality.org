import { pathToFileURL } from "node:url";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";

const playwrightSpecifier = process.env.PLAYWRIGHT_PACKAGE_DIR
  ? pathToFileURL(`${process.env.PLAYWRIGHT_PACKAGE_DIR}/node_modules/playwright/index.mjs`).href
  : "playwright";
const { chromium } = await import(playwrightSpecifier);

const [buildDir, samplesFile, outputDir] = process.argv.slice(2);
if (!buildDir || !samplesFile || !outputDir) {
  console.error("usage: node browser_capture.mjs <buildDir> <samples.json> <outputDir>");
  process.exit(2);
}

const samples = JSON.parse(await readFile(samplesFile, "utf8"));
const server = await startStaticServer(buildDir);
const browser = await launchBrowserWithMediaCodecs();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
const baseUrl = `http://127.0.0.1:${server.port}/index.html`;

try {
  for (const sample of samples) {
    const captureSlide = sample.kind === "slide" ? sample.slide : sample.from;
    const progress = sample.kind === "slide" ? 0 : sample.progress;
    await page.goto(`${baseUrl}?captureSlide=${captureSlide}&progress=${progress}`);
    await page.waitForFunction(() => document.querySelector("#loading")?.classList.contains("hidden"));
    if (sample.kind === "slide") {
      await page.evaluate((s) => {
        document.dispatchEvent(new CustomEvent("pptx-html-presenter:capture-at", {
          detail: {
            slide: s.slide,
            progress: 0,
            trackProgressOverrides: s.trackProgressOverrides || null,
          },
        }));
      }, sample);
    } else {
      await page.evaluate((s) => {
        document.dispatchEvent(new CustomEvent("pptx-html-presenter:capture-at", {
          detail: {
            slide: s.from,
            progress: s.progress,
            trackProgressOverrides: s.trackProgressOverrides || null,
          },
        }));
      }, sample);
    }
    const mediaTime = Number(sample.mediaSec ?? 0);
    await waitForRenderableFrame(page, Number.isFinite(mediaTime) ? mediaTime : 0, sample.mediaClocks || {});
    await page.screenshot({ path: `${outputDir}/${sample.id}.png`, fullPage: false });
  }
} finally {
  await closeBrowser(browser);
  await closeStaticServer(server.instance);
}
process.exit(0);

async function closeBrowser(browserInstance) {
  await withTimeout(browserInstance.close(), 3000).catch(() => {});
}

async function closeStaticServer(instance) {
  await withTimeout(new Promise((resolve) => {
    const timeout = setTimeout(resolve, 2000);
    instance.close(() => {
      clearTimeout(timeout);
      resolve();
    });
    if (typeof instance.closeAllConnections === "function") {
      instance.closeAllConnections();
    }
  }), 2500).catch(() => {});
}

async function withTimeout(promise, ms) {
  let timeout = null;
  try {
    return await Promise.race([
      promise,
      new Promise((resolve) => {
        timeout = setTimeout(resolve, ms);
      }),
    ]);
  } finally {
    if (timeout) {
      clearTimeout(timeout);
    }
  }
}

async function launchBrowserWithMediaCodecs() {
  for (const channel of ["chrome", "msedge"]) {
    try {
      return await chromium.launch({ channel, headless: true });
    } catch {
      // Fall through to the next installed browser channel.
    }
  }
  return chromium.launch({ headless: true });
}

async function waitForRenderableFrame(page, mediaTimeSec, mediaClocks) {
  await page.waitForFunction(() => {
    const frame = document.querySelector("#frame");
    if (!frame) return false;
    const images = Array.from(frame.querySelectorAll("img"));
    const videos = Array.from(frame.querySelectorAll("video"));
    const imagesReady = images.every((image) => image.complete && image.naturalWidth > 0);
    const videosReady = videos.every((video) => !video.error && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA);
    return imagesReady && videosReady;
  }, null, { timeout: 30000 });
  await seekVideos(page, mediaTimeSec, mediaClocks);
  await page.waitForTimeout(500);
}

async function seekVideos(page, seconds, mediaClocks) {
  await page.evaluate(async ({ targetSeconds, trackSeconds }) => {
    const videos = Array.from(document.querySelectorAll("#frame video"));
    await Promise.all(videos.map((video) => new Promise((resolve) => {
      if (video.error) {
        resolve();
        return;
      }
      const trackId = video.closest(".obj")?.dataset.trackId || "";
      const desiredSeconds = Number(trackSeconds?.[trackId] ?? targetSeconds);
      const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
      const loopTarget = duration > 0.08 ? (desiredSeconds % duration) : desiredSeconds;
      const target = duration > 0
        ? Math.max(0, Math.min(loopTarget, Math.max(0, duration - 0.05)))
        : Math.max(0, desiredSeconds);
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        video.pause();
        resolve();
      };
      video.pause();
      if (Math.abs(video.currentTime - target) < 0.04 && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        finish();
        return;
      }
      video.addEventListener("seeked", finish, { once: true });
      video.currentTime = target;
      setTimeout(finish, 1200);
    })));
  }, { targetSeconds: seconds, trackSeconds: mediaClocks || {} });
}

async function startStaticServer(rootDir) {
  const root = path.resolve(rootDir);
  const mime = {
    ".html": "text/html; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".svg": "image/svg+xml",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
  };
  const instance = createServer(async (request, response) => {
    try {
      const url = new URL(request.url || "/", "http://127.0.0.1");
      const rawPath = url.pathname === "/" ? "/index.html" : url.pathname;
      const filePath = path.resolve(root, `.${decodeURIComponent(rawPath)}`);
      if (!filePath.startsWith(root)) {
        response.writeHead(403);
        response.end("Forbidden");
        return;
      }
      const fileInfo = await stat(filePath);
      const contentType = mime[path.extname(filePath).toLowerCase()] || "application/octet-stream";
      const range = request.headers.range;
      if (range) {
        const match = /^bytes=(\d*)-(\d*)$/.exec(range);
        if (!match || (match[1] === "" && match[2] === "")) {
          response.writeHead(416, { "Content-Range": `bytes */${fileInfo.size}` });
          response.end();
          return;
        }
        const suffixLength = match[1] === "" ? Number(match[2]) : null;
        const start = suffixLength == null ? Number(match[1]) : Math.max(0, fileInfo.size - suffixLength);
        const end = match[2] === "" || suffixLength != null ? fileInfo.size - 1 : Number(match[2]);
        if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end < start || start >= fileInfo.size) {
          response.writeHead(416, { "Content-Range": `bytes */${fileInfo.size}` });
          response.end();
          return;
        }
        const boundedEnd = Math.min(end, fileInfo.size - 1);
        response.writeHead(206, {
          "Accept-Ranges": "bytes",
          "Content-Type": contentType,
          "Content-Length": String(boundedEnd - start + 1),
          "Content-Range": `bytes ${start}-${boundedEnd}/${fileInfo.size}`,
        });
        createReadStream(filePath, { start, end: boundedEnd }).pipe(response);
        return;
      }
      response.writeHead(200, {
        "Accept-Ranges": "bytes",
        "Content-Type": contentType,
        "Content-Length": String(fileInfo.size),
      });
      createReadStream(filePath).pipe(response);
    } catch {
      response.writeHead(404);
      response.end("Not found");
    }
  });
  await new Promise((resolve) => instance.listen(0, "127.0.0.1", resolve));
  return { instance, port: instance.address().port };
}
