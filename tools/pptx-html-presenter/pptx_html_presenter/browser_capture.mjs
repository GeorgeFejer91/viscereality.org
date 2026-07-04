import { pathToFileURL } from "node:url";
import { createReadStream } from "node:fs";
import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
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
await mkdir(outputDir, { recursive: true });
const server = await startStaticServer(buildDir);
const browser = await launchBrowserWithMediaCodecs();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
const baseUrl = `http://127.0.0.1:${server.port}/${server.basePath}/index.html`;
const report = [];
const pageEvents = [];
page.on("pageerror", (error) => {
  pageEvents.push({
    type: "pageerror",
    text: String(error?.stack || error?.message || error),
  });
});
page.on("console", (message) => {
  if (!["error", "warning"].includes(message.type())) return;
  pageEvents.push({
    type: `console:${message.type()}`,
    text: message.text(),
  });
});

try {
  await page.goto(`${baseUrl}?captureSlide=1&progress=0`, { waitUntil: "domcontentloaded", timeout: 45000 });
  await page.waitForFunction(() => document.querySelector("#loading")?.classList.contains("hidden"), null, { timeout: 45000 });
  for (const sample of samples) {
    const eventStart = pageEvents.length;
    const row = {
      id: sample.id,
      kind: sample.kind,
      status: "pending",
      file: `${sample.id}.png`,
    };
    try {
      if (sample.kind === "slide") {
        await page.evaluate((s) => {
          document.dispatchEvent(new CustomEvent("pptx-html-presenter:capture-at", {
            detail: {
              slide: s.slide,
              progress: 0,
              direction: s.direction || "forward",
              trackProgressOverrides: s.trackProgressOverrides || null,
              unmatchedFadeOverride: s.unmatchedFadeOverride || null,
              visualEffectOverrides: s.visualEffectOverrides || null,
            },
          }));
        }, sample);
      } else {
        await page.evaluate((s) => {
          document.dispatchEvent(new CustomEvent("pptx-html-presenter:capture-at", {
            detail: {
              slide: s.from,
              progress: s.progress,
              direction: s.direction || "forward",
              trackProgressOverrides: s.trackProgressOverrides || null,
              unmatchedFadeOverride: s.unmatchedFadeOverride || null,
              visualEffectOverrides: s.visualEffectOverrides || null,
            },
          }));
        }, sample);
      }
      const mediaTime = Number(sample.mediaSec ?? 0);
      const ready = await waitForRenderableFrame(page, Number.isFinite(mediaTime) ? mediaTime : 0, sample.mediaClocks || {});
      row.status = ready.ok ? "ok" : "failed";
      row.diagnostics = ready.diagnostics;
      if (!ready.ok) row.error = ready.error;
      const sampleEvents = pageEvents.slice(eventStart);
      if (sampleEvents.length) row.pageEvents = sampleEvents;
      const pageErrors = sampleEvents.filter((event) => event.type === "pageerror");
      if (pageErrors.length) {
        row.status = "failed";
        row.error = `page-error:${pageErrors[0].text}`;
      }
      await page.screenshot({ path: `${outputDir}/${sample.id}.png`, fullPage: false });
    } catch (error) {
      row.status = "failed";
      row.error = String(error?.message || error);
      const sampleEvents = pageEvents.slice(eventStart);
      if (sampleEvents.length) row.pageEvents = sampleEvents;
      row.diagnostics = await collectRenderableDiagnostics(page).catch((diagnosticError) => ({
        diagnosticError: String(diagnosticError?.message || diagnosticError),
      }));
      await page.screenshot({ path: `${outputDir}/${sample.id}.png`, fullPage: false }).catch(() => {});
    }
    report.push(row);
  }
} finally {
  await writeFile(`${outputDir}/capture-report.json`, JSON.stringify({
    generatedAt: new Date().toISOString(),
    sampleCount: samples.length,
    failures: report.filter((row) => row.status !== "ok").length,
    samples: report,
  }, null, 2));
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
  await ensureDiagnosticsHelpers(page);
  try {
    await page.waitForFunction(() => {
      const diagnostics = window.__pptxHtmlPresenterDiagnostics();
      return diagnostics.imagesPending === 0;
    }, null, { timeout: 8000 });
  } catch (error) {
    return {
      ok: false,
      error: `image-timeout:${error?.message || error}`,
      diagnostics: await collectRenderableDiagnostics(page),
    };
  }
  const seek = await seekVideos(page, mediaTimeSec, mediaClocks);
  if (!seek.ok) {
    const diagnostics = await collectRenderableDiagnostics(page);
    return { ok: false, error: seek.error, diagnostics };
  }
  try {
    await page.waitForFunction(() => {
      const diagnostics = window.__pptxHtmlPresenterDiagnostics();
      return diagnostics.videosPending === 0;
    }, null, { timeout: 5000 });
  } catch {
    const diagnostics = await collectRenderableDiagnostics(page);
    return { ok: false, error: "visible-video-not-ready", diagnostics };
  }
  const diagnostics = await collectRenderableDiagnostics(page);
  if (Number(diagnostics.videosPending || 0) > 0) {
    return { ok: false, error: "visible-video-not-ready", diagnostics };
  }
  return { ok: true, diagnostics };
}

async function seekVideos(page, seconds, mediaClocks) {
  try {
    await page.evaluate(async ({ targetSeconds, trackSeconds }) => {
      const videos = Array.from(document.querySelectorAll("#frame video"));
      const waitForVideo = (video, predicate, events, timeoutMs) => new Promise((resolve) => {
        if (predicate()) {
          resolve();
          return;
        }
        let done = false;
        let timer = null;
        const finish = () => {
          if (done) return;
          done = true;
          if (timer) clearTimeout(timer);
          for (const eventName of events) {
            video.removeEventListener(eventName, onEvent);
          }
          resolve();
        };
        const onEvent = () => {
          if (predicate()) finish();
        };
        for (const eventName of events) {
          video.addEventListener(eventName, onEvent);
        }
        timer = setTimeout(finish, timeoutMs);
      });
      await Promise.all(videos.filter((video) => window.__pptxHtmlPresenterElementVisible(video)).map(async (video) => {
        if (video.error) return;
        video.preload = "auto";
        video.pause();
        if (video.readyState < HTMLMediaElement.HAVE_METADATA || video.networkState === HTMLMediaElement.NETWORK_EMPTY) {
          try {
            video.load();
          } catch {
            // Keep going; diagnostics will report the unresolved media state.
          }
          await waitForVideo(
            video,
            () => video.error || video.readyState >= HTMLMediaElement.HAVE_METADATA,
            ["loadedmetadata", "loadeddata", "canplay", "error"],
            3500,
          );
        }
        const trackId = video.closest(".obj")?.dataset.trackId || "";
        const desiredSeconds = Number(trackSeconds?.[trackId] ?? targetSeconds);
        const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
        const loopTarget = duration > 0.08 ? (desiredSeconds % duration) : desiredSeconds;
        const target = duration > 0
          ? Math.max(0, Math.min(loopTarget, Math.max(0, duration - 0.05)))
          : Math.max(0, desiredSeconds);
        if (Math.abs(video.currentTime - target) >= 0.04 || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
          try {
            video.currentTime = target;
          } catch {
            // Some browsers reject seeks before metadata; the readiness wait below catches this.
          }
          await waitForVideo(
            video,
            () => video.error || (
              Math.abs(video.currentTime - target) < 0.08
              && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
            ),
            ["seeked", "loadeddata", "canplay", "timeupdate", "error"],
            4000,
          );
        }
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA && !video.error) {
          try {
            video.load();
          } catch {
            // Diagnostics will surface the remaining not-ready video if this does not recover.
          }
          await waitForVideo(
            video,
            () => video.error || video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA,
            ["loadeddata", "canplay", "canplaythrough", "error"],
            4000,
          );
        }
        if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA && !video.error) {
          const wasMuted = video.muted;
          video.muted = true;
          try {
            await video.play();
          } catch {
            // Headless browsers can still reject autoplay; diagnostics will report remaining state.
          }
          await waitForVideo(
            video,
            () => video.error || video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA,
            ["loadeddata", "canplay", "canplaythrough", "timeupdate", "error"],
            4000,
          );
          video.pause();
          video.muted = wasMuted;
        }
        video.pause();
      }));
    }, { targetSeconds: seconds, trackSeconds: mediaClocks || {} });
    return { ok: true };
  } catch (error) {
    return { ok: false, error: `seek-failed:${error?.message || error}` };
  }
}

async function collectRenderableDiagnostics(page) {
  await ensureDiagnosticsHelpers(page);
  return page.evaluate(() => window.__pptxHtmlPresenterDiagnostics());
}

async function ensureDiagnosticsHelpers(page) {
  await page.evaluate(() => {
    if (window.__pptxHtmlPresenterDiagnostics) return;
    window.__pptxHtmlPresenterElementVisible = (element) => {
      const frame = document.querySelector("#frame");
      if (!frame || !element) return false;
      const obj = element.closest(".obj") || element;
      const rect = obj.getBoundingClientRect();
      const frameRect = frame.getBoundingClientRect();
      const style = window.getComputedStyle(obj);
      const opacity = Number(style.opacity || 1);
      return (
        style.display !== "none"
        && style.visibility !== "hidden"
        && opacity > 0.005
        && rect.width > 1
        && rect.height > 1
        && rect.left < frameRect.right
        && rect.right > frameRect.left
        && rect.top < frameRect.bottom
        && rect.bottom > frameRect.top
      );
    };
    window.__pptxHtmlPresenterDiagnostics = () => {
      const frame = document.querySelector("#frame");
      if (!frame) {
        return { frame: false, imagesVisible: 0, imagesPending: 0, videosVisible: 0, videosPending: 0, objectsVisible: 0 };
      }
      const stableSrc = (value) => {
        try {
          const url = new URL(value || "", window.location.href);
          return `${url.pathname}${url.search}${url.hash}`;
        } catch {
          return value || "";
        }
      };
      const visibleObjects = Array.from(frame.querySelectorAll(".obj")).filter((obj) => window.__pptxHtmlPresenterElementVisible(obj));
      const images = Array.from(frame.querySelectorAll("img")).filter((image) => window.__pptxHtmlPresenterElementVisible(image));
      const videos = Array.from(frame.querySelectorAll("video")).filter((video) => window.__pptxHtmlPresenterElementVisible(video));
      const imageRows = images.map((image) => ({
        src: stableSrc(image.currentSrc || image.src),
        complete: image.complete,
        naturalWidth: image.naturalWidth,
        trackId: image.closest(".obj")?.dataset.trackId || "",
      }));
      const videoRows = videos.map((video) => ({
        src: stableSrc(video.currentSrc || video.src),
        readyState: video.readyState,
        paused: video.paused,
        error: video.error ? String(video.error.code || video.error.message || "video-error") : null,
        trackId: video.closest(".obj")?.dataset.trackId || "",
      }));
      return {
        frame: true,
        objectsVisible: visibleObjects.length,
        imagesVisible: images.length,
        imagesPending: imageRows.filter((image) => !image.complete || image.naturalWidth <= 0).length,
        videosVisible: videos.length,
        videosPending: videoRows.filter((video) => video.error || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA).length,
        images: imageRows,
        videos: videoRows,
      };
    };
  });
}

async function startStaticServer(rootDir) {
  const deckRoot = path.resolve(rootDir);
  const root = path.dirname(deckRoot);
  const basePath = path.basename(deckRoot);
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
  return { instance, port: instance.address().port, basePath };
}
