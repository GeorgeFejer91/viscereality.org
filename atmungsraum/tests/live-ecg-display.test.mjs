import test from 'node:test';
import assert from 'node:assert/strict';
import { ecgTrace } from '../quest-data.js';
import { LiveEcgDisplay } from '../live-ecg-display.js';

const start = 900719925474099100n;
const step = 7692307n;
test('display centers a DC-biased baseline without changing any raw sample', () => {
  const display = new LiveEcgDisplay();
  const packet = trace(0, 260, 0, i => 900 + (i % 20 === 0 ? 200 : 0));
  display.push(packet, 0);
  const frame = display.frame(0);
  assert.equal(frame.baseline, 900);
  assert.ok(frame.path.includes('46.0'));
  assert.ok(frame.range <= 200);
  assert.deepEqual(frame.points.map(p => [p.timestamp, p.value]), packet.points.slice(0, frame.points.length).map(p => [p.timestamp, p.value]));
  assert.equal(packet.points[1].value, 900);
});
function trace(first = 0, count = 260, age = 0, value = index => index % 2 ? -40 : 80) {
  return ecgTrace({ v: 1, rate: 130, unit: 'uV', age_ms: age,
    samples: Array.from({ length: count }, (_, index) =>
      [String(start + BigInt(first + index) * step), value(first + index)]) }, 0);
}

test('validated ECG exposes exact bigint timestamps, microvolts and real gaps', () => {
  const input = { v: 1, rate: 130, unit: 'uV', age_ms: 100,
    samples: [[String(start), 22], [String(start + step), -40], [String(start + 40000000n), 30]] };
  const result = ecgTrace(input, 60);
  assert.deepEqual(result.points, [
    { timestamp: start, value: 22, gap: true },
    { timestamp: start + step, value: -40, gap: false },
    { timestamp: start + 40000000n, value: 30, gap: true },
  ]);
  assert.equal(result.ttl, 2340);
  assert.equal(result.path.match(/M/g).length, 2);
  assert.equal(ecgTrace({ ...input, samples: [...input.samples].reverse() }, 0), null);
  assert.equal(ecgTrace({ ...input, age_ms: 1501 }, 0), null);
});

test('the rolling position advances gradually through received raw samples between packets', () => {
  const display = new LiveEcgDisplay();
  const packet = trace();
  display.push(packet, 0);
  const first = display.frame(0);
  const later = display.frame(250);
  assert.ok(first && later);
  assert.equal(later.cursorTimestamp - first.cursorTimestamp, 250000000n);
  assert.equal(later.latestTimestamp, start + 259n * step);
  assert.ok(later.points.length > first.points.length);
  const firstX = Number(first.path.match(/^M([\d.]+)/)[1]);
  const laterX = Number(later.path.match(/^M([\d.]+)/)[1]);
  assert.equal(firstX - laterX, 50);
  assert.ok(later.points.every(point => point.timestamp <= later.cursorTimestamp));
  assert.deepEqual(later.points.map(point => [point.timestamp, point.value]),
    packet.points.slice(0, later.points.length).map(point => [point.timestamp, point.value]));
});

test('overlapping one-second deliveries merge without duplicate samples or false segment breaks', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(), 0);
  display.frame(0);
  assert.equal(display.push(trace(130), 1000), true);
  const merged = display.frame(1000);
  assert.equal(merged.bufferedSamples, 390);
  assert.equal(merged.latestTimestamp, start + 389n * step);
  assert.equal(new Set(merged.points.map(point => point.timestamp)).size, merged.points.length);
  assert.equal(merged.path.match(/M/g).length, 1);
});

test('starvation reaches the received edge and freezes without extrapolated microvolts', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(), 0);
  const edge = display.frame(1000);
  const starved = display.frame(2200);
  assert.ok(edge && starved);
  assert.equal(edge.cursorTimestamp, start + 259n * step);
  assert.equal(starved.cursorTimestamp, edge.cursorTimestamp);
  assert.equal(starved.path, edge.path);
  assert.deepEqual(starved.points, edge.points);
  assert.equal(starved.bufferedMs, 0);
});

test('a replayed native watermark neither restarts playback nor renews its original expiry', () => {
  const display = new LiveEcgDisplay();
  const packet = trace();
  display.push(packet, 0);
  const first = display.frame(500);
  assert.equal(display.push(packet, 600), false);
  const later = display.frame(700);
  assert.ok(first && later);
  assert.equal(later.cursorTimestamp - first.cursorTimestamp, 200000000n);
  assert.equal(later.ttl, 1800);
  assert.equal(display.frame(2500), null);
  assert.equal(display.push(packet, 2600), false);
  assert.equal(display.frame(2600), null);
});

test('older deliveries cannot replace the latest trace or extend its freshness', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(130), 0);
  assert.equal(display.push(trace(), 1000), false);
  const current = display.frame(1000);
  assert.ok(current);
  assert.equal(current.latestTimestamp, start + 389n * step);
  assert.equal(current.ttl, 1500);
  assert.equal(display.frame(2500), null);
});

test('the trace expires at the validated age-adjusted deadline even with data still buffered', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(0, 260, 1400), 100);
  assert.ok(display.frame(1199));
  assert.equal(display.frame(1200), null);
});

test('a missing packet keeps an open SVG segment across separate deliveries', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(0, 130), 0);
  display.frame(1000);
  display.push(trace(150, 130), 1000);
  const current = display.frame(2000);
  assert.ok(current);
  assert.equal(current.path.match(/M/g).length, 2);
  assert.equal(current.points.find(point => point.timestamp === start + 150n * step).gap, true);
  assert.equal(current.points.some(point => point.timestamp > start + 129n * step
    && point.timestamp < start + 150n * step), false);
});

test('an explicit interior gap remains open when later overlap fills its timestamp interval', () => {
  const display = new LiveEcgDisplay();
  const packet = trace();
  packet.points[150].gap = true;
  display.push(packet, 0);
  display.push(trace(130), 1000);
  const current = display.frame(1000);
  assert.ok(current);
  assert.equal(current.path.match(/M/g).length, 2);
  assert.equal(current.points.find(point => point.timestamp === start + 150n * step).gap, true);
});

test('an invalid response clears the image and cannot let an old watermark revive it', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(), 0);
  display.push(null, 500);
  assert.equal(display.frame(500), null);
  assert.equal(display.push(trace(), 700), false);
  assert.equal(display.frame(700), null);
  assert.equal(display.push(trace(130), 800), true);
  assert.ok(display.frame(800));
});

test('reset retires previous playout, scale and watermark for a new source owner', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(0, 260, 0, () => 2000), 0);
  assert.equal(display.frame(0).baseline, 2000);
  assert.equal(display.frame(0).range, 100);
  display.reset();
  assert.equal(display.frame(500), null);
  assert.equal(display.push(trace(), 1000), true);
  const current = display.frame(1000);
  assert.equal(current.range, 100);
  assert.equal(current.ttl, 2500);
  assert.equal(current.cursorTimestamp, start + 259n * step - 1000000000n);
});

test('retained history is bounded in sensor time over a long sequence of deliveries', () => {
  const display = new LiveEcgDisplay();
  for (let index = 0; index < 10; index++) {
    display.push(trace(index * 130), index * 1000);
    const current = display.frame(index * 1000);
    assert.ok(current);
    assert.ok(current.latestTimestamp - current.oldestTimestamp <= 4000000000n);
    assert.ok(current.bufferedSamples <= 780);
    assert.ok(current.points.every(point => current.cursorTimestamp - point.timestamp <= 2000000000n));
  }
});

test('a dense valid packet sequence also obeys the hard sample-count bound', () => {
  const display = new LiveEcgDisplay();
  for (let first = 0; first < 1300; first += 130) {
    const packet = ecgTrace({ v: 1, rate: 130, unit: 'uV', age_ms: 0,
      samples: Array.from({ length: 260 }, (_, index) => [String(start + BigInt(first + index) * 1000000n), 22]) }, 0);
    display.push(packet, first);
    const current = display.frame(first);
    assert.ok(current);
    assert.ok(current.bufferedSamples <= 780);
  }
  assert.equal(display.frame(1300).bufferedSamples, 780);
});

test('scale keeps a recent peak then releases gradually while every raw value fits', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(0, 260, 0, index => index === 0 ? 900 : 40), 0);
  assert.equal(display.frame(0).range, 860);
  for (let index = 1; index <= 2; index++) {
    display.push(trace(index * 130, 260, 0, () => 40), index * 1000);
    assert.equal(display.frame(index * 1000).range, 860);
  }
  display.push(trace(390, 260, 0, () => 40), 3000);
  const releasing = display.frame(3000);
  const next = display.frame(3001);
  assert.ok(releasing.range > 100 && releasing.range < 860);
  assert.ok(releasing.range - next.range <= 1);
  assert.ok(next.points.every(point => Math.abs(point.value - next.baseline) <= next.range));
  const coordinates = [...next.path.matchAll(/[ML]([\d.]+) ([\d.]+)/g)];
  assert.ok(coordinates.every(point => Number(point[2]) >= 3 && Number(point[2]) <= 89));
});

test('large new peaks expand the displayed scale without clipping or altering samples', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(), 0);
  display.frame(0);
  const packet = trace(130, 260, 0, index => index < 260 ? (index % 2 ? -40 : 80) : -2000);
  const before = structuredClone(packet.points);
  display.push(packet, 1000);
  const current = display.frame(1500);
  assert.ok(current.range >= Math.abs(-2000 - current.baseline));
  assert.ok(current.points.some(point => point.value === -2000));
  assert.ok(current.points.every(point => Math.abs(point.value - current.baseline) <= current.range));
  assert.deepEqual(packet.points, before);
  current.points[0].value = 12345;
  assert.notEqual(display.frame(1500).points[0].value, 12345);
});

test('conflicting overlap clears the image and repeating that packet cannot revive it', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(), 0);
  const changed = trace(130, 260, 0, index => index === 180 ? 700 : index % 2 ? -40 : 80);
  assert.equal(display.push(changed, 1000), false);
  assert.equal(display.frame(1000), null);
  assert.equal(display.push(changed, 1100), false);
  assert.equal(display.frame(1100), null);
  assert.equal(display.push(trace(260), 1200), true);
  assert.ok(display.frame(1200));
});

test('a minimal validated packet shows only its two measured samples at the received edge', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(0, 2), 0);
  const current = display.frame(0);
  assert.ok(current);
  assert.equal(current.points.length, 2);
  assert.equal(current.cursorTimestamp, start + step);
  assert.equal(current.bufferedMs, 0);
  assert.equal(display.frame(1000).path, current.path);
});

test('a source gap beyond retained history resumes with measured data instead of an empty playout', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(), 0);
  display.frame(900);
  display.push(trace(1000), 1000);
  const current = display.frame(1000);
  assert.ok(current && current.path);
  assert.ok(current.points.every(point => point.timestamp >= start + 1000n * step));
  assert.equal(current.path.match(/M/g).length, 1);
});

test('an invalid or regressing display clock cannot revive expired samples', () => {
  const display = new LiveEcgDisplay();
  display.push(trace(), 0);
  assert.equal(display.frame(NaN), null);
  const current = display.frame(1000);
  assert.ok(current);
  assert.equal(display.frame(500).cursorTimestamp, current.cursorTimestamp);
  assert.equal(display.frame(2500), null);
  assert.equal(display.frame(0), null);
});
