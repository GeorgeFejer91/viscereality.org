const visibleNs = 2000000000n;
const retainedMs = 4000;
const retainedNs = BigInt(retainedMs) * 1000000n;
const playoutDelayNs = 1000000000n;
const gapNs = 12000000n;
const maxSamples = 780;

// Presentation only: callers supply ecgTrace's validated native samples and a
// monotonic browser clock. No samples, sensor timestamps or voltages are made up.
export class LiveEcgDisplay {
  #points;
  #watermark;
  #cursor;
  #clock;
  #deadline;
  #range;
  #rangeClock;
  #baseline = 0;

  constructor() { this.reset(); }

  reset() {
    this.#watermark = null;
    this.#clear();
  }

  suspend() {
    // Missing evidence hides the image immediately. The next fresh snapshot
    // must prove continuity before it can reuse this bounded history.
    this.#deadline = -Infinity;
  }

  #clear() {
    this.#points = [];
    this.#cursor = null;
    this.#clock = null;
    this.#deadline = -Infinity;
    this.#range = 100;
    this.#baseline = 0;
    this.#rangeClock = null;
    // Only explicit source-owner/lifecycle reset may forget the native watermark.
  }

  push(trace, nowMs) {
    if (!Number.isFinite(nowMs)) return false;
    const incoming = trace?.points;
    if (!Array.isArray(incoming) || incoming.length < 2 || incoming.length > 260
      || !Number.isFinite(trace.ttl) || trace.ttl <= 0
      || incoming.some((point, index) => typeof point?.timestamp !== 'bigint' || point.timestamp <= 0n
        || !Number.isSafeInteger(point.value) || point.value < -8388608 || point.value > 8388607
        || typeof point.gap !== 'boolean' || index > 0 && point.timestamp <= incoming[index - 1].timestamp)) {
      this.#clear();
      return false;
    }
    const latest = incoming.at(-1).timestamp;
    if (this.#watermark !== null && latest <= this.#watermark) return false;
    nowMs = Math.max(nowMs, this.#clock ?? nowMs);

    const merged = new Map(this.#points.map(point => [point.timestamp, { ...point }]));
    if (nowMs >= this.#deadline && (this.#clock === null
      || nowMs - this.#clock > retainedMs || !incoming.some(point => merged.has(point.timestamp)))) {
      this.#clear();
      merged.clear();
    } else {
      this.#advance(nowMs);
    }
    for (let index = 0; index < incoming.length; index++) {
      const point = incoming[index];
      const prior = merged.get(point.timestamp);
      if (prior && prior.value !== point.value) {
        // A contradictory packet cannot become acceptable merely because the
        // failed image was cleared and the same packet was sent again.
        this.#watermark = latest;
        this.#clear();
        return false;
      }
      // The first gap flag is a packet start, not evidence of a real gap.
      // Interior gaps stay open even if a later overlap supplies older samples.
      merged.set(point.timestamp, { timestamp: point.timestamp, value: point.value,
        gap: Boolean(prior?.gap || index > 0 && point.gap) });
    }
    const points = [...merged.values()].sort((a, b) => a.timestamp < b.timestamp ? -1 : 1);
    for (let index = 1; index < points.length; index++) {
      points[index].gap ||= points[index].timestamp - points[index - 1].timestamp > gapNs;
    }
    this.#points = points.filter(point => latest - point.timestamp <= retainedNs).slice(-maxSamples);
    // Display-only DC offset: the median keeps QRS peaks from lifting the
    // baseline. Raw microvolts/timestamps remain untouched for every consumer.
    const values = this.#points.map(point => point.value).sort((a, b) => a - b);
    const middle = Math.floor(values.length / 2);
    this.#baseline = values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
    this.#watermark = latest;
    this.#deadline = nowMs + trace.ttl;
    if (this.#cursor === null || this.#cursor < this.#points[0].timestamp) {
      // Keep about a second of actual samples for 1 Hz delivery. Tiny packets
      // expose at least two received samples instead of waiting in empty space.
      this.#cursor = latest - playoutDelayNs;
      if (this.#cursor < this.#points[1].timestamp) this.#cursor = this.#points[1].timestamp;
      this.#clock = nowMs;
      this.#rangeClock = nowMs;
      this.#range = this.#requiredRange();
    }
    return true;
  }

  #advance(nowMs) {
    const nextClock = Math.max(nowMs, this.#clock);
    const nextCursor = this.#cursor + BigInt(Math.round((nextClock - this.#clock) * 1000000));
    // A delayed/missing packet freezes at the last received native timestamp.
    this.#cursor = nextCursor < this.#watermark ? nextCursor : this.#watermark;
    this.#clock = nextClock;
  }

  #requiredRange() {
    return Math.max(100, ...this.#points.map(point => Math.abs(point.value - this.#baseline)));
  }

  frame(nowMs) {
    if (!Number.isFinite(nowMs) || this.#cursor === null) return null;
    nowMs = Math.max(nowMs, this.#clock);
    if (nowMs >= this.#deadline) {
      this.suspend();
      return null;
    }
    this.#advance(nowMs);
    const required = this.#requiredRange();
    // Recent raw peaks hold the scale while buffered. Expansion is immediate;
    // after they leave history, a two-second release avoids a vertical snap.
    this.#range = Math.max(required,
      required + (this.#range - required) * Math.exp(-(nowMs - this.#rangeClock) / 2000));
    this.#rangeClock = nowMs;
    const range = Math.ceil(this.#range);
    const points = this.#points.filter(point => point.timestamp <= this.#cursor
      && this.#cursor - point.timestamp <= visibleNs).map(point => ({ ...point }));
    const path = points.map(({ timestamp, value, gap }, index) => {
      const x = 400 - Number(this.#cursor - timestamp) / 5000000;
      const y = 46 - (value - this.#baseline) / range * 43;
      return `${index === 0 || gap ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ');
    return { path, range, baseline: this.#baseline, points, cursorTimestamp: this.#cursor, latestTimestamp: this.#watermark,
      oldestTimestamp: this.#points[0].timestamp, bufferedSamples: this.#points.length,
      bufferedMs: Number(this.#watermark - this.#cursor) / 1000000, ttl: this.#deadline - nowMs };
  }
}
