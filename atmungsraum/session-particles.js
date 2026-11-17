const count = 72;
const goldenAngle = Math.PI * (3 - Math.sqrt(5));
const points = Array.from({ length: count }, (_, i) => {
  const y = 1 - 2 * (i + .5) / count;
  const circle = Math.sqrt(1 - y * y);
  return { x: Math.cos(goldenAngle * i) * circle, y, depth: (1 + Math.sin(goldenAngle * i) * circle) / 2 };
});

export class SessionParticles {
  constructor(canvas) {
    this.canvas = canvas;
  }

  render({ key, progress, paused, reducedMotion, now }) {
    if (key !== this.key) {
      this.key = key;
      this.progress = undefined;
    }
    if (!key) this.progress = null;
    else if (!paused || this.progress == null) {
      this.progress = progress;
    }
    const painted = reducedMotion ? null : this.progress;
    if (painted === this.painted) return this.progress;
    if (painted !== null && Number.isFinite(this.painted) && now - this.lastPaint < 1000 / 24) return this.progress;
    this.painted = painted;
    this.lastPaint = now;
    const { width, height } = this.canvas;
    const context = this.canvas.getContext('2d');
    if (!context) return this.progress;
    context.clearRect(0, 0, width, height);
    if (painted === null) return this.progress;
    const radius = (1 + painted) * Math.min(width, height) * .22;
    context.fillStyle = '#a8ddf7';
    for (const { x, y, depth } of points) {
      context.globalAlpha = .25 + depth * .65;
      context.beginPath();
      context.arc(width / 2 + x * radius, height / 2 + y * radius, .75 + depth, 0, 2 * Math.PI);
      context.fill();
    }
    return this.progress;
  }
}
