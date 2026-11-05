// Static hosting replays a real NeuroKit2-generated trace against one page clock.
export class MockECG {
  async prepare() {
    if (this.samples) return;
    this.loading ??= this.load().catch(error => { this.loading = null; throw error; });
    return this.loading;
  }
  async load() {
    const url = new URL('./assets/ecg/mock-ecg.json', import.meta.url);
    const response = await fetch(url);
    if (!response.ok) throw new Error('Mock ECG manifest unavailable');
    const manifest = await response.json();
    if (manifest.schema_version !== 1 || manifest.source !== 'mock-neurokit2' || manifest.encoding !== 'float32-le'
      || manifest.file !== 'mock-ecg.f32' || !Number.isSafeInteger(manifest.samples) || manifest.samples < 1
      || !Number.isSafeInteger(manifest.settings?.sampling_rate) || manifest.settings.sampling_rate < 1) throw new Error('Invalid mock ECG manifest');
    const data = await fetch(new URL(manifest.file, url));
    if (!data.ok) throw new Error('Mock ECG samples unavailable');
    const bytes = await data.arrayBuffer();
    const hash = [...new Uint8Array(await crypto.subtle.digest('SHA-256', bytes))].map(x => x.toString(16).padStart(2, '0')).join('');
    if (hash !== manifest.sha256 || bytes.byteLength !== manifest.samples * 4) throw new Error('Mock ECG integrity mismatch');
    const view = new DataView(bytes);
    const samples = Float32Array.from({ length: manifest.samples }, (_, i) => view.getFloat32(i * 4, true));
    if (!samples.every(Number.isFinite)) throw new Error('Invalid ECG sample');
    this.manifest = manifest;
    this.samples = samples;
    this.rate = manifest.settings.sampling_rate;
    this.epoch = performance.now();
  }
  indexAt(now) { return Math.max(0, Math.floor((now - this.epoch) * this.rate / 1000)); }
  sampleAt(index) { return index < 0 ? 0 : this.samples[index % this.samples.length]; }
}
