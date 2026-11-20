import { audioMixStatus } from './quest-gateway.js?v=22';

const defaults = { music: 30, instructions: 90 };
const channels = ['music', 'instructions'];

export function mountQuestAudioMix() {
  const group = document.getElementById('quest-audio-mix');
  const reset = group.querySelector('button');
  const message = group.querySelector('[role="status"]');
  const sliders = Object.fromEntries(channels.map(name => [name, group.querySelector('input[name="' + name + '"]')]));
  const outputs = Object.fromEntries(channels.map(name => [name, sliders[name].previousElementSibling]));
  let gateway, lease, observed, draft, active, inFlight, queued, error;

  function render() {
    const enabled = Boolean(active && observed);
    for (const channel of channels) {
      const slider = sliders[channel];
      const level = (draft ?? queued ?? inFlight?.values ?? observed ?? defaults)[channel];
      slider.disabled = !enabled;
      const label = level + ' %';
      slider.value = level;
      if (outputs[channel].value !== label) outputs[channel].value = label;
      if (slider.getAttribute('aria-valuetext') !== label) slider.setAttribute('aria-valuetext', label);
    }
    reset.disabled = !enabled;
    const text = error ? 'Die Quest hat die Lautstärken nicht bestätigt.' : '';
    if (message.textContent !== text) message.textContent = text;
    message.hidden = !error;
  }

  async function send(values) {
    if (!active || !observed || document.hidden) { render(); return; }
    if (inFlight) { queued = values; render(); return; }
    const attempt = { values, gateway, lease };
    inFlight = attempt;
    error = false;
    render();
    try {
      const result = await attempt.gateway.audioMix({ v: 1, ...values }, AbortSignal.timeout(4000));
      if (inFlight === attempt && result.revision >= observed.revision) observed = result;
    } catch {
      if (inFlight === attempt) error = true;
    } finally {
      if (inFlight === attempt) {
        inFlight = null;
        const next = queued;
        queued = null;
        render();
        if (next) void send(next);
      }
    }
  }

  for (const name of channels) {
    const slider = sliders[name];
    slider.addEventListener('input', () => {
      if (!active || !observed) return;
      const values = draft ?? queued ?? inFlight?.values ?? observed;
      draft = { music: values.music, instructions: values.instructions, [name]: slider.valueAsNumber };
      render();
    });
    slider.addEventListener('change', () => {
      if (!draft) return;
      const values = draft;
      draft = null;
      void send(values);
    });
    slider.addEventListener('pointercancel', () => { draft = null; render(); });
  }
  reset.addEventListener('click', () => { draft = null; void send(defaults); });

  return {
    update(connection, status, controlActive) {
      const nextLease = status?.controller_access?.revision;
      if (connection !== gateway || nextLease !== lease || !controlActive && active) {
        if (connection !== gateway) observed = null;
        gateway = connection;
        lease = nextLease;
        draft = queued = inFlight = error = null;
      }
      const snapshot = audioMixStatus(status?.audio_mix);
      const supported = Boolean(gateway?.audioMix && snapshot);
      active = supported && controlActive;
      if (snapshot && (!observed || snapshot.revision >= observed.revision)) observed = snapshot;
      group.hidden = !supported;
      render();
      return supported;
    },
  };
}
