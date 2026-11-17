// A rejected CAS has no effect. Re-observe before a new request, and retain
// the original actor/clock/attempt/input intent. Never replay an uncertain ID.
export async function selectBreathingMethod(model, op, { refresh, onPending = () => {}, signal } = {}) {
  const gateway = model.gateway;
  const binding = model.view?.acceptance;
  if (!gateway || !binding) return;
  for (let attempt = 0; attempt < 3; attempt++) {
    signal?.throwIfAborted();
    await refresh();
    signal?.throwIfAborted();
    if (model.gateway !== gateway || model.view?.acceptance !== binding || !model.view?.actions[op]) return;
    const request = model.run(op, signal);
    onPending();
    if (await request !== 'stale-revision') return;
  }
}
