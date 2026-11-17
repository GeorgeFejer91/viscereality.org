import test from 'node:test';
import assert from 'node:assert/strict';
import { selectBreathingMethod } from '../breathing-selection.js';

function fixture(results = []) {
  const calls = [];
  const model = { gateway: {}, view: { acceptance: 'actor/clock/attempt/method', actions: { controller: true } },
    async run(op) { calls.push(op); return results.shift(); } };
  return { model, calls };
}
test('an explicitly rejected stale method uses fresh observations and a finite new request', async () => {
  const {model,calls} = fixture(['stale-revision', undefined]); let reads=0;
  await selectBreathingMethod(model, 'controller', { refresh: async()=>reads++ });
  assert.deepEqual(calls,['controller','controller']); assert.equal(reads,2);
});
test('accepted or unknown outcomes never cause a replay', async () => {
  for (const outcome of [undefined, 'accepted', 'outcome_unknown', 'pending']) {
    const {model,calls}=fixture([outcome]);
    await selectBreathingMethod(model,'controller',{refresh:async()=>{}});
    assert.equal(calls.length,1);
  }
});
test('actor, setup, or action-gate changes stop stale retry', async () => {
  for (const change of [m=>m.gateway={}, m=>m.view.acceptance='other-attempt', m=>m.view.actions.controller=false]) {
    const {model,calls}=fixture(['stale-revision']);let reads=0;
    await selectBreathingMethod(model,'controller',{refresh:async()=>{if(++reads===2)change(model);}});
    assert.equal(calls.length,1);
  }
});
test('repeated explicit stale rejections are bounded without timers', async () => {
  const {model,calls}=fixture(Array(9).fill('stale-revision'));
  await selectBreathingMethod(model,'controller',{refresh:async()=>{}});
  assert.equal(calls.length,3);
});
