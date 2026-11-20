// Presentation only: this module has no gateway, storage or experiment dependency.
export const operatorSteps = Object.freeze({
  connect: 'quest', participant: 'participant-screen', signals: 'signals-screen',
  input: 'input-screen', calibration: 'calibration-screen', tracking: 'breathing-preview',
  battery: 'battery-screen', session: 'session-screen', running: 'running-screen', recordings: 'recordings-screen',
  settings: 'settings-screen', update: 'update-screen', 'update-help': 'manual-update-screen',
  recovery: 'recovery-screen', questionnaire: 'questionnaire-screen',
});
const previous = { participant: 'connect', signals: 'participant', input: 'participant',
  calibration: 'input', tracking: 'input', battery: 'tracking', session: 'battery', recordings: 'session' };
export function navigateOperatorView(state, event, participantRequired = true) {
  if (event.type === 'return' && state.step === 'update-help') return {
    step: ['settings', 'update'].includes(state.returnTo) ? state.returnTo : 'settings',
    returnTo: Object.hasOwn(operatorSteps, state.helpReturnTo) ? state.helpReturnTo : null,
  };
  if (event.type === 'return') return { ...state,
    step: Object.hasOwn(operatorSteps, state.returnTo) && state.returnTo !== state.step ? state.returnTo
      : state.step === 'update' ? 'connect' : 'signals', returnTo: null };
  if (event.type === 'back') {
    const step = previous[state.step];
    return { ...state, step: step === 'participant' && !participantRequired ? 'connect' : step || 'connect' };
  }
  if (event.type !== 'go' || !Object.hasOwn(operatorSteps, event.step)) return state;
  if (state.step === 'update-help' && event.step !== 'update-help') {
    const origin = navigateOperatorView(state, { type: 'return' }, participantRequired);
    return event.step === origin.step ? origin : navigateOperatorView(origin, event, participantRequired);
  }
  if (event.step === 'update-help') return { step: event.step,
    returnTo: ['settings', 'update'].includes(state.step) ? state.step : 'settings',
    helpReturnTo: state.returnTo ?? null };
  const step = event.step === 'participant' && !participantRequired ? 'input' : event.step;
  return { ...state, step, returnTo: step === state.returnTo ? null
    : step === 'update' && ['settings', 'recordings'].includes(state.step) ? state.returnTo || state.step
    : ['settings', 'recordings', 'update'].includes(step) ? state.step : state.returnTo };
}
export function resolveOperatorView(step, context) {
  if (context.questionnaire && !context.active && !context.recovery && !context.instructionPending) return 'questionnaire';
  if (step === 'questionnaire') step = 'recordings';
  // Settings retains shared recovery and session controls in its own view.
  if (step === 'settings' || step === 'update-help') return step;
  if (!context.connected) {
    if (context.closeoutRecovering) return 'recordings';
    return step === 'recordings' && (!context.recovery || context.offlineArchives) ? 'recordings' : 'connect';
  }
  if (step === 'update' && !(context.updateBlocked ?? context.recovery) && !context.waitingForControl) return 'update';
  if (context.active) return 'running';
  if (context.instructionPending) return 'calibration';
  if (context.recovery) return 'recovery';
  if (step === 'recordings') return 'recordings';
  if (context.waitingForControl) return 'connect';
  if (step === 'update' || context.updateRequired && step && step !== 'connect') return 'update';
  if (!Object.hasOwn(operatorSteps, step)) return 'connect';
  if (context.participantRequired && !context.participantComplete && step !== 'connect') return 'participant';
  if (step === 'participant' && !context.participantRequired) return 'input';
  if (['battery', 'session'].includes(step) && !context.selectionAllowed) return context.previewAvailable === false ? 'input' : 'tracking';
  if (step === 'session' && !context.batteryReviewed) return 'battery';
  if (step === 'calibration' && context.calibrationComplete) return 'tracking';
  if (step === 'calibration' && context.calibrationAvailable === false
    || step === 'tracking' && context.previewAvailable === false) return 'input';
  if (['running', 'recovery'].includes(step)) return context.selectionAllowed
    ? context.batteryReviewed ? 'session' : 'battery' : 'signals';
  return step;
}

export function operatorCompatibilityWarning(state) {
  return ['required', 'check-unavailable', 'unknown'].includes(state);
}

export function restoreOperatorView(hash, entry) {
  const requested = hash.slice(1);
  const step = Object.hasOwn(operatorSteps, requested) ? requested : null;
  const returnTo = entry?.operatorStep === step && entry?.operatorReturnTo !== step
    && Object.hasOwn(operatorSteps, entry?.operatorReturnTo) ? entry.operatorReturnTo : null;
  return { step, returnTo, ...(step === 'update-help' ? {
    helpReturnTo: Object.hasOwn(operatorSteps, entry?.operatorHelpReturnTo) ? entry.operatorHelpReturnTo : null,
  } : {}) };
}

export function operatorSessionHeading(phase) {
  return { ARMED: 'Sitzung vorbereitet', RUNNING: 'Sitzung läuft', PAUSED: 'Sitzung pausiert',
    FINALIZING: 'Sitzung wird beendet', SAVING: 'Aufnahme wird gespeichert', ERROR: 'Sitzung prüfen' }[phase] ?? 'Sitzung läuft';
}
