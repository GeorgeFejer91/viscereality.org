export function mountAppInstall({
  button = document.getElementById('install-app'),
  dialog = document.getElementById('install-dialog'),
  closeButton = document.getElementById('close-install'),
  environment = window,
} = {}) {
  const display = environment.matchMedia('(display-mode: standalone)');
  let deferred, busy = false, installed = false;
  const sync = () => {
    button.hidden = installed || display.matches || environment.navigator.standalone === true;
    if (button.hidden) {
      deferred = undefined;
      if (dialog.open) dialog.close();
    }
  };
  const showHelp = () => {
    if (button.hidden || dialog.open) return;
    dialog.showModal();
    button.setAttribute('aria-expanded', 'true');
  };
  environment.addEventListener('beforeinstallprompt', event => {
    event.preventDefault();
    if (!button.hidden) deferred = event;
  });
  environment.addEventListener('appinstalled', () => { installed = true; sync(); });
  display.addEventListener('change', sync);
  button.addEventListener('click', async () => {
    if (busy || button.hidden) return;
    if (!deferred) return showHelp();
    const prompt = deferred;
    deferred = undefined;
    busy = true;
    button.disabled = true;
    try {
      await prompt.prompt();
      await prompt.userChoice;
    } catch {
      showHelp();
    } finally {
      busy = false;
      button.disabled = false;
    }
  });
  closeButton.addEventListener('click', () => dialog.close());
  dialog.addEventListener('keydown', event => { if (event.key === 'Escape') event.stopPropagation(); });
  dialog.addEventListener('close', () => {
    button.setAttribute('aria-expanded', 'false');
    (button.hidden ? button.nextElementSibling : button)?.focus({ preventScroll: true });
  });
  // Reserve the dock's actual height when zoom or wrapping enlarges its label.
  if (environment.ResizeObserver) {
    const dock = button.parentElement, root = button.ownerDocument.documentElement;
    new environment.ResizeObserver(() => {
      const height = `${Math.ceil(dock.getBoundingClientRect().height)}px`;
      if (root.style.getPropertyValue('--dock-height') !== height) root.style.setProperty('--dock-height', height);
    }).observe(dock);
  }
  sync();
}

if (typeof document !== 'undefined') mountAppInstall();
