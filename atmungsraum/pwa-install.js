export function mountAppInstall({
  button = document.getElementById('install-app'),
  dialog = document.getElementById('install-dialog'),
  closeButton = document.getElementById('close-install'),
  environment = window,
} = {}) {
  const display = environment.matchMedia('(display-mode: standalone)');
  let deferred, busy = false, installed = false;
  const sync = () => {
    const hadFocus = button.ownerDocument?.activeElement === button;
    button.hidden = installed || display.matches || environment.navigator.standalone === true;
    if (button.hidden) {
      deferred = undefined;
      if (dialog.open) dialog.close();
      else if (hadFocus) button.nextElementSibling?.focus({ preventScroll: true });
    }
  };
  const checkInstalled = async () => {
    if (button.hidden) return;
    try {
      const apps = await environment.navigator.getInstalledRelatedApps?.();
      if (!apps?.length) return;
      const id = new URL('/atmungsraum/', environment.location.href);
      const manifest = new URL('manifest.webmanifest', id);
      if (apps.some(app => app.platform === 'webapp' &&
          (app.id === id.href || (!app.id && app.url === manifest.href)))) {
        installed = true;
        sync();
      }
    } catch {
      // The installed-app lookup is optional.
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
      const choice = await prompt.userChoice;
      if (choice?.outcome === 'accepted') void checkInstalled();
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
  void checkInstalled();
}

if (typeof document !== 'undefined') mountAppInstall();
