// A read-only projection of native startup maintenance. Public release metadata
// and phone timers never authorize an install or release this admission fence.
const states = new Set(['waiting', 'checking', 'current', 'downloading', 'installing', 'updated', 'failed', 'restart_required']);
const blocking = new Set(['waiting', 'checking', 'downloading', 'installing', 'restart_required']);
const integer = value => Number.isSafeInteger(value) && value >= 0;
const positive = value => integer(value) && value > 0;

export function nativeMetaUpdate(status) {
  if (!status || !Object.hasOwn(status, 'meta_update')) return { advertised: false, allowed: true, value: null };
  const value = status.meta_update;
  const observation = status.o;
  const valid = value && !Array.isArray(value) && value.v === 1 && states.has(value.state)
    && typeof value.reason === 'string' && value.reason.length <= 96 && !/[^\x20-\x7e]/.test(value.reason)
    && positive(value.installed_code) && (value.available_code === null || positive(value.available_code))
    && integer(value.downloaded_bytes) && (value.total_bytes === null || positive(value.total_bytes)
      && value.downloaded_bytes <= value.total_bytes)
    && typeof value.session_allowed === 'boolean'
    && observation?.v === 1 && observation.s === 'observed'
    && typeof observation.e === 'string' && /^[1-9][0-9]{0,18}$/.test(observation.e)
    && typeof observation.i === 'string' && /^[1-9][0-9]{0,18}$/.test(observation.i)
    && integer(observation.a) && observation.a <= 1500
    && (!status.apk || status.apk.version_code === value.installed_code);
  return { advertised: true, allowed: Boolean(valid && value.session_allowed && !blocking.has(value.state)),
    value: valid ? value : null };
}

export const nativeSessionAllowed = status => nativeMetaUpdate(status).allowed;

const failureReasons = {
  offline: 'Prüfe die Internetverbindung der Quest.',
  storage: 'Prüfe den freien Speicherplatz auf der Quest.',
  authentication: 'Prüfe die Anmeldung beim Meta-Konto in der Quest.',
  'init-timeout': 'Die Verbindung zum Meta-Dienst hat zu lange gedauert.',
  'version-timeout': 'Die Versionsprüfung hat zu lange gedauert.',
  'download-timeout': 'Das Herunterladen hat zu lange gedauert.',
  'download-stalled': 'Das Herunterladen ist nicht weitergekommen.',
  'platform-initialization-timeout': 'Die Verbindung zum Meta-Dienst hat zu lange gedauert.',
  'version-check-timeout': 'Die Versionsprüfung hat zu lange gedauert.',
  'download-total-timeout': 'Das Herunterladen hat zu lange gedauert.',
  'download-progress-stalled': 'Das Herunterladen ist nicht weitergekommen.',
  'meta-platform-unavailable': 'Der Meta-Dienst ist gerade nicht verfügbar.',
  'fresh-process-required-for-update-check': 'Die App wurde im selben Quest-Prozess erneut geöffnet. Für eine neue Meta-Prüfung die Quest neu starten.',
  'pending-install-record-write-failed': 'Der Update-Stand konnte auf der Quest nicht gespeichert werden.',
};
const maintenanceReasons = new Set(['prior-install-unconfirmed-maintenance-required',
  'pending-install-record-invalid', 'pending-install-record-unreadable', 'pending-install-record-clear-failed']);

export function metaUpdateView(status) {
  const result = nativeMetaUpdate(status);
  if (!result.advertised) return { ...result, state: 'legacy', summary: 'Meta-Startprüfung wird von dieser APK nicht gemeldet.',
    detail: 'Prüfe den installierten APK-Stand und den Versionshinweis oben.', progress: null };
  if (!result.value) return { ...result, state: 'invalid', summary: 'Meta-Update-Status nicht verlässlich. Vorbereitung gesperrt.',
    detail: 'Prüfe die Quest-App und verbinde das Smartphone erneut. Nur ein gültiger nativer Status kann die Vorbereitung freigeben.', progress: null };
  const value = result.value;
  const maintenanceRequired = value.state === 'restart_required' && maintenanceReasons.has(value.reason);
  const summaries = {
    waiting: 'Quest wartet auf die Meta-Startprüfung. Vorbereitung gesperrt.',
    checking: 'Quest prüft den Meta-Kanal. Vorbereitung gesperrt.',
    current: `Meta-Startprüfung abgeschlossen · installierter APK-Code ${value.installed_code}.`,
    downloading: 'Quest lädt das Meta-Update herunter. Vorbereitung gesperrt.',
    installing: 'Quest hat die Installation angefragt. Vorbereitung gesperrt.',
    updated: `Quest meldet die erneuerte APK · installierter Code ${value.installed_code}.`,
    failed: result.allowed ? 'Meta-Startprüfung fehlgeschlagen. Die installierte APK kann verwendet werden.'
      : 'Meta-Startprüfung fehlgeschlagen. Vorbereitung bleibt gesperrt.',
    restart_required: maintenanceRequired ? 'Meta-Update nicht bestätigt. Vorbereitung gesperrt; Wartung erforderlich.'
      : 'Meta-Installation noch unklar. Vorbereitung gesperrt; Quest neu starten.',
  };
  let detail = maintenanceRequired
    ? 'Prüfe und vervollständige das Update in der Meta-Bibliothek der Quest. Bleibt der Update-Stand offen, ist Betreiberwartung erforderlich. Die Vorbereitung bleibt bis zur bestätigten APK-Kennung gesperrt.'
    : value.state === 'failed' && result.allowed
    ? 'Für eine erneute Prüfung die Quest neu starten. Die App nur zu schließen und neu zu öffnen reicht dafür nicht aus.'
    : ['installing', 'restart_required'].includes(value.state)
      ? 'Eine Installationsanfrage bestätigt keine neue APK. Starte die Quest neu; erst die danach tatsächlich gemeldete APK-Kennung bestätigt den Stand.'
      : result.allowed ? 'Die Quest hat die Vorbereitung freigegeben.' : 'Warte, bis die Quest die Vorbereitung freigibt.';
  if (!result.allowed && !blocking.has(value.state)) detail += ' Die Quest hat die Vorbereitung noch nicht freigegeben.';
  const progress = value.state === 'downloading' ? { value: value.downloaded_bytes, max: value.total_bytes } : null;
  if (progress) detail = `${value.downloaded_bytes.toLocaleString('de-DE')} Byte geladen${value.total_bytes === null
    ? ' · Gesamtgröße noch unbekannt.' : ` von ${value.total_bytes.toLocaleString('de-DE')} Byte.`} ${detail}`;
  if (value.available_code !== null) detail += ` Meta-Kanalcode: ${value.available_code}.`;
  if (value.state === 'failed' && failureReasons[value.reason]) detail += ` ${failureReasons[value.reason]}`;
  const summary = summaries[value.state] + (!result.allowed && ['current', 'updated'].includes(value.state)
    ? ' Vorbereitung bleibt gesperrt.' : '');
  return { ...result, state: value.state, summary, detail, progress };
}
