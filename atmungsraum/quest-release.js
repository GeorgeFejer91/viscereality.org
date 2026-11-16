const positive = value => Number.isSafeInteger(value) && value > 0;
const versionName = value => typeof value === 'string' && /^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$/.test(value);
const packageName = value => typeof value === 'string' && value.length <= 200
  && /^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$/.test(value);
const hash = value => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);

export function installedApk(status, owner) {
  const value = status?.apk;
  const observation = status?.o;
  return owner && /^[a-f0-9]{32}$/.test(owner.device) && typeof owner.native_epoch === 'string' && /^[1-9][0-9]{0,18}$/.test(owner.native_epoch)
    && observation?.v === 1 && observation.s === 'observed' && observation.e === owner.native_epoch
    && typeof observation.i === 'string' && /^[1-9][0-9]{0,18}$/.test(observation.i)
    && Number.isSafeInteger(observation.a) && observation.a >= 0 && observation.a <= 1500
    && value?.v === 1 && Object.keys(value).length === 5 && packageName(value.package_name)
    && positive(value.version_code) && versionName(value.version_name) && hash(value.build_id)
    && status.build?.v === 1 && value.build_id === status.build.packaging_fingerprint ? { ...value } : null;
}

function releasePolicy(value) {
  if (value?.v !== 1 || !value.packages || Array.isArray(value.packages)
    || Object.keys(value.packages).length === 0 || Object.keys(value.packages).length > 16) return null;
  for (const [name, entry] of Object.entries(value.packages)) {
    if (!packageName(name) || !entry || Object.keys(entry).length !== 3 || !positive(entry.minimum_version_code)
      || !positive(entry.latest_version_code) || entry.latest_version_code < entry.minimum_version_code || !versionName(entry.version_name)) return null;
  }
  return value;
}

// Metadata only. This owner has no headset command, install or experiment gates.
export class ApkRelease {
  policy = null;
  state = 'checking';
  refreshing = null;

  constructor({ fetcher = (...args) => fetch(...args), onChange = () => {} } = {}) {
    this.fetcher = fetcher; this.onChange = onChange;
  }

  refresh() {
    if (this.refreshing) return this.refreshing;
    this.policy = null; this.state = 'checking';
    this.refreshing = (async () => {
      try {
        const response = await this.fetcher(new URL('./quest-release.json', import.meta.url), {
          cache: 'no-store', credentials: 'omit', mode: 'same-origin', redirect: 'error',
          referrerPolicy: 'no-referrer', signal: AbortSignal.timeout(5000),
        });
        if (!response.ok) throw Error('Release policy unavailable');
        const body = await response.text();
        if (body.length > 16384) throw Error('Release policy too large');
        this.policy = releasePolicy(JSON.parse(body));
        this.state = this.policy ? 'ready' : 'check-unavailable';
      } catch { this.policy = null; this.state = 'check-unavailable'; }
    })().finally(() => { this.refreshing = null; this.onChange(); });
    this.onChange();
    return this.refreshing;
  }

  update(gateway, status) {
    // Never retain an installed identity across a missing status or actor swap.
    const installed = installedApk(status, gateway?.owner);
    const entry = installed && this.policy && Object.hasOwn(this.policy.packages, installed.package_name)
      ? this.policy.packages[installed.package_name] : null;
    let state = this.state === 'ready' ? 'unknown' : this.state;
    if (this.state === 'ready' && installed && entry) state = installed.version_code < entry.minimum_version_code ? 'required'
      : installed.version_code < entry.latest_version_code ? 'recommended' : 'compatible';
    const messages = {
      checking: 'APK-Versionsprüfung läuft …',
      'check-unavailable': 'Versionsprüfung nicht verfügbar. Der aktuelle Freigabestand konnte nicht geladen werden. Versuche es bei bestehender Internetverbindung erneut.',
      required: 'Update der Quest-App erforderlich. Die installierte APK liegt unter dem benötigten Versionscode.',
      recommended: 'Quest-App kompatibel. Ein neuerer APK-Stand wird empfohlen.',
      compatible: 'Quest-App erfüllt die Versionsvorgabe.',
      unknown: !status ? 'Installierte APK-Version unbekannt. Verbinde die Quest für eine Versionsprüfung.'
        : !installed ? 'Installierte APK-Version unbekannt. Diese Quest-App meldet keine prüfbare APK-Kennung. Prüfe im Headset, ob ein App-Update erforderlich ist.'
          : 'Für dieses APK-Paket ist kein Freigabestand hinterlegt. Ein Update-Bedarf kann nicht bestimmt werden.',
    };
    return {
      state, installed, compatible: ['compatible', 'recommended'].includes(state), message: messages[state],
      installedLabel: installed ? `Installiert: ${installed.version_name} · Code ${installed.version_code} · Build ${installed.build_id.slice(0, 12)}` : 'Installiert: nicht verlässlich gemeldet.',
      required: entry ? `Mindestens Code ${entry.minimum_version_code}. Referenz: ${entry.version_name} · Code ${entry.latest_version_code}.`
        : this.state === 'checking' ? 'Benötigter APK-Stand wird geprüft.' : 'Benötigter APK-Stand nicht verfügbar.',
      identity: installed ? `Paket: ${installed.package_name}\nBuild: ${installed.build_id}` : 'Keine bestätigte APK-Kennung verfügbar.',
    };
  }
}
