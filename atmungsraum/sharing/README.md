# Wormhole delivery to Discord

1. Open [Atmungsraum](https://viscereality.org/atmungsraum/) and unlock data
   delivery with the private sharing password. **Passwort im Browser merken**
   asks the browser to save it. Browsers that allow silent sign-in can restore
   it on a later visit. Cancel leaves the app usable and defers the unlock
   until the first send.
2. Open the bottom-right folder. Choose **Ungesendete Daten senden** for files
   without a receipt or **Alle Daten senden** to include every saved CSV again.
   Both actions reuse the unlocked session in that tab.
3. The app encrypts the ZIP and uploads it to Wormhole, then sends the complete
   download link directly to the configured Discord channel.

The folder shows pending counts, progress and confirmation. Each acknowledged
Discord post creates a browser-local JSON receipt. Failed sends leave files
pending. Check Discord after an ambiguous failure before retrying. Individual
downloads are available at the [recovery URL](https://viscereality.org/atmungsraum/#daten).

## Maintainer deployment

The site reads a central upload configuration. Once the service is active,
operators only need the sharing password. Render Free is the selected host for the upload-only
[Blueprint](../../render.yaml). GitHub Pages deploys the browser app separately.

1. Run `node sharing/configure-render.mjs` from `atmungsraum/` to prepare or
   reuse the ignored private key and setup page. In the project's signed-in
   Render account, [deploy the Blueprint](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FGeorgeFejer91%2Fviscereality.org)
   and set `LINK_RELAY_TOKEN` to that key. Render receives neither the sharing
   password nor the Discord webhook.
2. With `DISCORD_WEBHOOK_URL` set in ignored `sharing/.dev.vars`, run
   `node sharing/configure-discord.mjs https://ACTUAL-HOST.onrender.com`.
   This reuses `sharing/.dev.vars.discord-password` if present, otherwise
   it creates a random password. Publish only `sharing/discord-config.json`,
   the authenticated ciphertext. The generated `sharing/.dev.vars.setup.html`
   is a private shortcut for the existing password.
3. Check that the actual service returns HTTP 204 from `/health` and verify
   upload with a synthetic ZIP. Render Free can sleep. The browser waits for
   readiness before sending any encrypted chunks.

The browser unlocks both credentials locally with AES-256-GCM and
PBKDF2-HMAC-SHA256 at 600,000 iterations. The app holds unlocked credentials in
tab memory. Browser credential storage is opt-in and handled by the browser.
Public ciphertext permits offline password guesses. Anyone with the password
can recover both credentials. Rotate the Discord webhook and relay key to
revoke previously unlocked copies.

The ZIP is encrypted on-device with pinned MIT
[`wormhole-crypto@0.3.1`](../vendor/wormhole-crypto/README.md). Two passes hash
and upload identical ciphertext pieces of up to 8 MiB. Discord receives the
keyed URL with link previews and mentions suppressed. Acknowledgement confirms
Discord accepted the message. Wormhole controls expiry and download limits.

See [DATA-LOGGING.md](../for-ai/DATA-LOGGING.md) for the data contract and
[PROJECT.md](../for-ai/PROJECT.md) for the hosted relay status.
