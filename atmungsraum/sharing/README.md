# Password-protected Discord sharing

The website sends the complete Wormhole download link **directly to Discord**.
No Cloudflare account, relay, bot or server is needed for this notification.
The webhook is hosted as encrypted `discord-config.json` on GitHub Pages.
A masked password dialog unlocks it locally immediately before sending.
The ZIP stays on Wormhole; the Discord message contains its full keyed URL.

## Use it

1. In [Atmungsraum](https://viscereality.org/atmungsraum/), open the bottom-right
   folder. If needed, enter the service URL and key under **Upload-Dienst
   einrichten** using the Render steps below.
2. Select **Neue Dateien senden** for files without a receipt, or **Alle
   Dateien senden** to include every saved file again. Enter the private
   sharing password and press **Senden**. The password is checked before any
   file transfer. The popup shows upload progress and confirms when Discord
   accepts the URL.
3. To reuse the password, select **Passwort im Browser merken** and accept the
   browser's save prompt after success. Later prompts can autofill the
   **Atmungsraum Discord** entry. If your browser doesn't offer this, use its
   regular password manager or the private local `.dev.vars.setup.html` page.

The JSON history stays in this browser and marks exact file versions only after
Discord accepts the URL. Clearing site data resets it. On the first use, files
from older versions of the app count as new. A failed upload or Discord post
leaves the selected files pending. Closing the dialog cancels pending work; a
provider request already accepted cannot be undone. Check Discord after a
timeout before retrying. Individual CSV downloads remain at
[the recovery URL](https://viscereality.org/atmungsraum/#daten).

The app clears the password field on dismissal and does not write the password
or decrypted webhook into browser storage. Opt-in saving and autofill are
handled by the browser's credential manager.

## Configure or change the password locally

The supplied webhook is already in this machine's ignored `.dev.vars`.
For a fresh checkout, create that file privately with:

```dotenv
DISCORD_WEBHOOK_URL=<your complete Discord webhook URL>
```

Run from `atmungsraum/`:

```powershell
node sharing/configure-discord.mjs
```

The helper generates a random144-bit password if no private password file
exists, encrypts the webhook, verifies decryption and writes:

- `sharing/discord-config.json`: **public ciphertext**, safe to commit.
- `sharing/.dev.vars.discord-password`: **private password**, never commit.
- `sharing/.dev.vars.setup.html`: **private setup/password page**, never commit.

To choose a different password, edit `.dev.vars.discord-password` locally using
a strong single-line password of at least16 characters, rerun the helper and
publish the updated encrypted JSON. Never put the password/webhook in source,
chat, screenshots, command arguments or an online encryption tool. The helper
prints paths/status only. Review the staged diff and scan it for local secrets.

Encryption uses native Web Crypto: AES-256-GCM, a fresh96-bit IV and128-bit salt,
with PBKDF2-HMAC-SHA256 at600,000 iterations and fixed versioned associated data.
The envelope is bounded and authenticated; an invalid/tampered file fails closed.
The browser sends the password to neither GitHub nor Discord. See
[Web Crypto](https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/deriveKey)
and [OWASP's PBKDF2 work factor](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html#pbkdf2).
This does not assert a FIPS-certified implementation.

The ciphertext is public, so someone can try password guesses offline. Anyone
who knows the correct password can recover and reuse the webhook, including
through browser developer tools after unlocking it. Changing only the password
does not revoke a previously recovered webhook; old encrypted copies also remain
in Git history. To revoke access, rotate the webhook in Discord, update the
private source value and regenerate/publish the encrypted file. This is a shared
operator password, not individual accounts or server-enforced rate limiting.

Direct delivery uses Discord's
[execute-webhook API](https://docs.discord.com/developers/resources/webhook#execute-webhook)
with `wait=true`, no cookies, no referrer, no redirects, no mentions and suppressed
link previews. Only provider acknowledgement is reported; it is not proof that
someone read the message. The full Wormhole URL gives its recipient access to
the shared ZIP while Wormhole permits it.

## Automatic Wormhole upload on Render Free

Wormhole blocks direct cross-origin uploads from the site. The [Render
Blueprint](../../render.yaml) deploys an authenticated upload-only relay and
keeps the existing browser encryption and upload protocol. Discord delivery
still goes straight from the browser to Discord. The relay needs one separate
access key; it never needs the Discord webhook or password.

1. On this machine, open the ignored `sharing/.dev.vars.render-setup.html`.
   For a fresh setup, run `node sharing/configure-render.mjs` from `atmungsraum/`
   to generate the private page and access key. Neither belongs in Git.
2. [Deploy the upload relay on Render](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2FGeorgeFejer91%2Fviscereality.org).
   Sign in, keep the **Free** plan, and enter the private key as
   `LINK_RELAY_TOKEN`. The Blueprint sets the start command and health check.
3. When Render reports **Live**, copy its actual `https://…onrender.com` URL.
   In the folder, open **Upload-Dienst einrichten** and enter that URL and the
   same access key. The app clears both fields when you close the folder.
4. Choose **Neue Dateien senden** or **Alle Dateien senden**, then confirm the
   password prompt. The app uploads the encrypted ZIP and posts the URL to
   Discord in the same operation.

[Render Free](https://render.com/docs/free) sleeps after 15 minutes idle and
can take about a minute to wake. The browser waits up to two minutes using a
credential-free `/health` probe before it starts uploading. Cancellation stops
subsequent browser calls; an already dispatched provider request can finish.
Render's [outbound usage limits](https://render.com/docs/outbound-bandwidth) apply.
Use a workspace without a payment method to avoid bandwidth overage charges;
Render then pauses service if the included allowance is exhausted. Automatic
deploys are off; manually redeploy after updating the relay.

The Node adapter and live Wormhole integration have been tested locally.
Render hosting still requires a signed-in account and a deployed service URL;
GitHub Pages publication does not provision or verify that service. Existing
compatible upload endpoints remain supported.

Automatic upload encrypts the ZIP on-device with pinned MIT
`wormhole-crypto@0.3.1`. Two passes over the immutable Blob hash and upload
identical encrypted 8 MiB pieces. See [vendor provenance](../vendor/wormhole-crypto/README.md).
Wormhole controls expiry, storage and download limits. Interrupted uploads can
leave unused rooms.

Recording, participant identity and retention are governed by
[DATA-LOGGING.md](../for-ai/DATA-LOGGING.md). Test results are described in
[VERIFICATION.md](../for-ai/VERIFICATION.md). The Wormhole integration may need
changes if the provider changes its upload API.
