# Password-protected Discord sharing

The website sends the complete Wormhole download link **directly to Discord**.
No Cloudflare account, relay, bot or server is needed for this notification.
The webhook is hosted as encrypted `discord-config.json` on GitHub Pages.
A masked password dialog unlocks it locally immediately before sending.
The ZIP stays on Wormhole; the Discord message contains its full keyed URL.

## Use it

1. Open the private local `.dev.vars.setup.html` to copy the generated password.
   This file and `.dev.vars.discord-password` are ignored by Git.
2. Open [Atmungsraum](https://viscereality.org/atmungsraum/) → folder →
   **Per QR-Code teilen …**. Upload the prepared ZIP on Wormhole and paste its
   complete link, including `#` and its download key. An already configured
   automatic Wormhole upload can also supply the link.
3. Choose **QR-Code anzeigen**. With automatic forwarding enabled, the password
   dialog appears. Otherwise press **An Discord senden**. Enter the password
   and press **Senden**. Check the receiving channel after Discord confirms it.
4. To reuse the password, select **Passwort im Browser merken** in a supported
   browser, then accept its own save prompt after a successful send. Later
   prompts can autofill the saved **Atmungsraum Discord** entry. The prompt
   still requires **Senden**. If the browser does not expose this save option,
   use its regular password-manager autofill or paste from the private page.

Cancel, Escape or an incorrect password sends nothing to Discord. The QR/link
and saved files remain available. The password field clears on dismissal;
the app does not write the password or unlocked webhook into localStorage,
sessionStorage or IndexedDB. When opted in, it asks the browser's
[credential manager](https://developer.mozilla.org/en-US/docs/Web/API/CredentialsContainer/store)
to save the password only after Discord accepts the message. Saving/autofill
availability and any sync are governed by browser settings. The password manager
can remove or update the entry; the app never stores the unlocked webhook.
Closing the files dialog cancels pending local work. A provider request already
accepted cannot be undone. Check Discord after a timeout before retrying;
there is no automatic retry or durable exactly-once guarantee.

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
the shared ZIP while Wormhole permits it. QR generation remains local.

## Optional automatic Wormhole upload

Wormhole's upload API currently blocks cross-origin requests from this static
site. Its fully automatic upload therefore still needs the separately configured
`worker.js` relay. **Discord notification does not use that relay.** Manual
Wormhole upload/paste/QR and direct Discord sending need no Cloudflare setup.

For the optional upload relay, retain private `LINK_RELAY_TOKEN` in `.dev.vars`
and the checked-in `ALLOWED_ORIGIN=https://viscereality.org`. From `sharing/`:

```powershell
pnpm dlx wrangler@4.158.0 login
pnpm dlx wrangler@4.158.0 deploy --secrets-file .dev.vars
```

Enter the returned Worker URL/access key in **Automatischen Wormhole-Upload
einrichten**. They clear when the files dialog closes. Cloudflare's temporary
preview must be claimed within60 minutes or it is deleted; see its
[claim instructions](https://developers.cloudflare.com/workers/platform/claim-deployments/).
The prior preview/account requirement is optional upload setup, not a requirement
for password-protected Discord delivery. No website migration or paid binding
is needed. The old Worker `/notify` endpoint remains available for existing
integrations, but this website's current Discord path calls Discord directly.

Automatic upload encrypts the existing ZIP on-device using pinned official MIT
`wormhole-crypto@0.3.1`. Two passes over the immutable Blob hash then upload
identical encrypted8 MiB pieces. Only after final upload acknowledgement does
the app display a link/QR and request the Discord password. See
[vendor provenance](../vendor/wormhole-crypto/README.md). A confirmed share remains
in page memory for the same exact records; retrying notification does not upload
again. Reload loses this receipt. New saved records require a new upload.
No existing share is revoked by changing the local list. Wormhole controls
expiry, storage and download limits. Interrupted uploads can leave unused rooms.

Individual downloads and native ZIP sharing remain available. Recording,
participant identity and retention are governed by
[DATA-LOGGING.md](../for-ai/DATA-LOGGING.md). No participant data is used in tests.
Current local/public-browser and real-provider receipts are described in
[VERIFICATION.md](../for-ai/VERIFICATION.md); physical phones/Safari require
separate qualification. The Wormhole service integration may need changes if
its non-versioned upload API changes.
