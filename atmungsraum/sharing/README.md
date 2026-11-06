# Automatic Wormhole upload and Discord sharing

The folder dialog can package every saved CSV into `Atmungsraum-Daten.zip`,
encrypt it on the device, upload it to Wormhole, show its complete link/QR and
automatically send that link to a fixed Discord webhook. The ZIP preserves the
existing CSV bytes and names. A running recording appears only after it ends.

GitHub Pages serves the app. A small Cloudflare Worker handles Wormhole's
cross-origin upload restriction and holds the Discord webhook privately.
You do not need to operate a server, bot, database or email service. The Worker
uses ordinary outbound HTTPS requests and needs no paid binding. Current
[Workers Free limits](https://developers.cloudflare.com/workers/platform/limits/)
apply; do not enable a paid plan for this setup.

## Finish this installation

The supplied webhook and a generated access key are already in this machine's
ignored `sharing/.dev.vars`. Neither belongs in Git, public HTML or a screenshot.
The private `.dev.vars.setup.html` created during setup contains the connection
details and, when applicable, the Cloudflare preview claim link.

1. **Keep the Cloudflare deployment.** If setup supplied a preview claim link,
   open it, sign in and finish the claim **within 60 minutes of creation**.
   Opening the page alone does not keep it. Cloudflare deletes unclaimed
   previews. See [Cloudflare's claim instructions](https://developers.cloudflare.com/workers/platform/claim-deployments/).
2. **Connect the app.** Open [Atmungsraum](https://viscereality.org/atmungsraum/)
   → folder → **Per QR-Code teilen …** → **Automatisches Teilen einrichten**.
   Copy the Worker URL and private access key from the local setup page.
   Keep **Neue Links automatisch weiterleiten** checked.
3. **Share.** Press **ZIP automatisch hochladen**. Wait for both upload
   confirmation and Discord acceptance; check the receiving channel. The link
   and QR appear in the same dialog. Anyone with the complete link/key can
   download the data while Wormhole permits it.

The key and Worker URL are held only in the open dialog and cleared on close
or collapse. Reenter them for a later sharing operation. A confirmed uploaded
link is retained in page memory for the same exact saved files, so reopening
or retrying Discord does not upload another ZIP. Reloading the page loses this
receipt. New saved data requires a new upload. Existing shares are not revoked.

If Discord is not confirmed, the link/QR and downloads remain available. Check
the channel before pressing **Link weiterleiten** again: a timeout can occur
after a provider accepted the message. There is no automatic retry or durable
exactly-once guarantee. Keep the page open during upload. Cancel/close stops
local work; it cannot undo a request already accepted by a provider.

## Deploy or redeploy in your own account

From `atmungsraum/sharing/`, after installing Node.js and pnpm:

```powershell
pnpm dlx wrangler@4.158.0 login
pnpm dlx wrangler@4.158.0 deploy --secrets-file .dev.vars
```

Use the returned `https://atmungsraum-sharing.…workers.dev` URL in the app.
This deploys only the small Worker; the website remains on GitHub Pages.
If the preview expired, these commands create a permanent deployment in the
account you select. A claimed preview requires normal login for later updates.
If you have several accounts, choose the intended account in Wrangler.

For a fresh checkout, create the ignored `.dev.vars` locally:

```dotenv
DISCORD_WEBHOOK_URL=<your complete Discord webhook URL>
LINK_RELAY_TOKEN=<a random base64url access key of 32–128 characters>
```

Generate a key locally with
`node -e "console.log(require('node:crypto').randomBytes(32).toString('base64url'))"`.
Keep `.dev.vars` trusted; Wrangler expands dotenv variables. The checked-in
`wrangler.jsonc` sets `DESTINATION=discord` and
`ALLOWED_ORIGIN=https://viscereality.org`. An origin has no page path.
The [secrets-file flag](https://developers.cloudflare.com/workers/wrangler/commands/workers/)
uploads values as private Worker secrets. Redeploy after rotating either key.
The browser receives only the operator access key, never the webhook.

## Data and protocol boundary

- The browser uses pinned official MIT `wormhole-crypto@0.3.1` and Web Crypto.
  File encryption uses a fresh random key/salt per ZIP. Two passes over the
  same immutable Blob hash then upload identical encrypted 8 MiB pieces,
  without retaining a second whole encrypted archive. See
  [vendor provenance](../vendor/wormhole-crypto/README.md).
- The authenticated Worker permits only bounded Wormhole room/upload requests
  and Backblaze upload endpoints. It receives encrypted file bytes. The full
  download URL/key is sent to `/notify` only for the requested notification;
  the Worker and Discord can therefore access that share.
- The destination is fixed in private Worker configuration. Caller-supplied
  recipients are ignored. Origin restrictions supplement the access key;
  they do not replace authentication. No general CORS proxy is provided.
- The sample disables Worker observability. This code does not log request
  bodies, credentials, keyed URLs or provider exceptions. Preserve that policy.
- The Wormhole upload API is an integration with the current service, without
  a versioned public compatibility guarantee. Provider changes may require an
  update. Wormhole controls expiry, download counts and storage limits; the
  app does not invent availability guarantees. Large exports are also bounded
  by device browser memory and ZIP32. Interrupted uploads can leave an
  unused provider room until it expires.

Individual CSV downloads, native ZIP sharing and manual Wormhole upload →
paste link → local QR remain available without Cloudflare. QR generation uses
the local encoder and sends no link to a QR service. Recording, participant
identity, retention and file naming remain governed by
[DATA-LOGGING.md](../for-ai/DATA-LOGGING.md).

## Verification and remaining acceptance

The production upload client and Worker handler completed a real synthetic
Wormhole/Backblaze upload, independent download/decryption with byte-identical
ZIP comparison, then an authorized message to the supplied Discord webhook
with successful message readback. That first network check ran the Worker
handler locally. Cloudflare deployment and browser/public-site evidence are
recorded separately in [VERIFICATION.md](../for-ai/VERIFICATION.md).
No participant data was used. Physical phone/Safari behavior and durable
account ownership require their own acceptance.

## Other destinations

The Worker also supports Telegram and email if you later change destination.
For Telegram set `DESTINATION=telegram`, private `TELEGRAM_BOT_TOKEN`, and
`TELEGRAM_CHAT_ID`; start the bot in the receiving chat or grant it channel
posting permission. For email set `DESTINATION=email`, add a Cloudflare Send
Email binding named `EMAIL`, onboard the sender domain, set `EMAIL_FROM` and
`EMAIL_TO`, and verify the receiving address. Free email is restricted to
verified destinations; arbitrary recipients require a paid Workers plan.
See [Cloudflare Email Service](https://developers.cloudflare.com/email-service/).
These alternative real-provider routes have not been configured or tested by
this Discord installation. No extra setup is needed for Discord.
