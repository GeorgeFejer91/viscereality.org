# Atmungsraum access and file delivery

Atmungsraum, Quest and [local CSV recovery](https://viscereality.org/atmungsraum/#daten)
open directly. The shared password decrypts the webhook envelope on the device
when an operator chooses to send files. Supported browsers can save the password
and silently restore it on a later send. Decrypted values stay in tab memory.
GitHub Pages serves the app and its encrypted envelope.

The folder offers **Ungesendete Daten senden** and **Alle Daten senden**.
The GitHub Pages-only release reports the upload limitation when a send is
attempted. It leaves the saved CSVs and receipt ledger intact.

## Browser upload boundary

The phone creates the ZIP and starts the request. A mobile browser or
installed home-screen app still sends the Atmungsraum website's origin.
Wormhole's room API rejects that origin's upload preflight. Its framing
and opener policies also isolate its controls from Atmungsraum.

Wormhole's [manifest](https://wormhole.app/site.webmanifest) registers a
file share target. Its [service worker](https://wormhole.app/service-worker.js)
receives the multipart POST and passes the files to its own page. The user
selects Wormhole in the system share sheet. The
[Web Share API](https://www.w3.org/TR/web-share/#share-method) reports whether
the share was handed off and provides no return value from the target. The
Wormhole worker provides no callback that returns the generated link.

## Verified relay fixture

The repository retains an upload-only adapter for testing. A Chromium fixture
from the deployed page used the local adapter and real Wormhole/Backblaze.
It restored the exact 9,800,146-byte synthetic ZIP. The browser checked
Discord's payload and simulated acknowledgement without posting to a channel.
This fixture does not activate a hosted relay.

In that flow the phone encrypts the ZIP using pinned MIT
[`wormhole-crypto@0.3.1`](../vendor/wormhole-crypto/README.md). The relay
forwards ciphertext pieces of up to 8 MiB. After Wormhole acknowledges the
upload, the browser sends the complete keyed URL to Discord. A local receipt
records files only after Discord acknowledges the post.

See [DATA-LOGGING.md](../for-ai/DATA-LOGGING.md) for the data contract and
[PROJECT.md](../for-ai/PROJECT.md) for the current constraint.
