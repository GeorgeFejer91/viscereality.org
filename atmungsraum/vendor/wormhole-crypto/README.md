# Wormhole browser encryption

Pinned MIT sources from [Wormhole's encryption implementation](https://github.com/webtorrent/wormhole-crypto),
`wormhole-crypto@0.3.1`, and its dependency `base64-js@1.5.1`.
The adjacent LICENSE and BASE64-LICENSE retain their original notices.
`for-ai/scripts/vendor-wormhole.mjs` downloads exact npm tarballs, verifies
their SHA-512 integrity and copies only the runtime graph. No install scripts
run. Regenerate with `node for-ai/scripts/vendor-wormhole.mjs`.

The only adaptations are the keychain's local base64 import path and the
base64 dependency's CommonJS export lines converted to one ES-module export.
Encryption, key derivation, nonce handling and record framing are unchanged.
All modules are served locally and loaded on an explicit upload action.

The app uses the upstream `encryptStream` optional salt argument to make two
identical ciphertext passes over one immutable ZIP Blob: first to hash torrent
pieces, then to upload them. Each upload uses a new random key and file salt.
Never reuse that pair with different plaintext. The relay receives ciphertext
and encrypted metadata; its operator token is not sent to Wormhole/Backblaze.
