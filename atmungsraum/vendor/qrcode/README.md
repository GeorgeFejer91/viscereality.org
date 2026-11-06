# Local QR encoder

Unmodified `dist/qrcode.mjs` from `qrcode-generator@2.0.4` (MIT), by Kazuhiko
Arase: <https://github.com/kazuhikoarase/qrcode-generator>.
Renamed to `qrcode.js` so static servers send the JavaScript module MIME type.
The adjacent LICENSE is from the upstream repository. No runtime network request is involved.

Downloaded from the npm registry and verified against its package integrity:
`sha512-mZSiP6RnbHl4xL2Ap5HfkjLnmxfKcPWpWe/c+5XxCuetEenqmNFf1FH/ftXPCtFG5/TDobjsjz6sSNL0Sr8Z9g==`.
The static site serves this module directly; there is no build/install step.
Update only for an actual compatibility/security need and recheck QR decoding.
