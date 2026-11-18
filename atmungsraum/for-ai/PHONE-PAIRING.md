# Phone pairing and control

## Bluetooth bootstrap for independent installations

The user's 2026-11-18 requirement is one connection flow for every installed
Atmungsraum PWA on compatible smartphones and headsets of the same classes as
the lab devices. The user connects to the intended headset through Bluetooth.
Bluetooth supplies the selected installation identity and the bootstrap
information used to establish authenticated correspondence over Wi-Fi.
WebRTC with TURN supports routes where direct device-to-device Wi-Fi fails;
the normal flow requires no hotspot. Preserve the selected headset identity
through discovery, authentication, control and recording download.

Fresh and remembered connections must need no numeric pairing code or
device-specific APK. Platform Bluetooth/network permissions remain initial
setup steps. A release signing key identifies the app across its supported
devices. Each headset installation and authorized PWA membership retains
separate credentials, revocation and native ownership. Discovery or a
connected transport alone cannot grant experiment control. Keep the native
single-controller lease and receipt checks.

The existing v2 wire implementation is documented below. Its manual matching
code flow and the managed companion's one-Quest commissioning profile do not
establish general fresh-PWA, code-free acceptance. A production trust-bootstrap
path from the user's Bluetooth selection still needs implementation and
qualification without copied lab credentials or synthetic fixture APKs.
Use the [independent-installation matrix](VERIFICATION.md#independent-headset-and-pwa-installations)
and the active PROJECT.md checklist; keep this distinction explicit in
implementation and release claims.

## Existing v2 pairing implementation

User decisions, 2026-11-15: Remote advertises public read-only Bluetooth
discovery. A saved phone authenticates directly; a new phone requests pairing.
Both devices show a matching code and confirm using buttons, without headset
typing. Keep several independently revocable phone memberships and exactly one
active experiment controller. New-phone setup must not require a laptop, USB,
manually generated QR or entry of credentials. This is part of PROJECT.md's
active Quest acceptance work, not a separate lifecycle.

## Identity and transport

New opt-in `experiment_phone_pairing_enabled` defaults off for other products.
Selected public discovery/identity is v2 with `identity_kind: app-installation`.
`device` remains the stable random installation ID; `serial` is the actual
hardware serial when available and otherwise empty. Never substitute an app ID
for a hardware serial. V1 retains its nonempty hardware-serial contract.
Discovery metadata is `{v:2,identity_kind,device,serial,epoch,native_epoch,
mode:remote,gateways,pairing:1}`. Hardware identity provisioning remains a
separate optional machine-verification route.

Bluetooth supplies only discovery. Pairing uses the existing private-LAN HTTP
adapter with its peer, Origin, size, queue and timeout checks. The public
pairing operations grant no experiment commands, private data or member-list
access. Unknown users cannot claim active control.
Selected mode rejects public writes and queued effects through the legacy BLE
command service, including valid old pairing-code commands. Its internal
carrier/receipt journal and bounded nonprivate health reads remain available;
other unselected products keep their existing behavior.

The Developer app starts with its panel closed. Only a verified incoming
pairing request opens the wearer approval prompt. Allow/Reject, phone
cancellation and expiry restore that prompt's own opening; they cannot undo a
later manual visibility change. Triple-B still opens the regular app panel.

The selected native host coordinates the three existing startup permission
requesters through one attended runtime request. It asks only for needed,
declared permissions, then opens standard Location settings once if required.
The wearer may leave Location off without a repeated prompt loop; the app
never changes that system setting itself. The optional `y.l` reports enabled,
disabled, unavailable, legacy-unobserved, permission-blocked or unknown. Only
an unconnected Location-OFF phone flow blocks scanning; connected streams remain
usable. These OS surfaces remain separate from phone approval.

## Pairing wire contract

The pairing handshake is **v2 only**, with no fallback to the original
uncommitted v1 exchange. Discovery's `pairing:1` is a capability flag; the
ordinary authenticated command gateway remains a separate v1 protocol.
Pairing `v` is the exact JSON integer 2. `epoch` and `native_epoch` are canonical
positive decimal strings within signed 64-bit range. Every message has exactly
the documented fields. Public keys are canonical named-P-256, uncompressed-point
SPKI DER, encoded as unpadded canonical base64url (91 DER bytes). **Both nonces
are 16 random bytes encoded as base64url, exactly 22 characters**, never hex.
Pairing IDs are 16 random bytes encoded as 32 lower-case hex characters.
Commitments and HMAC proofs are 32 bytes encoded as 64 lower-case hex characters.
No private key leaves its side.

The ordered hash commitment follows the anti-grinding principle described in
[RFC 6189 section 4.4.1.1](https://www.rfc-editor.org/rfc/rfc6189.html#section-4.4.1.1)
and [section 7](https://www.rfc-editor.org/rfc/rfc6189.html#section-7).
This is a fixed-suite pairing protocol using that ordering, not ZRTP. An
uncommitted short code permits an intermediary to choose keys or nonce values
after learning both peers' contributions and search offline for matching codes.
Here the phone fixes and hides **both** its fresh public key and nonce before
the headset chooses its response; the headset freezes its own public key,
nonce and pairing ID before learning that opening. The phone pins its first
validated reply. Neither side may replace a contribution during a request.

Let `N = base64url(UTF8(phone_name))`, without normalization. Phone names are
nonempty, trimmed display labels of at most 64 UTF-16 code units / 128 UTF-8
bytes, without controls, format characters or unpaired surrogates. Names are
never authority claims. Encoding the name makes the following `|` delimiters
unambiguous. Define exact canonical strings:

```text
C = device|serial|identity_kind|epoch|native_epoch|N
commitment_input = RQ-PHONE-PAIR-2/commit|C|client_nonce|client_public_key
client_commitment = lower_hex(SHA256(UTF8(commitment_input)))
T = RQ-PHONE-PAIR-2|C|pairing_id|client_commitment|client_nonce|server_nonce|client_public_key|server_public_key
```

1. The phone generates its ephemeral P-256 key and nonce, computes the
   commitment, and sends exactly `{v:2,action:"pair-begin",device,serial,
   identity_kind,epoch,native_epoch,phone_name,client_commitment}`. It sends
   neither its key nor its nonce. The headset validates the complete owner
   context, stores the frozen commitment/name/context, then generates and
   freezes its own ephemeral key, nonce and ID. Reply:
   `{v:2,device,epoch,native_epoch,pairing_id,client_commitment,
   server_public_key,server_nonce,expires_in_ms}`. No matching code or headset
   prompt exists at this stage.
2. The phone validates identity/epochs, commitment echo, canonical key/nonce
   and expiry, pins the reply and derives the keys below using `T`. It sends
   exactly `{v:2,action:"pair-reveal",device,epoch,native_epoch,pairing_id,
   client_public_key,client_nonce,proof}`. The headset validates the canonical
   opening and equality with the **original** commitment, then derives the
   same keys and checks the action-specific proof. A mismatched opening or
   valid-shaped wrong reveal proof irreversibly fails the request. Early
   confirmation/status cannot bypass the reveal. Only a verified reveal
   enables the headset prompt and returns `{v:2,pairing_id,state:"pending"}`;
   only then may the phone call its matching-code UI. An exact duplicate
   reveal is idempotent and never resets the original deadline or SAS.
3. `pair-confirm`, `pair-status`, `pair-cancel` requests are exactly
   `{v:2,action,device,epoch,native_epoch,pairing_id,proof}`. Proof for every
   authenticated action, including reveal, is
   `lower_hex(HMAC-SHA256(PROOF, UTF8(T + "|" + action)))`. A reveal is
   possession proof, never either user's approval. Both actual buttons are
   still required before storage or a credential ticket.

HKDF-SHA256 uses salt `SHA256(UTF8(T))` and the 32-byte ECDH secret as input.
RFC 5869 extract/expand derives independent 32-byte keys with infos
`RQ-PHONE-PAIR-2/SAS`, `RQ-PHONE-PAIR-2/AEAD`, `RQ-PHONE-PAIR-2/PROOF`.
The displayed six digits are the unsigned big-endian first four SAS bytes
modulo 1,000,000, zero padded. Code comparison remains a human authentication
step, with a small per-attempt guessing probability; it is not cryptographic
proof by itself.

Status is exactly `{v:2,pairing_id,state}` and, only when approved, `iv` and
`data`. States are `pending`, `approved`, `cancelled`, `rejected`, `expired`,
`failed`; the internal pre-opening state is never a public prompt. Credentials
are AES-256-GCM encrypted with a fresh 12-byte IV, 128-bit tag and AAD
`UTF8("RQ-PHONE-PAIR-2/result|" + T)`. `iv` and ciphertext/tag `data` are
canonical base64url. Clear result contains exactly
`{v:2,controller,key,device,serial,identity_kind,epoch,native_epoch}`. The phone
checks every identity/epoch and saves that approved membership before joining,
so a lost subsequent gateway reply does not orphan the approved phone key.
It requires a matching authenticated gateway response before exposing control
or private data. A cancelled phone may send an authenticated cancel
after the reveal; an unopened commitment simply expires if the opening was
never delivered.

Require both confirmations before membership persists or a ticket is emitted.
Allow at most one in-progress commitment or pending request, with the same
120-second deadline starting at begin. Bound each live owner to four begin
attempts and 180 total exchanges per 60-second window and at most eight retained
results. Invalid requests also consume the applicable request budget. Reject
new pairing during a main recording or calibration.
Wrong keys/proofs, retirement, expiration, rejection, cancellation or malformed
requests cannot store membership. Keep approved-result retries idempotent until
expiry. No secret-bearing result in ordinary logs, public status or discovery.

## Membership and active control

Persist up to eight distinct members atomically with independent random keys
and bounded labels. Reconnecting never rotates unrelated members. Migrate a
valid existing legacy enrollment when this product opts in. A revoked phone
cannot reuse a cached protocol. Membership and native-owner retirement are
separate lifetimes.

Authenticated status reports `controller_access` with version, active/waiting
role, bounded active-phone label and monotonic revision. The first trusted
phone can claim an unowned/expired 15-second lease; only that phone's
authenticated status renews it. Other trusted phones wait. A release/claim
uses the enrolled `controller-access` action and closed payload
`{v:1,id,op,expected_control_revision}`, where op is `claim` or `release`.
The experiment's phone is the intended remote operator. After authenticated
connection or foreground resume, it automatically requests the existing native
claim for a free/expired lease. Several memberships remain saved and exactly
one native lease remains active. A live other owner is never silently replaced:
show its label and ask the operator to close that other app or release there.
Continue bounded status reads until native reports free control. No takeover
command exists in the current native contract. A disconnected controller does not
erase or finalize a running recording. Explicit handover retires deferred
mutations from the old controller; it does not replay them under a new owner.
Keep existing runtime/generation/replay fences in addition to the control lease.

Expose saved-phone removal and active-control release in the headset panel.
Phone UI shows native control in the top-row gamepad indicator, independently
of authenticated connection and Polar/input readiness. Green requires a fresh
active-role status readback; a claim response alone cannot grant green. Retire
green on backgrounding, failed status reads or expiry, charging transport age.
Show requesting, retrying, occupied, lost and unknown states in measured text
and the diagnostic popup. The former routine APK toolbar check is replaced;
actionable APK warnings and the update/identity page remain available.

Before each claim, read status and recheck the current gateway, visibility and
native barriers. Conclusive non-grants continue while this foreground phone
wants control and native reports a free lease: five-second spacing for the
first three requests, then thirty-second spacing. Continue this bounded cadence
during recording too, with fresh preflight and native admission on every request.
An occupied lease never receives automatic claims. A lost
or malformed reply stores the original ID/owner/revision in session storage
and permits only readback until native confirms active control, advances the
revision or retires that owner. There is no access-receipt query in this API;
do not invent one or replay an unknown request. Session/setup receipt journals
remain unchanged and can be recovered after free control is acquired.

Manual Release pauses automatic claims for this open app. Claim explicitly
requests control again without bypassing unknown fences or another live owner.
An authenticated fresh connection still defaults to automatic acquisition.
Read-only status and discovery remain separate from control authority.

## Verification

Require cross-runtime Java/WebCrypto interoperability, malformed/foreign key
and proof rejection, both confirmation orders, expiry/reject/cancel/retirement,
atomic persistence failure and capacity checks. Test independent membership,
revocation, stale protocol invalidation, one active writer, renewal, release,
reclaim, stale revision and deferred-start retirement. Verify actual package
closure/signature/installed bytes and cold start, real Bluetooth advertisement,
first phone pairing, remembered reconnect and the calibration flow on Quest 2.
Keep software, synthetic-browser and wearer acceptance evidence distinct.

The focused native gate `tools/checks/Test-ExperimentPhonePairing.ps1` accepts
`-WebTest <web-repo>/atmungsraum/tests/quest-pairing.test.mjs`. That mode compiles
the real Java owner and uses a synthetic stdin host to execute the browser's
unchanged WebCrypto begin/reveal/confirm/status/decrypt flow against it, in both
approval orders. It also tests mutated commitments/openings/proofs and ordering
before either UI, plus the browser's 8192-byte streaming response limit. The
Java test `--serve` harness and optional `--vector` output contain only synthetic
host data and are not packaged. A derived-key fixture alone is insufficient
wire interoperability evidence.

## Provisioned phones over WebRTC

A managed companion can recover a preauthorized existing membership using
a room-scoped ATMB1 request. The installation ID, gateway and native
epochs, room, request ID and ephemeral P-256 public key are bound to
the companion's HMAC. Quest selects the existing member by the
room-specific tag, verifies the proof and checks the shared replay
and retirement boundary. A TURN peer or bare discovery result has
no commissioning authority.

Quest encrypts the existing member credentials to the requesting
ephemeral key with ECDH, HKDF-SHA256 and AES-GCM. The companion verifies
a distinct reply HMAC over both public keys, the IV and the ciphertext
before the browser opens the ticket. This branch has no experiment
dispatch. The browser then enters the ordinary signed RQG1 status,
lease and receipt path. [Android commissioning procedure](../phone-discovery/README.md#managed-first-pairing)
