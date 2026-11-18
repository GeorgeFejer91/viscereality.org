package org.viscereality.atmungsraum.discovery;

import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.json.JSONException;
import org.json.JSONObject;

/** Proof broker for one commissioned Quest. The companion never receives an RQG1 member key. */
final class CommissioningProof {
    private final String device;
    private final String secret;
    private JSONObject pending;
    private String transcript;
    private long deadline;

    CommissioningProof(String device, String secret) {
        if (!device.matches("[a-f0-9]{32}") || !secret.matches("[a-f0-9]{64}")) {
            throw new IllegalArgumentException("invalid commissioning profile");
        }
        this.device = device;
        this.secret = secret;
    }

    boolean matches(String selectedDevice, String selectedSecret) {
        return device.equals(selectedDevice) && equal(secret, selectedSecret);
    }

    JSONObject identity() throws JSONException {
        return new JSONObject().put("v", 1).put("device", device).put("serial", "")
            .put("identity_kind", "app-installation");
    }

    JSONObject authorize(JSONObject offer, JSONObject metadata, long now) throws JSONException, GeneralSecurityException {
        if (metadata == null || !"remote".equals(metadata.opt("mode"))
                || !"app-installation".equals(metadata.opt("identity_kind")) || !device.equals(metadata.opt("device"))
                || !device.equals(offer.opt("device")) || offer.length() != 6 || metadata.optJSONObject("rtc") == null) {
            throw new SecurityException("commissioning discovery unavailable");
        }
        String room = field(offer, "room", "[a-f0-9]{32}");
        String epoch = field(offer, "epoch", "[1-9][0-9]{0,18}");
        String nativeEpoch = field(offer, "native_epoch", "[1-9][0-9]{0,18}");
        String id = field(offer, "id", "[a-f0-9]{32}");
        String publicKey = field(offer, "public_key", "[A-Za-z0-9_-]{122}");
        if (!room.equals(metadata.optJSONObject("rtc").opt("r")) || !epoch.equals(metadata.opt("epoch"))
                || !nativeEpoch.equals(metadata.opt("native_epoch"))) {
            throw new SecurityException("commissioning owner mismatch");
        }
        String binding = "ATMB1|RQG1-RTC1|" + device + "|" + epoch + "|" + nativeEpoch + "|" + room;
        String tag = sign(binding + "|member");
        transcript = binding + "|request|" + id + "|" + tag + "|" + publicKey;
        pending = new JSONObject().put("v", 2).put("id", id).put("tag", tag)
            .put("public_key", publicKey).put("proof", sign(transcript));
        deadline = now + 15000;
        return pending;
    }

    boolean verify(JSONObject request, JSONObject reply, long now) throws JSONException, GeneralSecurityException {
        if (pending == null || now >= deadline || request.length() != 5 || reply.length() != 7
                || !Integer.valueOf(2).equals(request.opt("v")) || !Integer.valueOf(2).equals(reply.opt("v"))) return false;
        for (String name : new String[]{"id", "tag", "public_key", "proof"}) {
            if (!pending.getString(name).equals(request.opt(name))) return false;
        }
        if (!pending.getString("id").equals(reply.opt("id")) || !pending.getString("tag").equals(reply.opt("tag"))) return false;
        String key = field(reply, "public_key", "[A-Za-z0-9_-]{122}");
        String iv = field(reply, "iv", "[A-Za-z0-9_-]{16}");
        String data = field(reply, "data", "[A-Za-z0-9_-]{24,512}");
        String supplied = field(reply, "proof", "[a-f0-9]{64}");
        if (!equal(sign(transcript + "|reply|" + key + "|" + iv + "|" + data), supplied)) return false;
        pending = null;
        transcript = null;
        return true;
    }

    private static String field(JSONObject value, String name, String pattern) {
        Object field = value.opt(name);
        if (!(field instanceof String) || !((String)field).matches(pattern)) {
            throw new SecurityException("invalid commissioning field");
        }
        return (String)field;
    }

    private String sign(String text) throws GeneralSecurityException {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.US_ASCII), "HmacSHA256"));
        byte[] signature = mac.doFinal(text.getBytes(StandardCharsets.US_ASCII));
        StringBuilder hex = new StringBuilder(signature.length * 2);
        for (byte value : signature) hex.append(Character.forDigit((value & 255) >>> 4, 16))
            .append(Character.forDigit(value & 15, 16));
        return hex.toString();
    }

    private static boolean equal(String left, String right) {
        return MessageDigest.isEqual(left.getBytes(StandardCharsets.US_ASCII), right.getBytes(StandardCharsets.US_ASCII));
    }
}
