package org.viscereality.atmungsraum.discovery;

import java.nio.charset.StandardCharsets;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.json.JSONObject;

final class CommissioningProofTest {
    private static final String DEVICE = "2".repeat(32);
    private static final String SECRET = "3".repeat(64);
    private static final String ROOM = "a".repeat(32);

    private static void serve(String secret, String room) throws Exception {
        CommissioningProof broker = new CommissioningProof(DEVICE, secret);
        JSONObject owner = new JSONObject().put("device", DEVICE).put("epoch", "42").put("native_epoch", "7")
            .put("identity_kind", "app-installation").put("mode", "remote").put("rtc", new JSONObject().put("r", room));
        java.io.BufferedReader input = new java.io.BufferedReader(new java.io.InputStreamReader(System.in, StandardCharsets.UTF_8));
        for (String line; (line = input.readLine()) != null;) {
            JSONObject message = new JSONObject(line);
            JSONObject response = new JSONObject();
            long now = System.nanoTime() / 1000000;
            try {
                switch (message.getString("op")) {
                    case "identity":
                        response.put("result", broker.identity());
                        break;
                    case "request":
                        response.put("result", broker.authorize(message.getJSONObject("body"), owner, now));
                        break;
                    case "verify":
                        JSONObject body = message.getJSONObject("body");
                        response.put("result", new JSONObject().put("verified",
                            broker.verify(body.getJSONObject("request"), body.getJSONObject("reply"), now)));
                        break;
                    default:
                        throw new IllegalArgumentException("unexpected test operation");
                }
                response.put("ok", true);
            } catch (Exception denied) { response.put("ok", false); }
            System.out.println(response);
            System.out.flush();
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length == 3 && "--serve".equals(args[0])) { serve(args[1], args[2]); return; }
        CommissioningProof broker = new CommissioningProof(DEVICE, SECRET);
        JSONObject owner = new JSONObject().put("device", DEVICE).put("epoch", "42").put("native_epoch", "7")
            .put("identity_kind", "app-installation").put("mode", "remote")
            .put("rtc", new JSONObject().put("r", ROOM));
        JSONObject offer = new JSONObject().put("device", DEVICE).put("epoch", "42")
            .put("native_epoch", "7").put("room", ROOM).put("id", "1".repeat(32)).put("public_key", "a".repeat(122));
        expectDenied(() -> broker.authorize(new JSONObject(offer.toString()).put("room", "b".repeat(32)), owner, 100));
        expectDenied(() -> broker.authorize(new JSONObject(offer.toString()).put("native_epoch", "8"), owner, 100));
        expectDenied(() -> broker.authorize(new JSONObject(offer.toString()).put("action", "recordings"), owner, 100));
        JSONObject request = broker.authorize(offer, owner, 100);
        String prefix = "ATMB1|RQG1-RTC1|" + DEVICE + "|42|7|" + ROOM;
        if (!request.getString("tag").equals(sign(prefix + "|member"))) throw new AssertionError("room binding changed");
        if (request.has("key") || request.has("device")) throw new AssertionError("private profile escaped");
        String transcript = prefix + "|request|" + request.getString("id") + "|" + request.getString("tag")
            + "|" + request.getString("public_key");
        JSONObject reply = new JSONObject().put("v", 2).put("id", request.getString("id"))
            .put("tag", request.getString("tag")).put("public_key", "b".repeat(122))
            .put("iv", "c".repeat(16)).put("data", "d".repeat(192));
        reply.put("proof", sign(transcript + "|reply|" + reply.getString("public_key") + "|" + reply.getString("iv")
            + "|" + reply.getString("data")));
        if (broker.verify(request, new JSONObject(reply.toString()).put("proof", "0".repeat(64)), 101))
            throw new AssertionError("forged Quest identity accepted");
        if (!broker.verify(request, reply, 102)) throw new AssertionError("real owner proof rejected");
        if (broker.verify(request, reply, 103)) throw new AssertionError("local verification replay accepted");
        request = broker.authorize(offer, owner, 110);
        if (broker.verify(request, reply, 15111)) throw new AssertionError("stale local offer accepted");
        System.out.println("PASS companion commissioning bound to device, room, owner epoch, ephemeral keys and one-use verification");
    }

    private static String sign(String text) throws Exception {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(SECRET.getBytes(StandardCharsets.US_ASCII), "HmacSHA256"));
        StringBuilder hex = new StringBuilder();
        for(byte b : mac.doFinal(text.getBytes(StandardCharsets.US_ASCII))) hex.append(String.format("%02x", b & 255));
        return hex.toString();
    }
    private interface Trial { void run() throws Exception; }
    private static void expectDenied(Trial trial) throws Exception {
        try { trial.run(); throw new AssertionError("invalid offer was admitted"); }
        catch (SecurityException expected) {}
    }
}
