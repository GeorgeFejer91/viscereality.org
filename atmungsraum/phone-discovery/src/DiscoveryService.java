package org.viscereality.atmungsraum.discovery;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothManager;
import android.content.Intent;
import android.content.RestrictionsManager;
import android.os.Bundle;
import android.os.IBinder;
import android.os.SystemClock;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONException;
import org.json.JSONObject;

public final class DiscoveryService extends Service {
    private static final String CHANNEL = "quest-discovery";
    private static final String PATH = "/v1/quest/";
    private static final String COMMISSION = "/v1/commissioning";
    private static final String ALLOWED_ORIGIN = "https://viscereality.org";
    private final ExecutorService requests = Executors.newSingleThreadExecutor();
    private final ExecutorService discovery = Executors.newSingleThreadExecutor();
    private volatile ServerSocket listener;
    private volatile boolean stopped;
    private BluetoothAdapter adapter;
    private boolean scanning;
    private JSONObject result;
    private String resultId;
    private long nextScan;
    private CommissioningProof commissioning;

    @Override public void onCreate() {
        super.onCreate();
        NotificationManager manager = getSystemService(NotificationManager.class);
        manager.createNotificationChannel(new NotificationChannel(CHANNEL, "Quest-Suche", NotificationManager.IMPORTANCE_LOW));
        startForeground(1, new Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.stat_sys_data_bluetooth)
            .setContentTitle("Atmungsraum Bluetooth")
            .setContentText("Findet die gespeicherte Quest bei Bedarf.")
            .setOngoing(true).build());
        adapter = getSystemService(BluetoothManager.class).getAdapter();
        requests.execute(this::listen);
    }

    @Override public IBinder onBind(Intent intent) { return null; }
    @Override public int onStartCommand(Intent intent, int flags, int startId) { return START_STICKY; }

    private void listen() {
        try (ServerSocket server = new ServerSocket()) {
            listener = server;
            if (stopped) return;
            server.bind(new InetSocketAddress(InetAddress.getByName("127.0.0.1"), 8767), 4);
            while (!stopped) {
                try (Socket socket = server.accept()) {
                    socket.setSoTimeout(750);
                    answer(socket);
                } catch (IOException | JSONException invalidRequest) {
                    if (server.isClosed()) break;
                }
            }
        } catch (IOException unavailable) {
            stopSelf();
        }
    }

    private void answer(Socket socket) throws IOException, JSONException {
        // Read bounded headers before dispatching a fixed loopback route.
        byte[] header = new byte[4096];
        int end = 0;
        int lastFour = 0;
        long deadline = SystemClock.elapsedRealtime() + 1000;
        while (end < header.length && lastFour != 0x0d0a0d0a && SystemClock.elapsedRealtime() < deadline) {
            int next = socket.getInputStream().read();
            if (next < 0) return;
            header[end++] = (byte) next;
            lastFour = (lastFour << 8) | next;
        }
        if (lastFour != 0x0d0a0d0a) return;
        String[] lines = new String(header, 0, end, StandardCharsets.US_ASCII).split("\r\n");
        String[] first = lines[0].split(" ");
        if (first.length != 3 || !"HTTP/1.1".equals(first[2])) return;
        String path = first[1];
        boolean rediscovery = path.matches(PATH + "[a-f0-9]{32}");
        boolean provisioning = COMMISSION.equals(path) || (COMMISSION + "/request").equals(path)
            || (COMMISSION + "/verify").equals(path);
        if (!rediscovery && !provisioning) return;
        int originHeaders = 0, lengthHeaders = 0, typeHeaders = 0, hostHeaders = 0;
        int length = 0;
        for (String line : lines) {
            int separator = line.indexOf(':');
            if (separator <= 0) continue;
            String value = line.substring(separator + 1).trim();
            switch (line.substring(0, separator).toLowerCase(java.util.Locale.ROOT)) {
                case "origin":
                    if (!ALLOWED_ORIGIN.equals(value)) return;
                    originHeaders++;
                    break;
                case "host":
                    if (provisioning && !"127.0.0.1:8767".equals(value)) return;
                    hostHeaders++;
                    break;
                case "content-length":
                    if (!value.matches("[0-9]{1,4}")) return;
                    length = Integer.parseInt(value);
                    lengthHeaders++;
                    break;
                case "content-type":
                    if (provisioning && !"application/json".equals(value)) return;
                    typeHeaders++;
                    break;
                case "transfer-encoding":
                    return;
                default:
            }
        }
        if (originHeaders != 1 || provisioning && hostHeaders != 1) return;
        int status = 200;
        JSONObject reply;
        if ("OPTIONS".equals(first[0])) reply = new JSONObject();
        else if (rediscovery && "GET".equals(first[0])) reply = status(path.substring(PATH.length()));
        else if (provisioning) {
            JSONObject body = null;
            if ("POST".equals(first[0])) {
                if (lengthHeaders != 1 || typeHeaders != 1 || length <= 0 || length > 2048) return;
                byte[] bytes = new byte[length];
                int received = 0;
                while (received < length && SystemClock.elapsedRealtime() < deadline) {
                    int count = socket.getInputStream().read(bytes, received, length - received);
                    if (count < 0) return;
                    received += count;
                }
                if (received != length) return;
                body = new JSONObject(new String(bytes, StandardCharsets.UTF_8));
            }
            try {
                reply = commission(first[0], path, body);
                if (reply == null) { status = 404; reply = new JSONObject(); }
            } catch (SecurityException | GeneralSecurityException denied) {
                status = 403;
                reply = new JSONObject();
            }
        } else return;
        byte[] value = reply.toString().getBytes(StandardCharsets.UTF_8);
        String statusLine = status == 200 ? "200 OK" : status == 404 ? "404 Not Found" : "403 Forbidden";
        String response = "HTTP/1.1 " + statusLine + "\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
            + "Access-Control-Allow-Origin: " + ALLOWED_ORIGIN + "\r\nVary: Origin\r\n"
            + "Access-Control-Allow-Methods: GET, POST\r\nAccess-Control-Allow-Headers: Content-Type\r\n"
            + "Connection: close\r\nContent-Length: " + value.length + "\r\n\r\n";
        socket.getOutputStream().write(response.getBytes(StandardCharsets.US_ASCII));
        socket.getOutputStream().write(value);
    }

    private CommissioningProof configuredCommissioning() {
        RestrictionsManager manager = getSystemService(RestrictionsManager.class);
        Bundle managed = manager == null ? null : manager.getApplicationRestrictions();
        String device = managed == null ? "" : managed.getString("atmungsraum_quest_device", "");
        String key = managed == null ? "" : managed.getString("atmungsraum_bootstrap_key", "");
        if (!device.matches("[a-f0-9]{32}") || !key.matches("[a-f0-9]{64}")) {
            commissioning = null;
        } else if (commissioning == null || !commissioning.matches(device, key)) {
            commissioning = new CommissioningProof(device, key);
        }
        return commissioning;
    }

    private synchronized JSONObject commission(String method, String path, JSONObject body)
            throws JSONException, GeneralSecurityException {
        CommissioningProof broker = configuredCommissioning();
        if (broker == null) return null;
        if ("GET".equals(method) && COMMISSION.equals(path)) return broker.identity();
        if (!"POST".equals(method) || body == null) throw new SecurityException("invalid commissioning route");
        long now = SystemClock.elapsedRealtime();
        if ((COMMISSION + "/request").equals(path)) {
            JSONObject fresh = now < nextScan ? result : null;
            return broker.authorize(body, fresh, now);
        }
        if (body.length() != 2) throw new SecurityException("invalid verification frame");
        return new JSONObject().put("verified",
            broker.verify(body.getJSONObject("request"), body.getJSONObject("reply"), now));
    }

    private synchronized JSONObject status(String device) throws JSONException {
        JSONObject reply = new JSONObject().put("v", 1);
        if (!SetupActivity.permitted(this)) return reply.put("state", "permission-required");
        if (adapter == null || !adapter.isEnabled()) {
            result = null;
            nextScan = 0;
            return reply.put("state", "bluetooth-off");
        }
        long now = SystemClock.elapsedRealtime();
        JSONObject cached = device.equals(resultId) && now < nextScan ? result : null;
        if (!scanning && (now >= nextScan || !device.equals(resultId))) {
            scanning = true;
            resultId = device;
            result = null;
            discovery.execute(() -> scan(device));
        }
        return cached == null ? reply.put("state", "searching") : reply.put("state", "ready").put("metadata", cached);
    }

    private void scan(String device) {
        JSONObject found = null;
        try {
            found = QuestScan.find(this, adapter, device);
        } catch (InterruptedException cancelled) {
            Thread.currentThread().interrupt();
        } catch (IllegalStateException | SecurityException unavailable) {
            // Bluetooth may be switched off or the OS grant removed while scanning.
        } finally {
            synchronized (this) {
                result = found;
                nextScan = SystemClock.elapsedRealtime() + (found == null ? 2500 : 12000);
                scanning = false;
            }
        }
    }

    @Override public synchronized void onDestroy() {
        stopped = true;
        try {
            if (listener != null) listener.close();
        } catch (IOException closed) {
            // The listener is already ending.
        }
        requests.shutdownNow();
        discovery.shutdownNow();
        super.onDestroy();
    }
}
