package org.viscereality.atmungsraum.discovery;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothManager;
import android.content.Intent;
import android.os.IBinder;
import android.os.SystemClock;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONException;
import org.json.JSONObject;

public final class DiscoveryService extends Service {
    private static final String CHANNEL = "quest-discovery";
    private static final String PATH = "/v1/quest/";
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
        // Read a bounded header block. This endpoint accepts only GET and CORS preflight.
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
        if (first.length != 3 || !first[1].matches(PATH + "[a-f0-9]{32}")) return;
        int originHeaders = 0;
        for (String line : lines) {
            int separator = line.indexOf(':');
            if (separator > 0 && "origin".equalsIgnoreCase(line.substring(0, separator))) {
                if (!ALLOWED_ORIGIN.equals(line.substring(separator + 1).trim())) return;
                originHeaders++;
            }
        }
        if (originHeaders != 1) return;
        String body;
        if ("OPTIONS".equals(first[0])) body = "{}";
        else if ("GET".equals(first[0])) body = status(first[1].substring(PATH.length())).toString();
        else return;
        byte[] value = body.getBytes(StandardCharsets.UTF_8);
        String response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nCache-Control: no-store\r\n"
            + "Access-Control-Allow-Origin: " + ALLOWED_ORIGIN + "\r\nVary: Origin\r\n"
            + "Access-Control-Allow-Methods: GET\r\nConnection: close\r\nContent-Length: " + value.length + "\r\n\r\n";
        socket.getOutputStream().write(response.getBytes(StandardCharsets.US_ASCII));
        socket.getOutputStream().write(value);
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
