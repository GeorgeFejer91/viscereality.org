package org.viscereality.atmungsraum.discovery;

import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCallback;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.BluetoothProfile;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanCallback;
import android.bluetooth.le.ScanFilter;
import android.bluetooth.le.ScanResult;
import android.bluetooth.le.ScanSettings;
import android.content.Context;
import android.os.ParcelUuid;
import android.os.SystemClock;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.json.JSONObject;

final class QuestScan {
    private static final UUID SERVICE = UUID.fromString("9a7b2001-7d6a-4b7f-9d4a-6f7c0a020101");
    private static final UUID METADATA = UUID.fromString("9a7b2001-7d6a-4b7f-9d4a-6f7c0a020103");

    static JSONObject find(Context context, BluetoothAdapter adapter, String rememberedId) throws InterruptedException {
        BluetoothLeScanner scanner = adapter.getBluetoothLeScanner();
        if (scanner == null) return null;
        ArrayBlockingQueue<BluetoothDevice> devices = new ArrayBlockingQueue<>(12);
        ScanCallback callback = new ScanCallback() {
            @Override public void onScanResult(int type, ScanResult result) { devices.offer(result.getDevice()); }
        };
        ScanFilter filter = new ScanFilter.Builder().setServiceUuid(new ParcelUuid(SERVICE)).build();
        ScanSettings settings = new ScanSettings.Builder().setScanMode(ScanSettings.SCAN_MODE_LOW_LATENCY).build();
        Set<BluetoothDevice> tried = new HashSet<>();
        long deadline = SystemClock.elapsedRealtime() + 9000;
        scanner.startScan(List.of(filter), settings, callback);
        try {
            while (SystemClock.elapsedRealtime() < deadline && adapter.isEnabled()) {
                BluetoothDevice device = devices.poll(500, TimeUnit.MILLISECONDS);
                if (device == null || !tried.add(device)) continue;
                JSONObject metadata = readMetadata(context, device);
                if (metadata != null && rememberedId.equals(metadata.optString("device"))
                        && "remote".equals(metadata.optString("mode"))) return metadata;
            }
            return null;
        } finally {
            scanner.stopScan(callback);
        }
    }

    private static JSONObject readMetadata(Context context, BluetoothDevice device) throws InterruptedException {
        CountDownLatch finished = new CountDownLatch(1);
        BluetoothGattCallback callback = new BluetoothGattCallback() {
            @Override public void onConnectionStateChange(BluetoothGatt gatt, int status, int state) {
                if (status != BluetoothGatt.GATT_SUCCESS || state != BluetoothProfile.STATE_CONNECTED
                        || !gatt.discoverServices()) finished.countDown();
            }

            @Override public void onServicesDiscovered(BluetoothGatt gatt, int status) {
                BluetoothGattService service = status == BluetoothGatt.GATT_SUCCESS ? gatt.getService(SERVICE) : null;
                BluetoothGattCharacteristic metadata = service == null ? null : service.getCharacteristic(METADATA);
                if (metadata == null || !gatt.readCharacteristic(metadata)) finished.countDown();
            }

            @Override public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic,
                                                      int status) {
                if (status != BluetoothGatt.GATT_SUCCESS) characteristic.setValue(new byte[0]);
                finished.countDown();
            }
        };
        BluetoothGatt gatt = device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE);
        if (gatt == null) return null;
        try {
            if (!finished.await(3500, TimeUnit.MILLISECONDS)) return null;
            BluetoothGattService service = gatt.getService(SERVICE);
            BluetoothGattCharacteristic characteristic = service == null ? null : service.getCharacteristic(METADATA);
            byte[] data = characteristic == null ? null : characteristic.getValue();
            if (data == null || data.length == 0 || data.length > 512) return null;
            try {
                return new JSONObject(new String(data, StandardCharsets.UTF_8));
            } catch (org.json.JSONException invalid) {
                return null;
            }
        } finally {
            gatt.close();
        }
    }
}
