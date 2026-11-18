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
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.json.JSONException;
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
                JSONObject metadata = readMetadata(context, device, rememberedId);
                if (metadata != null) return metadata;
            }
            return null;
        } finally {
            scanner.stopScan(callback);
        }
    }

    private static JSONObject readMetadata(Context context, BluetoothDevice device, String rememberedId)
            throws InterruptedException {
        MetadataRead callback = new MetadataRead(rememberedId);
        BluetoothGatt gatt = device.connectGatt(context, false, callback, BluetoothDevice.TRANSPORT_LE);
        if (gatt == null) return null;
        try {
            return callback.finished.await(3500, TimeUnit.MILLISECONDS) ? callback.result : null;
        } finally {
            callback.finished.countDown();
            gatt.close();
        }
    }

    private static final class MetadataRead extends BluetoothGattCallback {
        private final String rememberedId;
        private final CountDownLatch finished = new CountDownLatch(1);
        private Iterator<BluetoothGattService> services;
        private BluetoothGattCharacteristic active;
        private JSONObject result;

        MetadataRead(String rememberedId) {
            this.rememberedId = rememberedId;
        }

        @Override public void onConnectionStateChange(BluetoothGatt gatt, int status, int state) {
            if (finished.getCount() == 0) return;
            if (status != BluetoothGatt.GATT_SUCCESS || state != BluetoothProfile.STATE_CONNECTED
                    || !gatt.discoverServices()) finished.countDown();
        }

        @Override public void onServicesDiscovered(BluetoothGatt gatt, int status) {
            if (status != BluetoothGatt.GATT_SUCCESS) {
                finished.countDown();
                return;
            }
            services = gatt.getServices().iterator();
            readNext(gatt);
        }

        private void readNext(BluetoothGatt gatt) {
            // Multiple apps on one headset can register the same discovery service.
            while (finished.getCount() != 0 && services.hasNext()) {
                BluetoothGattService service = services.next();
                if (!SERVICE.equals(service.getUuid())) continue;
                active = service.getCharacteristic(METADATA);
                if (active != null && gatt.readCharacteristic(active)) return;
            }
            finished.countDown();
        }

        @Override public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic,
                                                   int status) {
            onCharacteristicRead(gatt, characteristic, characteristic.getValue(), status);
        }

        @Override public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic characteristic,
                                                   byte[] value, int status) {
            if (finished.getCount() == 0 || active == null
                    || characteristic.getInstanceId() != active.getInstanceId()) return;
            if (status == BluetoothGatt.GATT_SUCCESS && value != null && value.length > 0 && value.length <= 512) {
                try {
                    JSONObject metadata = new JSONObject(new String(value, StandardCharsets.UTF_8));
                    if (rememberedId.equals(metadata.optString("device")) && "remote".equals(metadata.optString("mode"))) {
                        result = metadata;
                        finished.countDown();
                        return;
                    }
                } catch (JSONException invalid) {
                    // Advance to the next instance when its metadata cannot be used.
                }
            }
            readNext(gatt);
        }
    }
}
