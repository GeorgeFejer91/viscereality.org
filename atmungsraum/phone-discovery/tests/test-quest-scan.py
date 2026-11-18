
"""Exercise the production scanner against a bounded, synthetic GATT link."""
import argparse
from pathlib import Path
import os
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--java-home", type=Path, required=True)
parser.add_argument("--json-jar", type=Path, required=True)
args = parser.parse_args()
java = args.java_home.resolve(strict=True) / "bin"
jar = args.json_jar.resolve(strict=True)
project = Path(__file__).resolve().parents[1]

stubs = {
    "android/content/Context": "public class Context {}",
    "android/os/ParcelUuid": "public class ParcelUuid { public ParcelUuid(java.util.UUID ignored) {} }",
    "android/os/SystemClock": """
        public class SystemClock {
            private static long now;
            public static long elapsedRealtime() { return now += 1000; }
        }""",
    "android/bluetooth/BluetoothProfile": "public interface BluetoothProfile { int STATE_CONNECTED = 2; }",
    "android/bluetooth/BluetoothGattCharacteristic": """
        public class BluetoothGattCharacteristic {
            final int handle;
            final int status;
            byte[] value;
            public BluetoothGattCharacteristic(int handle, String value, int status) {
                this.handle = handle;
                this.value = value.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                this.status = status;
            }
            public int getInstanceId() { return handle; }
            public byte[] getValue() { return value; }
            public boolean setValue(byte[] next) { value = next; return true; }
        }""",
    "android/bluetooth/BluetoothGattService": """
        public class BluetoothGattService {
            final java.util.UUID id;
            final BluetoothGattCharacteristic metadata;
            public BluetoothGattService(java.util.UUID id, BluetoothGattCharacteristic metadata) {
                this.id = id;
                this.metadata = metadata;
            }
            public java.util.UUID getUuid() { return id; }
            public BluetoothGattCharacteristic getCharacteristic(java.util.UUID ignored) { return metadata; }
        }""",
    "android/bluetooth/BluetoothGattCallback": """
        public class BluetoothGattCallback {
            public void onConnectionStateChange(BluetoothGatt gatt, int status, int state) {}
            public void onServicesDiscovered(BluetoothGatt gatt, int status) {}
            public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic value, int status) {}
            public void onCharacteristicRead(BluetoothGatt gatt, BluetoothGattCharacteristic value, byte[] data, int status) {
                onCharacteristicRead(gatt, value, status);
            }
        }""",
    "android/bluetooth/BluetoothGatt": """
        public class BluetoothGatt {
            public static final int GATT_SUCCESS = 0;
            public final java.util.List<Integer> reads = new java.util.ArrayList<>();
            public boolean closed;
            final java.util.List<BluetoothGattService> services;
            final boolean modern;
            final java.util.Queue<BluetoothGattCharacteristic> pending = new java.util.ArrayDeque<>();
            BluetoothGattCallback callback;
            public BluetoothGatt(java.util.List<BluetoothGattService> services, boolean modern) {
                this.services = services;
                this.modern = modern;
            }
            public boolean discoverServices() { callback.onServicesDiscovered(this, 0); return true; }
            public java.util.List<BluetoothGattService> getServices() { return services; }
            public BluetoothGattService getService(java.util.UUID id) {
                return services.stream().filter(service -> id.equals(service.getUuid())).findFirst().orElse(null);
            }
            public boolean readCharacteristic(BluetoothGattCharacteristic value) {
                if (!pending.isEmpty()) throw new AssertionError("overlapping metadata reads");
                reads.add(value.getInstanceId());
                pending.add(value);
                return true;
            }
            void deliver() {
                while (!pending.isEmpty()) {
                    BluetoothGattCharacteristic value = pending.remove();
                    if (modern) callback.onCharacteristicRead(this, value, value.getValue(), value.status);
                    else callback.onCharacteristicRead(this, value, value.status);
                }
            }
            public void close() { closed = true; }
        }""",
    "android/bluetooth/BluetoothDevice": """
        public class BluetoothDevice {
            public static final int TRANSPORT_LE = 2;
            final BluetoothGatt gatt;
            public BluetoothDevice(BluetoothGatt gatt) { this.gatt = gatt; }
            public BluetoothGatt connectGatt(android.content.Context ignored, boolean auto, BluetoothGattCallback callback, int transport) {
                gatt.callback = callback;
                callback.onConnectionStateChange(gatt, 0, BluetoothProfile.STATE_CONNECTED);
                gatt.deliver();
                return gatt;
            }
        }""",
    "android/bluetooth/BluetoothAdapter": """
        public class BluetoothAdapter {
            final android.bluetooth.le.BluetoothLeScanner scanner;
            public BluetoothAdapter(android.bluetooth.le.BluetoothLeScanner scanner) { this.scanner = scanner; }
            public android.bluetooth.le.BluetoothLeScanner getBluetoothLeScanner() { return scanner; }
            public boolean isEnabled() { return true; }
        }""",
    "android/bluetooth/le/ScanResult": """
        public class ScanResult {
            final android.bluetooth.BluetoothDevice device;
            public ScanResult(android.bluetooth.BluetoothDevice device) { this.device = device; }
            public android.bluetooth.BluetoothDevice getDevice() { return device; }
        }""",
    "android/bluetooth/le/ScanCallback": "public class ScanCallback { public void onScanResult(int type, ScanResult result) {} }",
    "android/bluetooth/le/ScanFilter": """
        public class ScanFilter {
            public static class Builder {
                public Builder setServiceUuid(android.os.ParcelUuid id) { return this; }
                public ScanFilter build() { return new ScanFilter(); }
            }
        }""",
    "android/bluetooth/le/ScanSettings": """
        public class ScanSettings {
            public static final int SCAN_MODE_LOW_LATENCY = 2;
            public static class Builder {
                public Builder setScanMode(int mode) { return this; }
                public ScanSettings build() { return new ScanSettings(); }
            }
        }""",
    "android/bluetooth/le/BluetoothLeScanner": """
        public class BluetoothLeScanner {
            public boolean stopped;
            final ScanResult fixture;
            public BluetoothLeScanner(ScanResult fixture) { this.fixture = fixture; }
            public void startScan(java.util.List<ScanFilter> filters, ScanSettings settings, ScanCallback callback) {
                callback.onScanResult(0, fixture);
            }
            public void stopScan(ScanCallback callback) { stopped = true; }
        }""",
}

main = """
package org.viscereality.atmungsraum.discovery;
import android.bluetooth.BluetoothAdapter;
import android.bluetooth.BluetoothDevice;
import android.bluetooth.BluetoothGatt;
import android.bluetooth.BluetoothGattCharacteristic;
import android.bluetooth.BluetoothGattService;
import android.bluetooth.le.BluetoothLeScanner;
import android.bluetooth.le.ScanResult;
import java.util.List;
import java.util.UUID;
import org.json.JSONObject;

public class QuestScanFixture {
    static final UUID SERVICE = UUID.fromString("9a7b2001-7d6a-4b7f-9d4a-6f7c0a020101");
    static BluetoothGattService entry(int handle, String data, int status) {
        return new BluetoothGattService(SERVICE, new BluetoothGattCharacteristic(handle, data, status));
    }
    static String metadata(String device, String mode) {
        return "{\\"device\\":\\"" + device + "\\",\\"mode\\":\\"" + mode + "\\"}";
    }
    static void require(boolean value, String caseName) {
        if (!value) throw new AssertionError(caseName);
    }
    static void run(String name, List<BluetoothGattService> services, List<Integer> readExpected, boolean match, boolean modern)
            throws Exception {
        BluetoothGatt gatt = new BluetoothGatt(services, modern);
        BluetoothLeScanner scanner = new BluetoothLeScanner(new ScanResult(new BluetoothDevice(gatt)));
        BluetoothAdapter adapter = new BluetoothAdapter(scanner) {
            int checks;
            @Override public boolean isEnabled() { return checks++ == 0; }
        };
        JSONObject found = QuestScan.find(null, adapter, "enrolled");
        require((found != null) == match, name + ": unexpected metadata result");
        require(gatt.reads.equals(readExpected), name + ": wrong GATT read order " + gatt.reads);
        require(gatt.closed && scanner.stopped, name + ": cleanup");
    }
    public static void main(String[] args) throws Exception {
        var enrolled = entry(2, metadata("enrolled", "remote"), 0);
        run("second registered app owns requested identity", List.of(entry(1, metadata("foreign", "remote"), 0), enrolled), List.of(1, 2), true, true);
        run("foreign and ineligible services skipped", List.of(entry(1, metadata("enrolled", "direct"), 0), enrolled), List.of(1, 2), true, false);
        run("malformed and failed reads skipped", List.of(entry(1, "broken", 0), entry(3, "", 13), enrolled), List.of(1, 3, 2), true, true);
        run("wrong session rejected", List.of(entry(1, metadata("wrong", "remote"), 0)), List.of(1), false, false);
        run("one matching instance stops reads", List.of(enrolled, entry(1, metadata("wrong", "remote"), 0)), List.of(2), true, false);
        System.out.println("PASS 5 selected production scan callback cases");
    }
}
"""

with tempfile.TemporaryDirectory(prefix="atmungsraum-ble-instance-test-") as scratch:
    folder = Path(scratch)
    sources = []
    for name, body in stubs.items():
        file = folder / (name + ".java")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("package " + name.rsplit("/",1)[0].replace("/",".") + ";\n" + body, encoding="utf-8")
        sources.append(str(file))
    fixture = folder / "QuestScanFixture.java"
    fixture.write_text(main, encoding="utf-8")
    sources += [str(fixture), str(project / "src" / "QuestScan.java")]
    subprocess.run([str(java / "javac"), "--release", "11", "-cp", str(jar), "-d", str(folder), *sources], check=True)
    subprocess.run([str(java / "java"), "-cp", os.pathsep.join((str(folder),str(jar))),
                    "org.viscereality.atmungsraum.discovery.QuestScanFixture"], check=True)
