package org.viscereality.atmungsraum.discovery;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;

public final class SetupActivity extends Activity {
    static boolean permitted(Context context) {
        return context.checkSelfPermission(Manifest.permission.BLUETOOTH_SCAN) == PackageManager.PERMISSION_GRANTED
            && context.checkSelfPermission(Manifest.permission.BLUETOOTH_CONNECT) == PackageManager.PERMISSION_GRANTED;
    }

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        if (permitted(this)) startDiscovery();
        else requestPermissions(new String[]{Manifest.permission.BLUETOOTH_SCAN, Manifest.permission.BLUETOOTH_CONNECT}, 1);
    }

    @Override public void onRequestPermissionsResult(int request, String[] permissions, int[] grants) {
        super.onRequestPermissionsResult(request, permissions, grants);
        if (permitted(this)) startDiscovery();
        else finish();
    }

    private void startDiscovery() {
        startForegroundService(new Intent(this, DiscoveryService.class));
        finish();
    }
}
