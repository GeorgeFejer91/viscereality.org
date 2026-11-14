package org.viscereality.atmungsraum.discovery;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

public final class StartAfterBoot extends BroadcastReceiver {
    @Override public void onReceive(Context context, Intent intent) {
        if (Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction()) && SetupActivity.permitted(context)) {
            context.startForegroundService(new Intent(context, DiscoveryService.class));
        }
    }
}
