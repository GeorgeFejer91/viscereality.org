package org.viscereality.atmungsraum.boot;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

public final class BootReceiver extends BroadcastReceiver {
    @Override public void onReceive(Context context, Intent intent) {
        // No exported/custom broadcast entry. Android protects BOOT_COMPLETED
        // against app/shell injection; no extras are used as evidence of boot.
        if (intent == null || !Intent.ACTION_BOOT_COMPLETED.equals(intent.getAction())) return;
        AndroidHost host = new AndroidHost(context);
        try {
            String status = BootPolicy.run(host, intent.getAction(), host.currentBoot());
            Log.i("AtmungsraumBoot", "receiver status=" + status + " confirmed=false");
        } catch (RuntimeException failure) {
            Log.e("AtmungsraumBoot", "Boot attempt unavailable; confirmed=false", failure);
        }
    }
}
