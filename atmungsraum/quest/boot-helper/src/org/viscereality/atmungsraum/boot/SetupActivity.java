package org.viscereality.atmungsraum.boot;

import android.app.Activity;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.provider.Settings;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

public final class SetupActivity extends Activity {
    @Override public void onCreate(Bundle state) { super.onCreate(state); }
    @Override public void onResume() { super.onResume();AndroidHost host=new AndroidHost(this);if(WatchPolicy.watching(host))WatchService.ensureRunning(this,false);render(); }
    private void render() {
        AndroidHost host = new AndroidHost(this);
        ScrollView scroll = new ScrollView(this);
        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        int pad = (int)(24 * getResources().getDisplayMetrics().density);
        body.setPadding(pad, pad, pad, pad);
        scroll.addView(body);
        TextView info = new TextView(this);
        info.setTextSize(19);
        String target;
        try {
            BootPolicy.Target found = host.resolveCurrentTarget();
            target = found == null ? "Not installed" : !found.sameSigner ? "Signing certificate mismatch" :
                found.activity.isEmpty() || !BootPolicy.TARGET_PACKAGE.equals(found.packageName) ? "No enabled MAIN/LAUNCHER activity" : "Ready: " + found.activity;
        } catch (RuntimeException failure) { target = "Status unavailable"; }
        info.setText("Atmungsraum Boot Helper\n\nTarget: " + BootPolicy.TARGET_PACKAGE +
            "\nTarget status: " + target + "\nOverlay permission: " + (host.overlayAllowed() ? "Allowed" : "Not allowed") +
            "\n\nEnable below and allow the overlay permission in Android settings. Compatible builds can request an update monitor that waits for the actual installed target and focused app acknowledgment. Long downloads do not expire. " +
            "Protected exit or Stop monitor cancels it. No download progress is inferred. A launch request does not prove the app opened.\n\n" +
            "Current boot: " + host.currentBoot() + "\nLast admitted boot: " + host.admittedBoot() +
            "\nLast status: " + host.state.getString("status", "No boot attempt") +
            "\nLast requested launcher: " + host.state.getString("launcher", "None") +
            "\nUpdate monitor: " + WatchPolicy.describe(host));
        body.addView(info);
        CheckBox enabled = new CheckBox(this);
        enabled.setText("Enable launch after restart");
        enabled.setChecked(host.enabled());
        enabled.setOnCheckedChangeListener((button, checked) -> {
            if (!host.setEnabled(checked)) {
                Toast.makeText(this, "Could not save setup", Toast.LENGTH_LONG).show(); render();
            } else if(!checked)stopService(new Intent(this,WatchService.class));
        });
        body.addView(enabled);
        Button stop=new Button(this);stop.setText("Stop update monitor");
        stop.setEnabled(WatchPolicy.watching(host));
        stop.setOnClickListener(view->{boolean saved=WatchPolicy.cancelInternal(host);stopService(new Intent(this,WatchService.class));
            host.record(saved?"MONITOR_CANCELLED_BY_OPERATOR":"MONITOR_CANCELLATION_NOT_SAVED");
            if(!saved)Toast.makeText(this,"Could not save cancellation. Check helper setup before restarting.",Toast.LENGTH_LONG).show();render();});
        body.addView(stop);
        Button overlay = new Button(this);
        overlay.setText("Open Android overlay settings");
        overlay.setOnClickListener(view -> {
            try { startActivity(new Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION, Uri.parse("package:" + getPackageName()))); }
            catch (RuntimeException failure) { Toast.makeText(this, "Android overlay settings unavailable", Toast.LENGTH_LONG).show(); }
        });
        body.addView(overlay);
        setContentView(scroll);
    }
}
