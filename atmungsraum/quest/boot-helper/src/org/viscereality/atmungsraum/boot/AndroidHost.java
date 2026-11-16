package org.viscereality.atmungsraum.boot;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.pm.ResolveInfo;
import android.provider.Settings;
import android.util.Log;

final class AndroidHost implements BootPolicy.Host {
    final Context context;
    final SharedPreferences state;
    private Intent currentIntent;
    AndroidHost(Context context) {
        this.context = context;
        state = context.getSharedPreferences("boot-helper", Context.MODE_PRIVATE);
    }
    public boolean enabled() { return state.getBoolean("enabled", false); }
    public boolean overlayAllowed() { return Settings.canDrawOverlays(context); }
    public long admittedBoot() { return state.getLong("admitted_boot", -1); }
    long currentBoot() { return Settings.Global.getInt(context.getContentResolver(), Settings.Global.BOOT_COUNT, -1); }
    public boolean admit(long boot) {
        return state.edit().putLong("admitted_boot", boot).putLong("attempt_time_ms", System.currentTimeMillis())
            .putString("status", "ADMITTED_UNCONFIRMED").remove("launcher").commit();
    }
    public BootPolicy.Target resolveCurrentTarget() {
        PackageManager pm = context.getPackageManager();
        try { pm.getApplicationInfo(BootPolicy.TARGET_PACKAGE, 0); }
        catch (PackageManager.NameNotFoundException absent) { return null; }
        // Use only MAIN+LAUNCHER, never getLaunchIntentForPackage's INFO fallback.
        Intent query = new Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_LAUNCHER)
            .setPackage(BootPolicy.TARGET_PACKAGE);
        ResolveInfo resolved = pm.resolveActivity(query, 0);
        ActivityInfo activity = resolved == null ? null : resolved.activityInfo;
        boolean signer = pm.checkSignatures(context.getPackageName(), BootPolicy.TARGET_PACKAGE) == PackageManager.SIGNATURE_MATCH;
        if (activity == null || !activity.enabled || !activity.exported || !activity.applicationInfo.enabled)
            return new BootPolicy.Target(BootPolicy.TARGET_PACKAGE, "", signer, true, true);
        currentIntent = new Intent(query).setComponent(new ComponentName(activity.packageName, activity.name))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED);
        return new BootPolicy.Target(activity.packageName, activity.name, signer,
            Intent.ACTION_MAIN.equals(currentIntent.getAction()), currentIntent.hasCategory(Intent.CATEGORY_LAUNCHER));
    }
    public void dispatch(BootPolicy.Target target) {
        // Recheck package/component and signature at dispatch; no Intent extras,
        // URLs, external package names, pinned activities or APK paths accepted.
        if (currentIntent == null || currentIntent.getComponent() == null ||
            !BootPolicy.TARGET_PACKAGE.equals(currentIntent.getComponent().getPackageName()) ||
            !target.activity.equals(currentIntent.getComponent().getClassName()) ||
            context.getPackageManager().checkSignatures(context.getPackageName(), BootPolicy.TARGET_PACKAGE) != PackageManager.SIGNATURE_MATCH)
            throw new SecurityException("Target identity changed");
        if (!state.edit().putString("launcher", currentIntent.getComponent().flattenToShortString()).commit())
            throw new IllegalStateException("Cannot persist launch request");
        context.startActivity(currentIntent);
    }
    public void record(String status) {
        boolean saved = state.edit().putString("status", status).commit();
        Log.i("AtmungsraumBoot", "boot=" + admittedBoot() + " status=" + status + " saved=" + saved + " confirmed=false");
    }
}
