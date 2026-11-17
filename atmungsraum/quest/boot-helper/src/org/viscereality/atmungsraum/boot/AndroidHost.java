package org.viscereality.atmungsraum.boot;

import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.content.pm.PackageInfo;
import android.content.pm.Signature;
import android.os.SystemClock;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import android.content.pm.ResolveInfo;
import android.provider.Settings;
import android.util.Log;

final class AndroidHost implements BootPolicy.Host, WatchPolicy.Host {
    final Context context;
    final SharedPreferences state;
    private Intent currentIntent;
    AndroidHost(Context context) {
        this.context = context;
        state = context.getSharedPreferences("boot-helper", Context.MODE_PRIVATE);
    }
    public boolean enabled() { return state.getBoolean("enabled", false); }
    private static final String[] WATCH_FIELDS={"watch_format","watch_kind","watch_attempt","watch_installed","watch_target","watch_deadline","watch_detected","watch_not_before","watch_dispatched","watch_started"};
    public boolean setEnabled(boolean enabled) {
        synchronized(WatchPolicy.class) {
            SharedPreferences.Editor edit=state.edit().putBoolean("enabled",enabled);
            if(!enabled)for(String field:WATCH_FIELDS)edit.remove(field);
            return edit.commit();
        }
    }
    public boolean overlayAllowed() { return Settings.canDrawOverlays(context); }
    public long admittedBoot() { return state.getLong("admitted_boot", -1); }
    public long currentBoot() { return Settings.Global.getInt(context.getContentResolver(), Settings.Global.BOOT_COUNT, -1); }
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
        boolean signer = sameCurrentSigner(pm, context.getPackageName(), BootPolicy.TARGET_PACKAGE);
        long version=0; int protocol=0;
        try {
            PackageInfo info=pm.getPackageInfo(BootPolicy.TARGET_PACKAGE, PackageManager.GET_META_DATA);
            version=info.getLongVersionCode();
            if(info.applicationInfo.metaData!=null)protocol=info.applicationInfo.metaData.getInt("org.viscereality.atmungsraum.boot_protocol",0);
        } catch(PackageManager.NameNotFoundException absent) { return null; }
        if (activity == null || !activity.enabled || !activity.exported || !activity.applicationInfo.enabled)
            return new BootPolicy.Target(BootPolicy.TARGET_PACKAGE, "", signer, true, true);
        currentIntent = new Intent(query).setComponent(new ComponentName(activity.packageName, activity.name))
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_RESET_TASK_IF_NEEDED);
        return new BootPolicy.Target(activity.packageName, activity.name, signer,
            Intent.ACTION_MAIN.equals(currentIntent.getAction()), currentIntent.hasCategory(Intent.CATEGORY_LAUNCHER),version,protocol,
            activity.launchMode==ActivityInfo.LAUNCH_SINGLE_TASK);
    }
    public void dispatch(BootPolicy.Target target) {
        // Recheck package/component and signature at dispatch; no Intent extras,
        // URLs, external package names, pinned activities or APK paths accepted.
        if (currentIntent == null || currentIntent.getComponent() == null ||
            !BootPolicy.TARGET_PACKAGE.equals(currentIntent.getComponent().getPackageName()) ||
            !target.activity.equals(currentIntent.getComponent().getClassName()) ||
            !sameCurrentSigner(context.getPackageManager(), context.getPackageName(), BootPolicy.TARGET_PACKAGE))
            throw new SecurityException("Target identity changed");
        if (!state.edit().putString("launcher", currentIntent.getComponent().flattenToShortString()).commit())
            throw new IllegalStateException("Cannot persist launch request");
        if(target.protocol==2||target.protocol==3) {
            WatchPolicy.Result watch=WatchPolicy.armBoot(this,target);
            if(!watch.accepted||!WatchService.ensureRunning(context,false)) {
                WatchPolicy.cancelInternal(this);
                throw new IllegalStateException("Boot monitor unavailable");
            }
        }
        context.startActivity(currentIntent);
    }
    static boolean sameCurrentSigner(PackageManager pm,String first,String second) {
        try {
            Signature[] a=pm.getPackageInfo(first,PackageManager.GET_SIGNING_CERTIFICATES).signingInfo.getApkContentsSigners();
            Signature[] b=pm.getPackageInfo(second,PackageManager.GET_SIGNING_CERTIFICATES).signingInfo.getApkContentsSigners();
            return a!=null&&b!=null&&a.length>0&&new HashSet<Signature>(Arrays.asList(a)).equals(new HashSet<Signature>(Arrays.asList(b)));
        }catch(PackageManager.NameNotFoundException|RuntimeException unavailable){return false;}
    }
    public long now(){return SystemClock.elapsedRealtime();}
    public WatchPolicy.Ledger load() {
        long boot=state.getLong("watch_boot",-1);
        Set<String> attempts=new HashSet<String>(state.getStringSet("watch_attempts",java.util.Collections.<String>emptySet()));
        if(attempts.size()>WatchPolicy.MAX_ATTEMPTS)throw new IllegalStateException("Invalid attempt ledger");
        for(String id:attempts)if(!id.matches("[1-9][0-9]{0,18}")||Long.parseLong(id)<=0)throw new IllegalStateException("Invalid attempt");
        WatchPolicy.Watch watch=null;
        String kind=state.getString("watch_kind","");
        if(!kind.isEmpty()) {
            int format=state.getInt("watch_format",0);
            if((format!=1&&format!=2)||(!kind.equals("boot")&&!kind.equals("update")))throw new IllegalStateException("Invalid watch");
            String attempt=state.getString("watch_attempt","");
            long installed=state.getLong("watch_installed",0),target=state.getLong("watch_target",0);
            long dispatched=state.getLong("watch_dispatched",0),started=state.getLong("watch_started",0);
            if(boot<0||!WatchPolicy.code(installed)||target<=installed||!WatchPolicy.code(target)
                ||dispatched!=0&&(!WatchPolicy.code(dispatched)||dispatched<target)
                ||started!=0&&(!WatchPolicy.code(started)||started<installed||kind.equals("update")&&started<target)
                ||kind.equals("update")&&!attempts.contains(attempt)||kind.equals("boot")&&!attempt.equals("boot"))throw new IllegalStateException("Invalid watch binding");
            // Format1 timing fields intentionally do not become monitor deadlines.
            watch=new WatchPolicy.Watch(kind,attempt,installed,target,dispatched,started);
        }
        return new WatchPolicy.Ledger(boot,attempts,watch);
    }
    public boolean save(WatchPolicy.Ledger value) {
        android.content.SharedPreferences.Editor edit=state.edit().putLong("watch_boot",value.boot).putStringSet("watch_attempts",new HashSet<String>(value.attempts));
        for(String field:WATCH_FIELDS)edit.remove(field);
        WatchPolicy.Watch w=value.watch;
        if(w!=null)edit.putInt("watch_format",2).putString("watch_kind",w.kind).putString("watch_attempt",w.attempt)
            .putLong("watch_installed",w.installed).putLong("watch_target",w.target)
            .putLong("watch_dispatched",w.dispatchedCode).putLong("watch_started",w.startedCode);
        return edit.commit();
    }
    public void dispatchFallback(BootPolicy.Target expected) {
        long boot=currentBoot(),deadline=now()+WatchPolicy.COMMAND_MS;
        BootPolicy.Target current=resolveCurrentTarget();
        if(current==null||current.version!=expected.version||(current.protocol!=2&&current.protocol!=3)||!current.sameSigner||!current.singleTask||!current.activity.equals(expected.activity)
            ||!BootPolicy.TARGET_PACKAGE.equals(current.packageName)||currentIntent==null||!enabled()||!overlayAllowed()
            ||!WatchPolicy.commandLive(this,boot,deadline))throw new SecurityException("Fallback target changed or command expired");
        context.startActivity(currentIntent);
        record("FALLBACK_REQUESTED_UNCONFIRMED");
    }
    public void record(String status) {
        boolean saved = state.edit().putString("status", status).commit();
        Log.i("AtmungsraumBoot", "boot=" + admittedBoot() + " status=" + status + " saved=" + saved + " confirmed=false");
    }
}
