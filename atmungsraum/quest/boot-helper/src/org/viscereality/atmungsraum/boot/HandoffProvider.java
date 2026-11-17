package org.viscereality.atmungsraum.boot;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Binder;
import android.os.Bundle;
import java.util.Arrays;
import java.util.HashSet;

/** Signature permission plus exact package/UID/current certificate; no caller-selected effects. */
public final class HandoffProvider extends ContentProvider {
    static final String PERMISSION="org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF";
    public boolean onCreate(){return true;}
    private void caller() {
        getContext().enforceCallingPermission(PERMISSION,"Exact main handoff permission required");
        PackageManager pm=getContext().getPackageManager();int uid=Binder.getCallingUid();
        try {
            ApplicationInfo main=pm.getApplicationInfo(BootPolicy.TARGET_PACKAGE,0);
            if(!WatchPolicy.exactCaller(pm.getPackagesForUid(uid),uid,main.uid,AndroidHost.sameCurrentSigner(pm,getContext().getPackageName(),BootPolicy.TARGET_PACKAGE)))
                throw new SecurityException("Exact current main caller required");
        }catch(PackageManager.NameNotFoundException absent){throw new SecurityException("Main package unavailable");}
    }
    private static void fields(Bundle b,String... names){
        if(b==null||!new HashSet<String>(b.keySet()).equals(new HashSet<String>(Arrays.asList(names)))
            ||!(b.get("v") instanceof Integer)||b.getInt("v")!=1)throw new IllegalArgumentException("closed-v1-bundle-required");
    }
    private static long number(Bundle b,String name){Object x=b.get(name);if(!(x instanceof Long))throw new IllegalArgumentException("long-code-required");return ((Long)x).longValue();}
    private static void command(AndroidHost host,Bundle extras,boolean monitor) {
        if(monitor&&(!Integer.valueOf(3).equals(extras.get("monitor_protocol"))||
            !WatchPolicy.commandLive(host,number(extras,"boot_count"),number(extras,"deadline_elapsed"))))
            throw new IllegalArgumentException("stale-or-unsupported-monitor-command");
    }
    private static void requestFields(Bundle extras,boolean monitor,String... legacy){
        if(!monitor){fields(extras,legacy);return;}
        String[] names=Arrays.copyOf(legacy,legacy.length+3);
        names[legacy.length]="monitor_protocol";names[legacy.length+1]="boot_count";names[legacy.length+2]="deadline_elapsed";
        fields(extras,names);
    }
    @Override public Bundle call(String method,String arg,Bundle extras) {
        caller();AndroidHost host=new AndroidHost(getContext());WatchPolicy.Result result;
        boolean monitor=extras!=null&&extras.keySet().contains("monitor_protocol");
        try {
            if(arg!=null)throw new IllegalArgumentException("argument-not-allowed");
            command(host,extras,monitor);
            if("arm-update".equals(method)) {
                requestFields(extras,monitor,"v","attempt","installed_code","target_code");Object id=extras.get("attempt");
                if(!(id instanceof String))throw new IllegalArgumentException("attempt-required");
                long installed=number(extras,"installed_code"),target=number(extras,"target_code");
                command(host,extras,monitor);
                result=monitor?WatchPolicy.armUpdate(host,(String)id,installed,target,number(extras,"boot_count"),number(extras,"deadline_elapsed")):
                    WatchPolicy.armUpdate(host,(String)id,installed,target);
                if(result.accepted){
                    boolean newlyArmed="armed".equals(result.reason),available=false;
                    try{command(host,extras,monitor);available=WatchService.ensureRunning(getContext(),true);command(host,extras,monitor);}
                    catch(RuntimeException stale){available=false;}
                    if(!available){
                        if(newlyArmed)WatchPolicy.cancelArm(host,(String)id,installed,target);
                        result=new WatchPolicy.Result(false,"stale-command-or-monitor-service-unavailable");
                        if(!WatchPolicy.watching(host))getContext().stopService(new android.content.Intent(getContext(),WatchService.class));
                    }
                }
            } else if("launch-started".equals(method)||"launch-ready".equals(method)||"cancel-watch".equals(method)) {
                requestFields(extras,monitor,"v","installed_code");command(host,extras,monitor);
                result=monitor?WatchPolicy.ack(host,method,number(extras,"installed_code"),number(extras,"boot_count"),number(extras,"deadline_elapsed")):
                    WatchPolicy.ack(host,method,number(extras,"installed_code"));
                if(!WatchPolicy.watching(host))getContext().stopService(new android.content.Intent(getContext(),WatchService.class));
            } else if("status".equals(method)){requestFields(extras,monitor,"v");result=new WatchPolicy.Result(true,"status");}
            else result=new WatchPolicy.Result(false,"unsupported-method");
        }catch(RuntimeException invalid){result=new WatchPolicy.Result(false,"invalid-or-unavailable-handoff");}
        Bundle reply=new Bundle();reply.putInt("v",1);reply.putInt("protocol",monitor?3:2);reply.putBoolean("accepted",result.accepted);
        reply.putInt("monitor_protocol",3);reply.putBoolean("monitor",true);
        reply.putString("reason",result.reason);reply.putBoolean("watching",WatchPolicy.watching(host));return reply;
    }
    public Cursor query(Uri u,String[] p,String s,String[] a,String o){throw new UnsupportedOperationException();}
    public String getType(Uri u){return null;}
    public Uri insert(Uri u,ContentValues v){throw new UnsupportedOperationException();}
    public int update(Uri u,ContentValues v,String s,String[] a){throw new UnsupportedOperationException();}
    public int delete(Uri u,String s,String[] a){throw new UnsupportedOperationException();}
}
