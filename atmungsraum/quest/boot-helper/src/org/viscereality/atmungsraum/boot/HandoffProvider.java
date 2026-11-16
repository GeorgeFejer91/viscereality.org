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
    @Override public Bundle call(String method,String arg,Bundle extras) {
        caller();AndroidHost host=new AndroidHost(getContext());WatchPolicy.Result result;
        try {
            if(arg!=null)throw new IllegalArgumentException("argument-not-allowed");
            if("arm-update".equals(method)) {
                fields(extras,"v","attempt","installed_code","target_code");Object id=extras.get("attempt");
                if(!(id instanceof String))throw new IllegalArgumentException("attempt-required");
                result=WatchPolicy.armUpdate(host,(String)id,number(extras,"installed_code"),number(extras,"target_code"));
                if(result.accepted&&!WatchService.ensureRunning(getContext(),true)){
                    WatchPolicy.cancelInternal(host);result=new WatchPolicy.Result(false,"watch-service-unavailable");
                }
            } else if("launch-started".equals(method)||"launch-ready".equals(method)||"cancel-watch".equals(method)) {
                fields(extras,"v","installed_code");result=WatchPolicy.ack(host,method,number(extras,"installed_code"));
                if(!WatchPolicy.watching(host))getContext().stopService(new android.content.Intent(getContext(),WatchService.class));
            } else if("status".equals(method)){fields(extras,"v");result=new WatchPolicy.Result(true,"status");}
            else result=new WatchPolicy.Result(false,"unsupported-method");
        }catch(RuntimeException invalid){result=new WatchPolicy.Result(false,"invalid-or-unavailable-handoff");}
        Bundle reply=new Bundle();reply.putInt("v",1);reply.putInt("protocol",2);reply.putBoolean("accepted",result.accepted);
        reply.putString("reason",result.reason);reply.putBoolean("watching",WatchPolicy.watching(host));return reply;
    }
    public Cursor query(Uri u,String[] p,String s,String[] a,String o){throw new UnsupportedOperationException();}
    public String getType(Uri u){return null;}
    public Uri insert(Uri u,ContentValues v){throw new UnsupportedOperationException();}
    public int update(Uri u,ContentValues v,String s,String[] a){throw new UnsupportedOperationException();}
    public int delete(Uri u,String s,String[] a){throw new UnsupportedOperationException();}
}
