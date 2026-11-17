package org.viscereality.atmungsraum.boot;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.pm.ServiceInfo;
import android.os.Build;
import android.os.Handler;
import android.os.IBinder;
import android.os.Looper;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** Durable observation until actual focused-ready or explicit cancellation. */
public final class WatchService extends Service {
    private static final String STOP_MONITOR="org.viscereality.atmungsraum.boot.STOP_MONITOR";
    private static volatile boolean foregroundReady;
    private static volatile CountDownLatch foreground=new CountDownLatch(1);
    private final Handler handler=new Handler(Looper.getMainLooper());
    private AndroidHost host;
    private boolean registered;
    private long pollDelay=5000;
    static boolean ensureRunning(Context context,boolean awaitForeground) {
        CountDownLatch started=foreground;
        try {
            context.startForegroundService(new Intent(context,WatchService.class));
            if(!awaitForeground)return true; // BOOT receiver must return before main-looper service creation.
            return foregroundReady||(Looper.myLooper()!=Looper.getMainLooper()&&started.await(3,TimeUnit.SECONDS)&&foregroundReady);
        }catch(RuntimeException unavailable){return false;}
        catch(InterruptedException interrupted){Thread.currentThread().interrupt();return false;}
    }
    private final Runnable poll=new Runnable(){public void run(){
        WatchPolicy.poll(host);
        if(!WatchPolicy.watching(host)){stopSelf();return;}
        ((NotificationManager)getSystemService(NOTIFICATION_SERVICE)).notify(2,notification());
        handler.postDelayed(this,pollDelay);
        pollDelay=Math.min(60000,pollDelay*2);
    }};
    private final BroadcastReceiver replacement=new BroadcastReceiver(){public void onReceive(Context c,Intent i){
        if(i!=null&&i.getData()!=null&&BootPolicy.TARGET_PACKAGE.equals(i.getData().getSchemeSpecificPart())){
            pollDelay=5000;handler.removeCallbacks(poll);handler.post(poll);
        }
    }};
    @Override public void onCreate(){
        super.onCreate();host=new AndroidHost(this);
        NotificationManager manager=(NotificationManager)getSystemService(NOTIFICATION_SERVICE);
        manager.createNotificationChannel(new NotificationChannel("bounded-launch", "Atmungsraum update launch",NotificationManager.IMPORTANCE_LOW));
        Notification notification=notification();
        if(Build.VERSION.SDK_INT>=34)startForeground(2,notification,ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        else startForeground(2,notification);
        IntentFilter filter=new IntentFilter();filter.addAction(Intent.ACTION_PACKAGE_ADDED);filter.addAction(Intent.ACTION_PACKAGE_REPLACED);
        filter.addAction(Intent.ACTION_PACKAGE_REMOVED);filter.addDataScheme("package");
        if(Build.VERSION.SDK_INT>=33)registerReceiver(replacement,filter,Context.RECEIVER_NOT_EXPORTED);else registerReceiver(replacement,filter);
        registered=true;foregroundReady=true;foreground.countDown();
    }
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        if(intent!=null&&STOP_MONITOR.equals(intent.getAction())){
            boolean saved=WatchPolicy.cancelInternal(host);host.record(saved?"MONITOR_CANCELLED_BY_OPERATOR":"MONITOR_CANCELLATION_NOT_SAVED");stopSelf();return START_NOT_STICKY;
        }
        handler.removeCallbacks(poll);handler.post(poll);
        // Process death does not reset durable per-boot dispatch consumption.
        return START_STICKY;
    }
    private Notification notification(){
        PendingIntent stop=PendingIntent.getService(this,3,new Intent(this,WatchService.class).setAction(STOP_MONITOR),
            PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_IMMUTABLE);
        PendingIntent setup=PendingIntent.getActivity(this,4,new Intent(this,SetupActivity.class),
            PendingIntent.FLAG_UPDATE_CURRENT|PendingIntent.FLAG_IMMUTABLE);
        return new Notification.Builder(this,"bounded-launch").setSmallIcon(android.R.drawable.ic_popup_sync)
            .setContentTitle("Atmungsraum update monitor").setContentText(WatchPolicy.describe(host))
            .setContentIntent(setup).addAction(new Notification.Action.Builder(null,"Stop monitor",stop).build())
            .setOngoing(true).setCategory(Notification.CATEGORY_SERVICE).build();
    }
    @Override public void onDestroy(){
        handler.removeCallbacks(poll);if(registered)unregisterReceiver(replacement);
        foregroundReady=false;foreground=new CountDownLatch(1);stopForeground(STOP_FOREGROUND_REMOVE);super.onDestroy();
    }
    @Override public IBinder onBind(Intent intent){return null;}
}
