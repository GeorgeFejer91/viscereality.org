package org.viscereality.atmungsraum.boot;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
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

/** Foreground lifetime is finite and durable; a BOOT receiver is never used as a watcher. */
public final class WatchService extends Service {
    private static volatile boolean foregroundReady;
    private static volatile CountDownLatch foreground=new CountDownLatch(1);
    private final Handler handler=new Handler(Looper.getMainLooper());
    private AndroidHost host;
    private boolean registered;
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
        handler.postDelayed(this,1000);
    }};
    private final BroadcastReceiver replacement=new BroadcastReceiver(){public void onReceive(Context c,Intent i){
        if(i!=null&&i.getData()!=null&&BootPolicy.TARGET_PACKAGE.equals(i.getData().getSchemeSpecificPart())){
            handler.removeCallbacks(poll);handler.post(poll);
        }
    }};
    @Override public void onCreate(){
        super.onCreate();host=new AndroidHost(this);
        NotificationManager manager=(NotificationManager)getSystemService(NOTIFICATION_SERVICE);
        manager.createNotificationChannel(new NotificationChannel("bounded-launch", "Atmungsraum update launch",NotificationManager.IMPORTANCE_LOW));
        Notification notification=new Notification.Builder(this,"bounded-launch").setSmallIcon(android.R.drawable.ic_popup_sync)
            .setContentTitle("Atmungsraum launch assistance").setContentText("Waiting briefly for an installed update to open. Stops automatically.")
            .setOngoing(true).setCategory(Notification.CATEGORY_SERVICE).build();
        if(Build.VERSION.SDK_INT>=34)startForeground(2,notification,ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        else startForeground(2,notification);
        IntentFilter filter=new IntentFilter();filter.addAction(Intent.ACTION_PACKAGE_ADDED);filter.addAction(Intent.ACTION_PACKAGE_REPLACED);
        filter.addAction(Intent.ACTION_PACKAGE_REMOVED);filter.addDataScheme("package");
        if(Build.VERSION.SDK_INT>=33)registerReceiver(replacement,filter,Context.RECEIVER_NOT_EXPORTED);else registerReceiver(replacement,filter);
        registered=true;foregroundReady=true;foreground.countDown();
    }
    @Override public int onStartCommand(Intent intent,int flags,int startId){
        handler.removeCallbacks(poll);handler.post(poll);
        // Android may recreate the service after process death; persisted boot/elapsed expiry
        // remains mandatory. No intent extra, automatic renewal or restart after cancellation.
        return START_STICKY;
    }
    @Override public void onDestroy(){
        handler.removeCallbacks(poll);if(registered)unregisterReceiver(replacement);
        foregroundReady=false;foreground=new CountDownLatch(1);stopForeground(STOP_FOREGROUND_REMOVE);super.onDestroy();
    }
    @Override public IBinder onBind(Intent intent){return null;}
}
