"""Actual helper provider/AndroidHost/service against inert Android seams; no APK/device."""
from pathlib import Path
import subprocess,tempfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
STUBS={
'android/os/Bundle.java': r'''package android.os; public class Bundle { public java.util.Map<String,Object> m=new java.util.HashMap<>(); public Object get(String k){return m.get(k);} public java.util.Set<String> keySet(){return m.keySet();} public void putInt(String k,int v){m.put(k,v);} public void putLong(String k,long v){m.put(k,v);} public void putString(String k,String v){m.put(k,v);} public void putBoolean(String k,boolean v){m.put(k,v);} public int getInt(String k){return getInt(k,0);} public int getInt(String k,int d){Object x=m.get(k);return x==null?d:(Integer)x;} public boolean getBoolean(String k){return Boolean.TRUE.equals(m.get(k));} public String getString(String k){return (String)m.get(k);} }''',
'android/os/Binder.java':'package android.os;public class Binder {public static int uid=100;public static int getCallingUid(){return uid;}}',
'android/os/SystemClock.java':'package android.os;public class SystemClock {public static long now=1000;public static long elapsedRealtime(){return now;}}',
'android/os/Build.java':'package android.os;public class Build {public static class VERSION {public static int SDK_INT=34;}}',
'android/os/IBinder.java':'package android.os;public interface IBinder {}',
'android/os/Looper.java':'package android.os;public class Looper {static Looper main=new Looper();public static Looper getMainLooper(){return main;}public static Looper myLooper(){return null;}}',
'android/os/Handler.java':'package android.os;public class Handler {public static long lastDelay;public static java.util.List<Runnable> q=new java.util.ArrayList<>();public Handler(Looper l){}public void removeCallbacks(Runnable r){q.removeIf(x->x==r);}public boolean post(Runnable r){q.add(r);return true;}public boolean postDelayed(Runnable r,long n){lastDelay=n;q.add(r);return true;}public static void once(){if(!q.isEmpty())q.remove(0).run();}}',
'android/net/Uri.java':'package android.net;public class Uri {String value;public Uri(String v){value=v;}public String getSchemeSpecificPart(){return value;}}',
'android/database/Cursor.java':'package android.database;public interface Cursor {}',
'android/content/ContentValues.java':'package android.content;public class ContentValues {}',
'android/content/ContentResolver.java':'package android.content;public class ContentResolver {}',
'android/content/SharedPreferences.java': r'''package android.content;public class SharedPreferences {public java.util.Map<String,Object> m=new java.util.HashMap<>();public boolean commits=true;
public long getLong(String k,long d){return m.containsKey(k)?(Long)m.get(k):d;}public int getInt(String k,int d){return m.containsKey(k)?(Integer)m.get(k):d;}public String getString(String k,String d){return m.containsKey(k)?(String)m.get(k):d;}public boolean getBoolean(String k,boolean d){return m.containsKey(k)?(Boolean)m.get(k):d;}@SuppressWarnings("unchecked")public java.util.Set<String> getStringSet(String k,java.util.Set<String>d){return m.containsKey(k)?(java.util.Set<String>)m.get(k):d;}public Editor edit(){return new Editor();}
public class Editor {java.util.Map<String,Object> values=new java.util.HashMap<>(m);public Editor putLong(String k,long v){values.put(k,v);return this;}public Editor putInt(String k,int v){values.put(k,v);return this;}public Editor putString(String k,String v){values.put(k,v);return this;}public Editor putBoolean(String k,boolean v){values.put(k,v);return this;}public Editor putStringSet(String k,java.util.Set<String> v){values.put(k,new java.util.HashSet<>(v));return this;}public Editor remove(String k){values.remove(k);return this;}public boolean commit(){if(commits)m=values;return commits;}}}''',
'android/content/ComponentName.java':'package android.content;public class ComponentName {String p,c;public ComponentName(String p,String c){this.p=p;this.c=c;}public String getPackageName(){return p;}public String getClassName(){return c;}public String flattenToShortString(){return p+"/"+c;}}',
'android/content/Intent.java':r'''package android.content;public class Intent {public static final String ACTION_MAIN="android.intent.action.MAIN",CATEGORY_LAUNCHER="android.intent.category.LAUNCHER",ACTION_PACKAGE_ADDED="added",ACTION_PACKAGE_REPLACED="replaced",ACTION_PACKAGE_REMOVED="removed",ACTION_BOOT_COMPLETED="android.intent.action.BOOT_COMPLETED";public static final int FLAG_ACTIVITY_NEW_TASK=1,FLAG_ACTIVITY_RESET_TASK_IF_NEEDED=2;String action,category;ComponentName component;android.net.Uri data;public Intent(String a){action=a;}public Intent(Intent i){action=i.action;category=i.category;component=i.component;}public Intent(Context c,Class<?> t){}public Intent setAction(String a){action=a;return this;}public Intent addCategory(String c){category=c;return this;}public Intent setPackage(String p){return this;}public Intent setComponent(ComponentName c){component=c;return this;}public Intent addFlags(int f){return this;}public ComponentName getComponent(){return component;}public String getAction(){return action;}public boolean hasCategory(String c){return c.equals(category);}public android.net.Uri getData(){return data;}public Intent data(String d){data=new android.net.Uri(d);return this;}}''',
'android/content/IntentFilter.java':'package android.content;public class IntentFilter {public void addAction(String s){}public void addDataScheme(String s){}}',
'android/content/BroadcastReceiver.java':'package android.content;public abstract class BroadcastReceiver {public abstract void onReceive(Context c,Intent i);}',
'android/content/pm/Signature.java':'package android.content.pm;public class Signature {String s;public Signature(String s){this.s=s;}public boolean equals(Object o){return o instanceof Signature&&s.equals(((Signature)o).s);}public int hashCode(){return s.hashCode();}}',
'android/content/pm/SigningInfo.java':'package android.content.pm;public class SigningInfo {public Signature[] signatures={new Signature("same")};public Signature[] getApkContentsSigners(){return signatures;}}',
'android/content/pm/ApplicationInfo.java':'package android.content.pm;public class ApplicationInfo {public int uid=100;public boolean enabled=true;public android.os.Bundle metaData=new android.os.Bundle();}',
'android/content/pm/PackageInfo.java':'package android.content.pm;public class PackageInfo {public long code=19;public ApplicationInfo applicationInfo=new ApplicationInfo();public SigningInfo signingInfo=new SigningInfo();public long getLongVersionCode(){return code;}}',
'android/content/pm/ActivityInfo.java':'package android.content.pm;public class ActivityInfo {public static final int LAUNCH_SINGLE_TASK=2;public int launchMode=2;public boolean enabled=true,exported=true;public ApplicationInfo applicationInfo=new ApplicationInfo();public String packageName="org.viscereality.atmungsraum.developer",name="Current";}',
'android/content/pm/ResolveInfo.java':'package android.content.pm;public class ResolveInfo {public ActivityInfo activityInfo=new ActivityInfo();}',
'android/content/pm/PackageManager.java':r'''package android.content.pm;public class PackageManager {public static final int GET_META_DATA=1,GET_SIGNING_CERTIFICATES=2;public PackageInfo main=new PackageInfo(),helper=new PackageInfo();public int launchMode=2;public String[] packages={"org.viscereality.atmungsraum.developer"};public PackageManager(){main.applicationInfo.metaData.putInt("org.viscereality.atmungsraum.boot_protocol",2);}public static class NameNotFoundException extends Exception {}public ApplicationInfo getApplicationInfo(String p,int f)throws NameNotFoundException{return getPackageInfo(p,f).applicationInfo;}public PackageInfo getPackageInfo(String p,int f)throws NameNotFoundException{return p.endsWith(".boot")?helper:main;}public String[] getPackagesForUid(int uid){return packages;}public ResolveInfo resolveActivity(android.content.Intent i,int f){ResolveInfo r=new ResolveInfo();r.activityInfo.launchMode=launchMode;return r;}}''',
'android/content/pm/ServiceInfo.java':'package android.content.pm;public class ServiceInfo {public static final int FOREGROUND_SERVICE_TYPE_SPECIAL_USE=1073741824;}',
'android/content/Context.java':r'''package android.content;public class Context {public static final int MODE_PRIVATE=0,RECEIVER_NOT_EXPORTED=4;public static final String NOTIFICATION_SERVICE="notification";public SharedPreferences prefs=new SharedPreferences();public android.content.pm.PackageManager pm=new android.content.pm.PackageManager();public boolean permission=true,serviceAllowed=true;public int launches,registrations;public long serviceAdvance;public BroadcastReceiver receiver;public org.viscereality.atmungsraum.boot.WatchService service;
public Context(){prefs.m.put("enabled",true);}public SharedPreferences getSharedPreferences(String n,int m){return prefs;}public android.content.pm.PackageManager getPackageManager(){return pm;}public String getPackageName(){return "org.viscereality.atmungsraum.boot";}public ContentResolver getContentResolver(){return new ContentResolver();}public void enforceCallingPermission(String p,String m){if(!permission)throw new SecurityException(m);}public void startActivity(Intent i){launches++;}public ComponentName startForegroundService(Intent i){android.os.SystemClock.now+=serviceAdvance;if(!serviceAllowed)throw new IllegalStateException("ineligible");android.app.Service.base=this;if(service==null){service=new org.viscereality.atmungsraum.boot.WatchService();service.onCreate();}service.onStartCommand(i,0,1);return new ComponentName(getPackageName(),"WatchService");}public boolean stopService(Intent i){if(service!=null){service.onDestroy();service=null;}return true;}public Object getSystemService(String n){return new android.app.NotificationManager();}public Intent registerReceiver(BroadcastReceiver r,IntentFilter f,int flags){registrations++;receiver=r;return null;}public Intent registerReceiver(BroadcastReceiver r,IntentFilter f){return registerReceiver(r,f,0);}public void unregisterReceiver(BroadcastReceiver r){registrations--;receiver=null;}}''',
'android/content/ContentProvider.java':r'''package android.content;public abstract class ContentProvider {Context c;public void attach(Context c){this.c=c;}public Context getContext(){return c;}public abstract boolean onCreate();public android.os.Bundle call(String m,String a,android.os.Bundle e){return null;}public abstract android.database.Cursor query(android.net.Uri u,String[] p,String s,String[] a,String o);public abstract String getType(android.net.Uri u);public abstract android.net.Uri insert(android.net.Uri u,ContentValues v);public abstract int update(android.net.Uri u,ContentValues v,String s,String[]a);public abstract int delete(android.net.Uri u,String s,String[]a);}''',
'android/provider/Settings.java':'package android.provider;public class Settings {public static boolean overlay=true;public static boolean canDrawOverlays(android.content.Context c){return overlay;}public static class Global {public static final String BOOT_COUNT="boot";public static int boot=8;public static int getInt(android.content.ContentResolver c,String k,int d){return boot;}}}',
'android/util/Log.java':'package android.util;public class Log {public static int i(String t,String m){return 0;}public static int e(String t,String m,Throwable x){return 0;}}',
'android/R.java':'package android;public class R {public static class drawable {public static int ic_popup_sync=1;}}',
 'org/viscereality/atmungsraum/boot/SetupActivity.java':'package org.viscereality.atmungsraum.boot;public class SetupActivity {}',
'android/app/PendingIntent.java':'package android.app;public class PendingIntent {public static final int FLAG_UPDATE_CURRENT=1,FLAG_IMMUTABLE=2;public static PendingIntent getService(android.content.Context c,int id,android.content.Intent i,int f){return new PendingIntent();}public static PendingIntent getActivity(android.content.Context c,int id,android.content.Intent i,int f){return new PendingIntent();}}',
'android/app/Notification.java':'package android.app;public class Notification {public static final String CATEGORY_SERVICE="service";public static class Action {public static class Builder {public Builder(Object i,String s,PendingIntent p){}public Action build(){return new Action();}}}public static class Builder {public Builder(android.content.Context c,String n){}public Builder setSmallIcon(int i){return this;}public Builder setContentTitle(String s){return this;}public Builder setContentText(String s){return this;}public Builder setContentIntent(PendingIntent p){return this;}public Builder addAction(Action a){return this;}public Builder setOngoing(boolean o){return this;}public Builder setCategory(String c){return this;}public Notification build(){return new Notification();}}}',
'android/app/NotificationChannel.java':'package android.app;public class NotificationChannel {public NotificationChannel(String i,String n,int p){}}',
'android/app/NotificationManager.java':'package android.app;public class NotificationManager {public static final int IMPORTANCE_LOW=2;public void createNotificationChannel(NotificationChannel n){}public void notify(int id,Notification n){}}',
'android/app/Service.java':r'''package android.app;public class Service extends android.content.Context {public static android.content.Context base;public static int foregroundType;public static final int START_STICKY=1,START_NOT_STICKY=2,STOP_FOREGROUND_REMOVE=1;public void onCreate(){}public int onStartCommand(android.content.Intent i,int f,int id){return 0;}public void onDestroy(){}public android.os.IBinder onBind(android.content.Intent i){return null;}public void startForeground(int id,Notification n,int type){foregroundType=type;}public void startForeground(int id,Notification n){}public void startActivity(android.content.Intent i){base.startActivity(i);}public void stopForeground(int flags){}public void stopSelf(){base.stopService(null);}public android.content.SharedPreferences getSharedPreferences(String n,int m){return base.getSharedPreferences(n,m);}public android.content.pm.PackageManager getPackageManager(){return base.getPackageManager();}public android.content.ContentResolver getContentResolver(){return base.getContentResolver();}public android.content.Intent registerReceiver(android.content.BroadcastReceiver r,android.content.IntentFilter f,int flags){return base.registerReceiver(r,f,flags);}public android.content.Intent registerReceiver(android.content.BroadcastReceiver r,android.content.IntentFilter f){return base.registerReceiver(r,f);}public void unregisterReceiver(android.content.BroadcastReceiver r){base.unregisterReceiver(r);}}'''
}
TEST=r'''package org.viscereality.atmungsraum.boot;
import android.os.*;import android.content.*;import android.provider.Settings;
public class AndroidBoundaryTest {
 static int checks;static Context prior;
 static void check(boolean b,String m){checks++;if(!b)throw new AssertionError(m);}
 static Context fresh(){if(prior!=null)prior.stopService(null);Handler.q.clear();Settings.Global.boot=8;SystemClock.now=1000;Settings.overlay=true;return prior=new Context();}
 static Bundle body(String method,boolean monitor){Bundle b=new Bundle();b.putInt("v",1);if(!method.equals("status"))b.putLong("installed_code",19);if(method.equals("arm-update")){b.putLong("target_code",20);b.putString("attempt","42");}if(monitor){b.putInt("monitor_protocol",3);b.putLong("boot_count",Settings.Global.boot);b.putLong("deadline_elapsed",SystemClock.now+5000);}return b;}
 static Bundle body(String method){return body(method,true);}
 static HandoffProvider provider(Context c){HandoffProvider p=new HandoffProvider();p.attach(c);return p;}
 interface Op {void run();}static void denied(Op op){try{op.run();throw new AssertionError("caller admitted");}catch(SecurityException good){checks++;}}
 public static void main(String[] args){
  Context c=fresh();HandoffProvider p=provider(c);Bundle arm=body("arm-update");
  c.permission=false;final HandoffProvider perm=p;denied(()->perm.call("status",null,body("status")));c.permission=true;
  Binder.uid=9;denied(()->perm.call("arm-update",null,arm));Binder.uid=100;
  c.pm.packages=new String[]{BootPolicy.TARGET_PACKAGE,"shared"};denied(()->perm.call("status",null,body("status")));c.pm.packages=new String[]{BootPolicy.TARGET_PACKAGE};
  c.pm.main.signingInfo.signatures=new android.content.pm.Signature[]{new android.content.pm.Signature("other")};denied(()->perm.call("status",null,body("status")));c.pm.main.signingInfo.signatures=c.pm.helper.signingInfo.signatures;
  Bundle invalid=body("arm-update");invalid.putString("package","arbitrary");check(!p.call("arm-update",null,invalid).getBoolean("accepted"),"closed bundle rejects arbitrary target");
  invalid=body("arm-update");invalid.m.put("installed_code",19);check(!p.call("arm-update",null,invalid).getBoolean("accepted"),"integer cannot replace long code");
  check(!p.call("arm-update","component",arm).getBoolean("accepted"),"arg cannot inject component");
  invalid=body("arm-update");invalid.putInt("monitor_protocol",2);check(!p.call("arm-update",null,invalid).getBoolean("accepted"),"wrong monitor protocol denied");
  invalid=body("arm-update");invalid.putLong("boot_count",7);check(!p.call("arm-update",null,invalid).getBoolean("accepted"),"stale boot IPC denied");
  invalid=body("arm-update");invalid.putLong("deadline_elapsed",999);check(!p.call("arm-update",null,invalid).getBoolean("accepted"),"expired IPC denied");
  invalid=body("arm-update");invalid.putLong("deadline_elapsed",6001);check(!p.call("arm-update",null,invalid).getBoolean("accepted"),"unbounded IPC deadline denied");
  Bundle reply=p.call("arm-update",null,arm);
  check(reply.getBoolean("accepted")&&reply.getInt("protocol")==3&&reply.getInt("monitor_protocol")==3&&reply.getBoolean("monitor")&&reply.getBoolean("watching"),"explicit monitor capability plus actual FGS before ACK");
  check(android.app.Service.foregroundType==1073741824&&c.registrations==1,"specialUse and protected dynamic receiver");
  AndroidHost owner=new AndroidHost(c);check(owner.load().watch.target==20&&owner.load().attempts.contains("42"),"durable target/attempt");
  check(p.call("arm-update",null,body("arm-update")).getBoolean("accepted"),"duplicate exact arm idempotent");
  SystemClock.now+=9*60*60*1000L;Handler.once();check(WatchPolicy.watching(owner)&&c.launches==0,"actual service does not expire long download");
  check(Handler.lastDelay==5000,"low-rate observation begins at five seconds, no completion grace");
  c.pm.main.code=20;WatchPolicy.poll(owner);check(c.launches==1,"actual PM replacement launches immediately without grace");
  c.service.onDestroy();c.service=null;WatchService.ensureRunning(c,true);Handler.once();check(c.launches==1,"service recreation reloads durable consumption");
  Bundle started=body("launch-started");check(!p.call("launch-started",null,started).getBoolean("accepted"),"old loaded code rejected");
  started.putLong("installed_code",20);check(p.call("launch-started",null,started).getBoolean("accepted"),"actual new process start acknowledged");
  check(owner.load().watch!=null&&c.service!=null,"started ACK retains monitor awaiting real focus");
  Bundle ready=body("launch-ready");ready.putLong("installed_code",20);check(p.call("launch-ready",null,ready).getBoolean("accepted"),"actual new focused-ready ACK");
  check(owner.load().watch==null&&c.service==null&&c.registrations==0,"ready clears durable watch and service");
  c=fresh();p=provider(c);p.call("arm-update",null,body("arm-update"));Handler.q.clear();
  c.receiver.onReceive(c,new Intent(Intent.ACTION_PACKAGE_REPLACED).data("other.package"));check(Handler.q.isEmpty()&&c.launches==0,"unrelated package event cannot wake fixed monitor");
  c.receiver.onReceive(c,new Intent(Intent.ACTION_PACKAGE_REPLACED).data(BootPolicy.TARGET_PACKAGE));Handler.once();check(c.launches==0,"package event alone never proves new PM identity");
  c.pm.main.code=20;c.receiver.onReceive(c,new Intent(Intent.ACTION_PACKAGE_REPLACED).data(BootPolicy.TARGET_PACKAGE));Handler.once();check(c.launches==1,"actual protected package event observes installed target and dispatches once");
  c.receiver.onReceive(c,new Intent(Intent.ACTION_PACKAGE_REPLACED).data(BootPolicy.TARGET_PACKAGE));Handler.once();check(c.launches==1,"duplicate package events cannot duplicate dispatch");
  c=fresh();p=provider(c);reply=p.call("arm-update",null,body("arm-update",false));
  check(reply.getBoolean("accepted")&&reply.getInt("protocol")==2&&reply.getBoolean("monitor"),"immutable legacy protocol2 request/reply preserved");
  check(p.call("cancel-watch",null,body("cancel-watch",false)).getBoolean("accepted"),"legacy protected escape retained");
  c=fresh();p=provider(c);c.pm.main.applicationInfo.metaData.putInt("org.viscereality.atmungsraum.boot_protocol",3);
  check(p.call("arm-update",null,body("arm-update")).getBoolean("accepted"),"new main protocol3 metadata supported");
  c=fresh();p=provider(c);c.pm.launchMode=0;check(!p.call("arm-update",null,body("arm-update")).getBoolean("accepted"),"actual PM non-singleTask blocks arm");
  c=fresh();p=provider(c);c.serviceAllowed=false;check(!p.call("arm-update",null,body("arm-update")).getBoolean("accepted"),"FGS failure denies handoff");
  check(new AndroidHost(c).load().watch==null,"failed FGS cancels only new arm");
  c=fresh();p=provider(c);c.serviceAdvance=5001;check(!p.call("arm-update",null,body("arm-update")).getBoolean("accepted"),"slow service creation expires IPC ACK");
  check(new AndroidHost(c).load().watch==null,"stale newly-created arm revoked");
  c=fresh();p=provider(c);p.call("arm-update",null,body("arm-update"));c.serviceAdvance=5001;
  check(!p.call("arm-update",null,body("arm-update")).getBoolean("accepted")&&new AndroidHost(c).load().watch!=null,"late duplicate ACK cannot cancel earlier accepted monitor");
  c=fresh();p=provider(c);p.call("arm-update",null,body("arm-update"));c.service.onStartCommand(new Intent(c,WatchService.class).setAction("org.viscereality.atmungsraum.boot.STOP_MONITOR"),0,2);
  check(c.service==null&&new AndroidHost(c).load().watch==null,"operator notification Stop monitor cancels durable authority");
  c.pm.main.code=20;Settings.Global.boot++;WatchPolicy.poll(new AndroidHost(c));check(c.launches==0,"explicit stop survives replacement/reboot");
  c=fresh();p=provider(c);p.call("arm-update",null,body("arm-update"));c.stopService(null);Settings.Global.boot++;SystemClock.now=10;
  new BootReceiver().onReceive(c,new Intent(Intent.ACTION_BOOT_COMPLETED));
  owner=new AndroidHost(c);check(owner.load().boot==9&&owner.load().watch!=null&&c.launches==0&&c.service!=null,"real boot resumes target monitor without launching old main");
  check(owner.admittedBoot()==9,"resume consumes normal boot launch admission");
  c.pm.main.code=20;WatchPolicy.poll(owner);check(c.launches==1,"resumed target observed and launched once");
  c=fresh();p=provider(c);p.call("arm-update",null,body("arm-update"));c.stopService(null);Settings.Global.boot++;c.prefs.commits=false;
  new BootReceiver().onReceive(c,new Intent(Intent.ACTION_BOOT_COMPLETED));check(c.launches==0&&c.service==null,"failed durable resume cannot fall through into old experiment");
  c=fresh();p=provider(c);p.call("arm-update",null,body("arm-update"));owner=new AndroidHost(c);c.prefs.commits=false;
  check(!owner.setEnabled(false)&&owner.enabled()&&owner.load().watch!=null,"failed disable preserves both enabled and monitor; never claims cancellation");
  c.prefs.commits=true;check(owner.setEnabled(false)&&!owner.enabled()&&owner.load().watch==null,"successful disable atomically cancels durable watch");
  check(owner.setEnabled(true)&&owner.enabled()&&owner.load().watch==null,"re-enable cannot resurrect cancelled monitor");
  c.pm.main.code=20;WatchPolicy.poll(owner);check(c.launches==0,"cancelled watch cannot launch after later actual replacement");
  c=fresh();p=provider(c);p.call("arm-update",null,body("arm-update"));c.prefs.m.put("watch_format",1);c.prefs.m.put("watch_deadline",1L);
  owner=new AndroidHost(c);SystemClock.now=86400000L;check(WatchPolicy.watching(owner),"legacy pending target migrates without old expiry");
  owner.save(owner.load());check(!c.prefs.m.containsKey("watch_deadline")&&c.prefs.getInt("watch_format",0)==2,"new persistence removes timing authority");
  c.prefs.m.put("watch_kind","broken");check(!WatchPolicy.armUpdate(owner,"43",19,20).accepted,"corrupt state fails closed");
  System.out.println("PASS AndroidBoundaryTest: "+checks+" actual provider/host/service/boot checks; inert Android");
 }
}'''

def manifest():
 ns={'a':'http://schemas.android.com/apk/res/android'}; a='{'+ns['a']+'}'
 m=ET.parse(ROOT/'AndroidManifest.xml').getroot()
 assert m.attrib[a+'versionCode']=='3' and m.attrib[a+'versionName']=='3.0'
 assert {x.attrib[a+'name'] for x in m.findall('uses-permission')}=={'android.permission.RECEIVE_BOOT_COMPLETED','android.permission.SYSTEM_ALERT_WINDOW','android.permission.FOREGROUND_SERVICE','android.permission.FOREGROUND_SERVICE_SPECIAL_USE'}
 assert m.find('permission').attrib=={a+'name':'org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF',a+'protectionLevel':'signature'}
 app=m.find('application'); assert len(app.findall('provider'))==1 and len(app.findall('service'))==1
 metadata={x.attrib[a+'name']:x.attrib[a+'value'] for x in app.findall('meta-data')}
 assert metadata['org.viscereality.atmungsraum.boot_protocol']=='2' and metadata['org.viscereality.atmungsraum.monitor_protocol']=='3'
 provider=app.find('provider');assert provider.attrib=={a+'name':'.HandoffProvider',a+'authorities':'org.viscereality.atmungsraum.boot.handoff',a+'exported':'true',a+'permission':'org.viscereality.atmungsraum.boot.permission.UPDATE_HANDOFF'}
 service=app.find('service');assert service.attrib[a+'exported']=='false' and service.attrib[a+'foregroundServiceType']=='specialUse'
 assert service.find('property').attrib[a+'name']=='android.app.PROPERTY_SPECIAL_USE_FGS_SUBTYPE'
 assert app.find('receiver').attrib[a+'exported']=='false'
 print('PASS exact helper3 manifest permission/provider/service closure')

def main():
 manifest()
 with tempfile.TemporaryDirectory(prefix='atmungsraum-helper-android-') as tmp:
  out=Path(tmp); sources=[]
  for name,body in STUBS.items():
   p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body,encoding='utf-8');sources.append(str(p))
  test=out/'AndroidBoundaryTest.java';test.write_text(TEST,encoding='utf-8');sources.append(str(test))
  prod=ROOT/'src/org/viscereality/atmungsraum/boot'
  sources += [str(prod/(name+'.java')) for name in ['BootPolicy','WatchPolicy','AndroidHost','HandoffProvider','WatchService','BootReceiver']]
  subprocess.run(['javac','--release','8','-encoding','UTF-8','-d',str(out),*sources],check=True)
  subprocess.run(['java','-cp',str(out),'org.viscereality.atmungsraum.boot.AndroidBoundaryTest'],check=True)
if __name__=='__main__':main()
