package org.viscereality.atmungsraum.boot;
import java.util.*;
public final class WatchPolicyTest {
 static int checks;
 static void check(boolean x,String m){checks++;if(!x)throw new AssertionError(m);}
 static class Host implements WatchPolicy.Host {
  long boot=8,now=1000;boolean enabled=true,overlay=true,commit=true;int launches;long resolveDelay;
  WatchPolicy.Ledger saved=new WatchPolicy.Ledger(8,new HashSet<String>(),null);
  BootPolicy.Target target=target(20,2,true);
  public boolean enabled(){return enabled;}public boolean overlayAllowed(){return overlay;}
  public long currentBoot(){return boot;}public long now(){return now;}
  public WatchPolicy.Ledger load(){return saved;}public boolean save(WatchPolicy.Ledger next){if(commit)saved=next;return commit;}
  public BootPolicy.Target resolveCurrentTarget(){now+=resolveDelay;return target;}
  public void dispatchFallback(BootPolicy.Target target){check(saved.watch.dispatchedCode==target.version,"durable admit before dispatch");launches++;}
 }
 static BootPolicy.Target target(long code,int protocol,boolean signer){return new BootPolicy.Target(BootPolicy.TARGET_PACKAGE,"Current",signer,true,true,code,protocol,true);}
 static void arm(Host h){check(WatchPolicy.armUpdate(h,"42",20,21).accepted,"arm observed update");}
 public static void main(String[] args){
  check(WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE},25,25,true),"exact current signer/UID");
  check(!WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE,"shared"},25,25,true),"shared UID denied");
  check(!WatchPolicy.exactCaller(new String[]{"other"},25,25,true),"other samecert package denied");
  check(!WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE},24,25,true),"wrong UID denied");
  check(!WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE},25,25,false),"wrong signer denied");
  for(String id:new String[]{"","0","01","-1","+2","9223372036854775808"}){
   Host h=new Host();check(!WatchPolicy.armUpdate(h,id,20,21).accepted,"canonical bounded attempt");
  }
  for(long[] pair:new long[][]{{20,20},{20,19},{19,21},{20,9007199254740992L},{-1,21}}){
   Host h=new Host();check(!WatchPolicy.armUpdate(h,"42",pair[0],pair[1]).accepted,"actual package and strictly newer safe version");
  }
  Host h=new Host();h.overlay=false;check(!WatchPolicy.armUpdate(h,"42",20,21).accepted,"effective overlay required to arm");
  h=new Host();h.enabled=false;check(!WatchPolicy.armUpdate(h,"42",20,21).accepted,"retained checkbox required");
  h=new Host();h.target=target(20,0,true);check(!WatchPolicy.armUpdate(h,"42",20,21).accepted,"legacy target cannot arm");
  check(!WatchPolicy.armBoot(h,h.target).accepted,"legacy boot remains one-shot");
  h=new Host();h.target=target(20,3,false);check(!WatchPolicy.armUpdate(h,"42",20,21).accepted,"changed signer denied");
  h=new Host();h.target=new BootPolicy.Target(BootPolicy.TARGET_PACKAGE,"Current",true,true,true,20,3,false);
  check(!WatchPolicy.armUpdate(h,"42",20,21).accepted,"non-singleTask cannot promise idempotent launch");
  h=new Host();h.commit=false;check(!WatchPolicy.armUpdate(h,"42",20,21).accepted,"persistence failure denies arm");
  h=new Host();arm(h);check(WatchPolicy.armUpdate(h,"42",20,21).accepted,"exact duplicate arm is idempotent");
  check(!WatchPolicy.armUpdate(h,"43",20,21).accepted,"busy monitor not replaced");
  check(WatchPolicy.ack(h,"cancel-watch",20).accepted,"protected escape cancellation");
  check(!WatchPolicy.armUpdate(h,"42",20,21).accepted,"cancelled attempt cannot replay");
  h=new Host();for(int i=1;i<=WatchPolicy.MAX_ATTEMPTS;i++){
   check(WatchPolicy.armUpdate(h,""+i,20,21).accepted,"bounded attempt ledger admission");WatchPolicy.ack(h,"cancel-watch",20);
  }
  check(!WatchPolicy.armUpdate(h,"9999",20,21).accepted,"no replay eviction at ledger capacity");
  check(!WatchPolicy.armUpdate(h,"1",20,21).accepted,"old replay retained");
  h=new Host();check(WatchPolicy.armBoot(h,h.target).accepted,"protocol2 boot watch compatible");
  check(WatchPolicy.ack(h,"launch-started",20).accepted&&h.saved.watch!=null,"original start is not focused-ready");
  h.target=target(21,3,true);check(WatchPolicy.ack(h,"launch-started",21).accepted,"replacement PM-equal start observed");
  h.now+=86400000L;WatchPolicy.poll(h);check(h.launches==0&&h.saved.watch!=null,"actual replacement start suppresses fallback while awaiting real ready");
  check(WatchPolicy.ack(h,"launch-ready",21).accepted&&h.saved.watch==null,"actual focused-ready completes");
  h=new Host();WatchPolicy.armBoot(h,h.target);WatchPolicy.ack(h,"launch-ready",20);h.target=target(21,3,true);WatchPolicy.poll(h);
  check(h.launches==0,"ready boot monitor never becomes foreground watchdog");
  h=new Host();h.resolveDelay=5001;check(!WatchPolicy.armUpdate(h,"42",20,21,8,6000).accepted,"PM delay expires command, not download");
  h=new Host();arm(h);h.resolveDelay=5001;
  check(!WatchPolicy.ack(h,"cancel-watch",20,8,6000).accepted&&h.saved.watch!=null,"late cancel cannot affect accepted monitor");
  h=new Host();check(!WatchPolicy.commandLive(h,7,2000),"IPC wrong boot denied");
  check(!WatchPolicy.commandLive(h,8,999),"IPC expired denied");
  check(!WatchPolicy.commandLive(h,8,6001),"IPC caller cannot extend command window");
  System.out.println("PASS WatchPolicyTest: "+checks+" checks");
 }
}
