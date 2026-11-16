package org.viscereality.atmungsraum.boot;
import java.util.*;
public final class WatchPolicyTest {
 static int checks;
 static void check(boolean x,String m){checks++;if(!x)throw new AssertionError(m);}
 static class Host implements WatchPolicy.Host {
  long boot=8,now=1000; boolean enabled=true,overlay=true,commit=true; int launches; long launched;
  WatchPolicy.Ledger saved=new WatchPolicy.Ledger(8,new HashSet<String>(),null);
  BootPolicy.Target target=target(19,2,true);
  public boolean enabled(){return enabled;} public boolean overlayAllowed(){return overlay;}
  public long currentBoot(){return boot;}public long now(){return now;}
  public WatchPolicy.Ledger load(){return saved;}public boolean save(WatchPolicy.Ledger x){if(commit)saved=x;return commit;}
  public BootPolicy.Target resolveCurrentTarget(){return target;}
  public void dispatchFallback(BootPolicy.Target t){check(saved.watch.dispatchedCode==t.version,"durable version consumed before launch");launches++;launched=t.version;}
 }
 static BootPolicy.Target target(long code,int protocol,boolean signer){return new BootPolicy.Target(BootPolicy.TARGET_PACKAGE,"Current",signer,true,true,code,protocol);}
 static void arm(Host h){check(WatchPolicy.armUpdate(h,"42",19,20).accepted,"arm expected update");}
 static void replacement(Host h){h.target=target(20,2,true);WatchPolicy.poll(h);h.now+=WatchPolicy.META_GRACE_MS;WatchPolicy.poll(h);}
 static void bootReplacementStartedStopsFallback(){
  Host h=new Host();check(WatchPolicy.armBoot(h,h.target).accepted,"boot watch armed for replacement-start regression");
  check(WatchPolicy.ack(h,"launch-started",19).accepted&&h.saved.watch!=null,"original installed-code start retains boot watch until ready");
  h.target=target(20,2,true);WatchPolicy.poll(h);
  check(!WatchPolicy.ack(h,"launch-started",19).accepted&&h.saved.watch!=null,"old loaded code cannot retire replaced boot watch");
  check(!WatchPolicy.ack(h,"launch-started",21).accepted&&h.saved.watch!=null,"future claimed code must equal actual PM identity");
  check(WatchPolicy.ack(h,"launch-started",20).accepted,"replacement trusted start acknowledged");
  h.now+=WatchPolicy.META_GRACE_MS;WatchPolicy.poll(h);
  check(h.launches==0&&h.saved.watch==null,"acknowledged replacement launch must stop boot fallback despite slow first frame");
  Host recreated=new Host();recreated.saved=h.saved;recreated.now=h.now;recreated.target=h.target;WatchPolicy.poll(recreated);
  check(recreated.launches==0&&recreated.saved.watch==null,"retired boot watch cannot revive after service recreation");
  check(!WatchPolicy.ack(h,"launch-started",20).accepted&&!WatchPolicy.ack(h,"launch-ready",20).accepted,"replayed completed ACKs cannot recreate watch");
  h=new Host();WatchPolicy.armBoot(h,h.target);h.target=target(21,2,true);
  check(WatchPolicy.ack(h,"launch-started",21).accepted&&h.saved.watch==null,"actual code above expected target also retires boot watch");
  h=new Host();WatchPolicy.armBoot(h,h.target);h.target=target(20,2,true);
  check(WatchPolicy.ack(h,"launch-ready",20).accepted&&h.saved.watch==null,"new ready ACK still retires boot watch");
 }
 public static void main(String[] a){
  bootReplacementStartedStopsFallback();
  check(WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE},25,25,true),"exact current signer/UID");
  check(!WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE,"shared"},25,25,true),"shared UID denied");
  check(!WatchPolicy.exactCaller(new String[]{"other"},25,25,true),"other samecert package denied");
  check(!WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE},24,25,true),"wrong UID denied");
  check(!WatchPolicy.exactCaller(new String[]{BootPolicy.TARGET_PACKAGE},25,25,false),"wrong signer denied");
  Host h=new Host();arm(h);check(WatchPolicy.armUpdate(h,"42",19,20).accepted,"duplicate arm returns same acknowledgment");
  check(h.saved.watch.deadline==1000+WatchPolicy.WATCH_MS,"duplicate delivery cannot extend expiry");
  h.target=target(20,2,true);WatchPolicy.poll(h);check(h.launches==0,"Meta gets replacement grace");
  h.now+=WatchPolicy.META_GRACE_MS-1;WatchPolicy.poll(h);check(h.launches==0,"no early fallback");
  h.now++;WatchPolicy.poll(h);check(h.launches==1&&h.launched==20,"fallback resolves actual expected-newer binary");
  WatchPolicy.poll(h);check(h.launches==1,"two deliveries cannot duplicate fallback");
  Host restarted=new Host();restarted.saved=h.saved;restarted.now=h.now;restarted.target=h.target;WatchPolicy.poll(restarted);
  check(restarted.launches==0,"process death cannot replay consumed version");
  check(!WatchPolicy.ack(h,"launch-started",19).accepted,"replaced old loaded process cannot ack");
  check(WatchPolicy.ack(h,"launch-started",20).accepted&&h.saved.watch==null,"new trusted started ends update watch");
  check(!WatchPolicy.armUpdate(h,"42",19,20).accepted,"completed attempt cannot rearm");
  h=new Host();arm(h);check(WatchPolicy.ack(h,"cancel-watch",19).accepted,"protected escape cancels");replacement(h);
  check(h.launches==0,"escape survives replacement and prevents fallback");
  h=new Host();arm(h);h.boot++;replacement(h);check(h.launches==0,"old boot watch cannot cross reboot");
  h=new Host();arm(h);h.now+=WatchPolicy.WATCH_MS;replacement(h);check(h.launches==0,"expiry cannot launch");
  h=new Host();arm(h);h.commit=false;replacement(h);check(h.launches==0,"failed durable fallback consumption cannot launch");
  h=new Host();h.commit=false;check(!WatchPolicy.armUpdate(h,"42",19,20).accepted,"failed arm persistence denied");
  for(String bad:new String[]{"","0","01","-1","+2","9223372036854775808"}){h=new Host();check(!WatchPolicy.armUpdate(h,bad,19,20).accepted,"canonical bounded attempt");}
  for(long[] pair:new long[][]{{19,19},{19,18},{18,20},{19,9007199254740992L},{-1,20}}){h=new Host();check(!WatchPolicy.armUpdate(h,"42",pair[0],pair[1]).accepted,"actual package and strict newer safe version");}
  h=new Host();h.overlay=false;check(!WatchPolicy.armUpdate(h,"42",19,20).accepted,"overlay eligibility required");
  h=new Host();h.enabled=false;check(!WatchPolicy.armUpdate(h,"42",19,20).accepted,"retained checkbox required");
  h=new Host();h.target=target(19,1,true);check(!WatchPolicy.armUpdate(h,"42",19,20).accepted,"legacy target cannot arm");
  check(!WatchPolicy.armBoot(h,h.target).accepted&&h.saved.watch==null,"legacy boot remains one shot");
  h=new Host();h.target=target(19,2,false);check(!WatchPolicy.armUpdate(h,"42",19,20).accepted,"changed signer denied");
  h=new Host();check(WatchPolicy.armBoot(h,h.target).accepted,"compatible original boot arms durable watch");
  check(WatchPolicy.ack(h,"launch-started",19).accepted&&h.saved.watch!=null,"started keeps original boot unconfirmed");
  replacement(h);check(h.launches==1,"replacement during unconfirmed original boot recovered");
  check(WatchPolicy.ack(h,"launch-ready",20).accepted&&h.saved.watch==null,"focused first frame ends boot watch");
  h=new Host();WatchPolicy.armBoot(h,h.target);WatchPolicy.ack(h,"launch-ready",19);replacement(h);
  check(h.launches==0,"confirmed original boot has no indefinite watcher");
  h=new Host();arm(h);h.overlay=false;replacement(h);check(h.launches==0,"revoked overlay cancels watch");
  h=new Host();for(int i=1;i<=WatchPolicy.MAX_ATTEMPTS;i++){check(WatchPolicy.armUpdate(h,""+i,19,20).accepted,"bounded attempt ledger admission");WatchPolicy.ack(h,"cancel-watch",19);}
  check(!WatchPolicy.armUpdate(h,"9999",19,20).accepted,"ledger exhaustion fails closed without eviction");
  check(!WatchPolicy.armUpdate(h,"1",19,20).accepted,"old replay is still retained at exhaustion");
  System.out.println("PASS WatchPolicyTest: "+checks+" checks");
 }
}
