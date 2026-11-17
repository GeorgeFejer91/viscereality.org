package org.viscereality.atmungsraum.boot;
import java.util.*;
/** A long observed download is not an expired command or completed installation. */
public final class MonitorPolicyTest {
 static int checks;
 static void check(boolean value,String message){checks++;if(!value)throw new AssertionError(message);}
 static final class Host implements WatchPolicy.Host {
  long boot=8,now=1000;boolean enabled=true,overlay=true,commit=true;int launches;
  WatchPolicy.Ledger saved=new WatchPolicy.Ledger(8,new HashSet<String>(),null);
  BootPolicy.Target target=target(20,true,true);
  boolean ackDuringDispatch;
  public boolean enabled(){return enabled;}public boolean overlayAllowed(){return overlay;}
  public long currentBoot(){return boot;}public long now(){return now;}
  public WatchPolicy.Ledger load(){return saved;}public boolean save(WatchPolicy.Ledger next){if(commit)saved=next;return commit;}
  public BootPolicy.Target resolveCurrentTarget(){return target;}
  public void dispatchFallback(BootPolicy.Target target){check(saved.watch.dispatchedCode==target.version,"admit before dispatch");launches++;if(ackDuringDispatch)WatchPolicy.ack(this,"launch-started",target.version);}
 }
 static BootPolicy.Target target(long code,boolean signer,boolean single){return new BootPolicy.Target(BootPolicy.TARGET_PACKAGE,"Current",signer,true,true,code,3,single);}
 static Host armed(){Host h=new Host();check(WatchPolicy.armUpdate(h,"42",20,21).accepted,"arm actual update");return h;}
 public static void main(String[] args){
  Host h=armed();
  h.now+=9*60*60*1000L;WatchPolicy.poll(h);
  check(WatchPolicy.watching(h)&&h.launches==0,"nine-hour download remains monitored without false expiry or completion");
  h.target=target(21,true,true);WatchPolicy.poll(h);
  check(h.launches==1&&h.saved.watch!=null,"actual PM replacement launches once immediately without estimated Meta grace");
  WatchPolicy.poll(h);h.now+=86400000L;WatchPolicy.poll(h);check(h.launches==1,"no repeat while waiting for actual start ACK");
  Host restart=new Host();restart.saved=h.saved;restart.target=h.target;WatchPolicy.poll(restart);check(restart.launches==0,"helper process restart preserves dispatch consumption");
  check(!WatchPolicy.ack(h,"launch-started",20).accepted,"old loaded process cannot claim replacement launch");
  check(!WatchPolicy.ack(h,"launch-ready",22).accepted,"claimed future code not equal to PM rejected");
  check(WatchPolicy.ack(h,"launch-started",21).accepted&&WatchPolicy.watching(h),"real start suppresses fallback but does not claim focus");
  h.now+=86400000L;WatchPolicy.poll(h);check(h.launches==1,"slow first focus never causes another launcher request");
  check(WatchPolicy.ack(h,"launch-ready",21).accepted&&!WatchPolicy.watching(h),"PM-equal focused-ready completes monitor");
  check(!WatchPolicy.ack(h,"launch-ready",21).accepted,"completed ACK replay cannot recreate monitor");
  h=armed();h.target=target(21,true,true);WatchPolicy.ack(h,"launch-started",21);WatchPolicy.poll(h);
  check(h.launches==0&&WatchPolicy.watching(h),"Meta warm start ACK before poll suppresses helper launch");
  h=armed();h.target=target(21,true,true);h.ackDuringDispatch=true;WatchPolicy.poll(h);WatchPolicy.poll(h);
  check(h.launches==1&&h.saved.watch.startedCode==21,"Meta/helper concurrent start consumes only one helper request and retains real started ACK");
  h=armed();WatchPolicy.cancelInternal(h);h.target=target(21,true,true);h.boot++;WatchPolicy.poll(h);
  check(h.launches==0&&!WatchPolicy.watching(h),"explicit cancel survives replacement and reboot");
  h=armed();h.boot++;h.now=10;check(WatchPolicy.resumeUpdate(h),"pending target recovers after real reboot");
  WatchPolicy.poll(h);check(h.launches==0&&h.saved.boot==9,"resume never launches outdated installed code");
  h.target=target(21,true,true);WatchPolicy.poll(h);check(h.launches==1,"new boot observes real replacement and launches");
  h.boot++;h.now=5;WatchPolicy.poll(h);check(h.launches==2,"only newer real boot resets one-per-boot launch admission");
  WatchPolicy.poll(h);check(h.launches==2,"duplicate boot/poll cannot reset admission");
  h=armed();h.target=target(21,true,true);h.overlay=false;WatchPolicy.poll(h);
  check(h.launches==0&&WatchPolicy.watching(h),"permission loss blocks launch without inventing download failure");
  h.overlay=true;WatchPolicy.poll(h);check(h.launches==1,"restored authority can launch observed replacement once");
  h=armed();h.enabled=false;WatchPolicy.poll(h);h.enabled=true;h.target=target(21,true,true);WatchPolicy.poll(h);
  check(h.launches==0&&h.saved.watch==null,"explicit disable cancels durably");
  h=armed();h.target=target(21,false,true);WatchPolicy.poll(h);check(h.launches==0,"signer mismatch never dispatches");
  h=armed();h.target=target(21,true,false);WatchPolicy.poll(h);check(h.launches==0,"non-singleTask replacement cannot race a duplicate Activity");
  h=armed();h.target=null;WatchPolicy.poll(h);check(WatchPolicy.watching(h)&&h.launches==0,"package replacement gap is not completion or abandonment");
  h.target=target(22,true,true);WatchPolicy.poll(h);check(h.launches==1,"actual code above target qualifies");
  h=armed();h.target=target(21,true,true);h.commit=false;WatchPolicy.poll(h);check(h.launches==0,"failed durable dispatch admission blocks launch");
  System.out.println("PASS MonitorPolicyTest: "+checks+" checks");
 }
}
