package org.viscereality.atmungsraum.boot;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/** Observed installed-package/launch monitor. Time never proves download completion. */
public final class WatchPolicy {
    public static final long SAFE_INTEGER=9007199254740991L, COMMAND_MS=5000;
    public static final int MAX_ATTEMPTS=128;
    public static final class Result {
        public final boolean accepted; public final String reason;
        Result(boolean accepted,String reason){this.accepted=accepted;this.reason=reason;}
    }
    public static final class Watch {
        public final String kind,attempt;
        public final long installed,target,dispatchedCode,startedCode;
        public Watch(String kind,String attempt,long installed,long target,long dispatchedCode,long startedCode){
            this.kind=kind;this.attempt=attempt;this.installed=installed;this.target=target;
            this.dispatchedCode=dispatchedCode;this.startedCode=startedCode;
        }
        Watch dispatched(long code){return new Watch(kind,attempt,installed,target,code,startedCode);}
        Watch started(long code){return new Watch(kind,attempt,installed,target,dispatchedCode,code);}
        Watch rebooted(){return new Watch(kind,attempt,installed,target,0,0);}
    }
    public static final class Ledger {
        public final long boot; public final Set<String> attempts; public final Watch watch;
        public Ledger(long boot,Set<String> attempts,Watch watch){this.boot=boot;this.attempts=Collections.unmodifiableSet(new HashSet<String>(attempts));this.watch=watch;}
        Ledger with(Watch value){return new Ledger(boot,attempts,value);}
    }
    public interface Host {
        boolean enabled(); boolean overlayAllowed(); long currentBoot(); long now();
        Ledger load(); boolean save(Ledger value); BootPolicy.Target resolveCurrentTarget();
        void dispatchFallback(BootPolicy.Target target);
    }
    public static boolean exactCaller(String[] packages,int uid,int mainUid,boolean sameSigner){
        return uid>=0&&uid==mainUid&&sameSigner&&packages!=null&&packages.length==1&&BootPolicy.TARGET_PACKAGE.equals(packages[0]);
    }
    static boolean code(long n){return n>0&&n<=SAFE_INTEGER;}
    static boolean attempt(String n){
        if(n==null||!n.matches("[1-9][0-9]{0,18}"))return false;
        try{return Long.parseLong(n)>0;}catch(NumberFormatException invalid){return false;}
    }
    private static boolean target(BootPolicy.Target t){return t!=null&&BootPolicy.TARGET_PACKAGE.equals(t.packageName)&&t.sameSigner&&t.main&&t.launcher
        &&t.activity!=null&&!t.activity.isEmpty()&&code(t.version)&&(t.protocol==2||t.protocol==3);}
    private static Result result(boolean ok,String reason){return new Result(ok,reason);}
    /** Deadline bounds individual IPC effects, never the pending package monitor. */
    public static boolean commandLive(Host h,long boot,long deadline){
        long now=h.now();return boot>=0&&boot==h.currentBoot()&&now>=0&&deadline>=now&&deadline-now<=COMMAND_MS;
    }
    private static Ledger ledger(Host h){
        Ledger l=h.load();long boot=h.currentBoot();
        if(boot<0||l==null||boot<l.boot)throw new IllegalStateException("boot-state-unavailable");
        if(l.boot==boot)return l;
        // Only a newer real BOOT_COUNT resets consumption, not service/process death.
        Watch pending=l.watch!=null&&l.watch.kind.equals("update")?l.watch.rebooted():null;
        Set<String> used=new HashSet<String>();if(pending!=null)used.add(pending.attempt);
        return new Ledger(boot,used,pending);
    }
    private static boolean live(Host h,Ledger l){return l.watch!=null&&l.boot==h.currentBoot()&&h.enabled();}
    public static synchronized boolean watching(Host h){try{return live(h,ledger(h));}catch(RuntimeException unavailable){return false;}}
    public static synchronized String describe(Host h){
        try {
            Ledger l=ledger(h);if(!live(h,l))return "Inactive";
            Watch w=l.watch;BootPolicy.Target t=h.resolveCurrentTarget();
            String binding="attempt "+w.attempt+", target "+w.target+", boot "+l.boot;
            if(!h.overlayAllowed())return "Overlay permission required; monitoring "+binding;
            if(!target(t))return "Package identity unavailable; monitoring "+binding;
            if(w.kind.equals("boot")&&t.version==w.installed)return "Waiting for focused-ready acknowledgment; "+binding;
            if(t.version<w.target)return "Waiting for installed target; observed "+t.version+"; "+binding;
            if(w.startedCode==t.version)return "Launch started; waiting for focused-ready acknowledgment; "+binding;
            if(w.dispatchedCode!=0)return "Launch requested, unconfirmed; waiting for app acknowledgment; "+binding;
            if(!t.singleTask)return "Installed launcher is not singleTask; automatic launch blocked; "+binding;
            return "Installed target observed; launch pending; "+binding;
        }catch(RuntimeException unavailable){return "Monitor state unavailable";}
    }
    public static synchronized Result armUpdate(Host h,String id,long installed,long expected){
        return armUpdate(h,id,installed,expected,h.currentBoot(),h.now()+COMMAND_MS);
    }
    public static synchronized Result armUpdate(Host h,String id,long installed,long expected,long commandBoot,long commandDeadline){
        try {
            if(!commandLive(h,commandBoot,commandDeadline))return result(false,"stale-monitor-command");
            if(!attempt(id)||!code(installed)||!code(expected)||expected<=installed)return result(false,"invalid-update-binding");
            BootPolicy.Target t=h.resolveCurrentTarget();
            if(!target(t)||!t.singleTask||t.version!=installed)return result(false,"target-identity-disagreement");
            if(!h.enabled()||!h.overlayAllowed())return result(false,"helper-not-enabled-or-overlay-denied");
            Ledger l=ledger(h);
            if(!commandLive(h,commandBoot,commandDeadline))return result(false,"stale-monitor-command");
            if(l.attempts.contains(id))return result(live(h,l)&&l.watch.kind.equals("update")&&id.equals(l.watch.attempt)
                &&l.watch.installed==installed&&l.watch.target==expected,"attempt-replayed");
            if(live(h,l)&&l.watch.kind.equals("update"))return result(false,"watch-busy");
            if(l.attempts.size()>=MAX_ATTEMPTS)return result(false,"attempt-ledger-full");
            Set<String> used=new HashSet<String>(l.attempts);used.add(id);
            Watch w=new Watch("update",id,installed,expected,0,0);
            return h.save(new Ledger(l.boot,used,w))?result(true,"armed"):result(false,"persistence-unavailable");
        }catch(RuntimeException unavailable){return result(false,"helper-state-unavailable");}
    }
    public static synchronized Result armBoot(Host h,BootPolicy.Target t){
        try {
            if(!target(t)||!t.singleTask||t.version==SAFE_INTEGER||!h.enabled()||!h.overlayAllowed())return result(false,"legacy-or-unavailable-target");
            Ledger l=ledger(h);
            if(l.watch!=null)return result(false,"boot-watch-already-recorded");
            return h.save(l.with(new Watch("boot","boot",t.version,t.version+1,0,0)))?result(true,"armed"):result(false,"persistence-unavailable");
        }catch(RuntimeException unavailable){return result(false,"helper-state-unavailable");}
    }
    /** True only for a still-authorized update. Boot must not launch the old experiment. */
    public static synchronized boolean pendingUpdate(Host h){
        Watch w=h.load().watch;return w!=null&&w.kind.equals("update");
    }
    public static synchronized boolean resumeUpdate(Host h){
        try{Ledger l=ledger(h);return live(h,l)&&l.watch.kind.equals("update")&&h.save(l);}
        catch(RuntimeException unavailable){return false;}
    }
    public static synchronized Result ack(Host h,String method,long loadedCode){
        return ack(h,method,loadedCode,h.currentBoot(),h.now()+COMMAND_MS);
    }
    public static synchronized Result ack(Host h,String method,long loadedCode,long commandBoot,long commandDeadline){
        try {
            BootPolicy.Target t=h.resolveCurrentTarget();
            if(!target(t)||!code(loadedCode)||t.version!=loadedCode)return result(false,"loaded-package-identity-disagreement");
            if(!method.equals("launch-started")&&!method.equals("launch-ready")&&!method.equals("cancel-watch"))return result(false,"unsupported-method");
            Ledger l=ledger(h);
            if(!commandLive(h,commandBoot,commandDeadline))return result(false,"stale-monitor-command");
            if(method.equals("cancel-watch"))return h.save(l.with(null))?result(true,"watch-cancelled"):result(false,"persistence-unavailable");
            if(!live(h,l))return result(false,"no-live-watch");
            Watch w=l.watch;
            if(w.kind.equals("update")&&loadedCode<w.target)return result(false,"expected-newer-package-not-started");
            // A trusted start suppresses fallback but is not focused-ready completion.
            boolean ready=method.equals("launch-ready");
            return h.save(l.with(ready?null:w.started(loadedCode)))?result(true,ready?"launch-ready-acknowledged":"launch-started-awaiting-ready"):result(false,"persistence-unavailable");
        }catch(RuntimeException unavailable){return result(false,"helper-state-unavailable");}
    }
    public static synchronized boolean cancelInternal(Host h){try{Ledger l=ledger(h);return h.save(l.with(null));}catch(RuntimeException unavailable){return false;}}
    public static synchronized void cancelArm(Host h,String id,long installed,long expected){
        try{Ledger l=ledger(h);Watch w=l.watch;if(w!=null&&w.kind.equals("update")&&w.attempt.equals(id)&&w.installed==installed&&w.target==expected)h.save(l.with(null));}
        catch(RuntimeException unavailable){}
    }
    public static synchronized void poll(Host h){
        try {
            Ledger l=ledger(h);if(!live(h,l)){if(l.watch!=null)h.save(l.with(null));return;}
            if(h.load().boot!=l.boot&&!h.save(l))return;
            Watch w=l.watch;
            if(w.dispatchedCode!=0||!h.overlayAllowed())return;
            BootPolicy.Target t=h.resolveCurrentTarget();
            if(!target(t)||!t.singleTask||t.version<w.target||w.startedCode==t.version)return;
            // At most one fallback per actual boot. singleTask makes a simultaneous
            // Meta launch reuse the same Activity rather than starting an experiment twice.
            if(h.save(l.with(w.dispatched(t.version))))h.dispatchFallback(t);
        }catch(RuntimeException unavailable){/* no repeat after durable consumption */}
    }
}
