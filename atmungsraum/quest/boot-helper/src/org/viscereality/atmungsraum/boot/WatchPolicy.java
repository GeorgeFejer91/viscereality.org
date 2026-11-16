package org.viscereality.atmungsraum.boot;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/** Finite, durable launch assistance. Never installs or treats dispatch as readiness. */
public final class WatchPolicy {
    public static final long WATCH_MS=300000, META_GRACE_MS=20000, SAFE_INTEGER=9007199254740991L;
    public static final int MAX_ATTEMPTS=128;
    public static final class Result {
        public final boolean accepted; public final String reason;
        Result(boolean accepted,String reason){this.accepted=accepted;this.reason=reason;}
    }
    public static final class Watch {
        public final String kind,attempt;
        public final long installed,target,deadline,detectedCode,notBefore,dispatchedCode,startedCode;
        public Watch(String kind,String attempt,long installed,long target,long deadline,long detectedCode,long notBefore,long dispatchedCode,long startedCode){
            this.kind=kind;this.attempt=attempt;this.installed=installed;this.target=target;this.deadline=deadline;
            this.detectedCode=detectedCode;this.notBefore=notBefore;this.dispatchedCode=dispatchedCode;this.startedCode=startedCode;
        }
        Watch detected(long code,long time){return new Watch(kind,attempt,installed,target,deadline,code,time,dispatchedCode,startedCode);}
        Watch dispatched(long code){return new Watch(kind,attempt,installed,target,deadline,detectedCode,notBefore,code,startedCode);}
        Watch started(long code){return new Watch(kind,attempt,installed,target,deadline,detectedCode,notBefore,dispatchedCode,code);}
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
    private static boolean code(long n){return n>0&&n<=SAFE_INTEGER;}
    private static boolean attempt(String n){
        if(n==null||!n.matches("[1-9][0-9]{0,18}"))return false;
        try{return Long.parseLong(n)>0;}catch(NumberFormatException invalid){return false;}
    }
    private static boolean target(BootPolicy.Target t){return t!=null&&BootPolicy.TARGET_PACKAGE.equals(t.packageName)&&t.sameSigner&&t.main&&t.launcher
        &&t.activity!=null&&!t.activity.isEmpty()&&code(t.version)&&t.protocol==2;}
    private static Result result(boolean ok,String reason){return new Result(ok,reason);}
    private static Ledger ledger(Host h){
        Ledger l=h.load();long boot=h.currentBoot();
        if(boot<0||l==null)throw new IllegalStateException("boot-state-unavailable");
        return l.boot==boot?l:new Ledger(boot,Collections.<String>emptySet(),null);
    }
    private static boolean live(Host h,Ledger l){return l.watch!=null&&l.boot==h.currentBoot()&&h.now()>=0&&h.now()<l.watch.deadline&&h.enabled()&&h.overlayAllowed();}
    public static synchronized boolean watching(Host h){try{return live(h,ledger(h));}catch(RuntimeException unavailable){return false;}}
    public static synchronized Result armUpdate(Host h,String id,long installed,long expected){
        try {
            if(!attempt(id)||!code(installed)||!code(expected)||expected<=installed)return result(false,"invalid-update-binding");
            BootPolicy.Target t=h.resolveCurrentTarget();
            if(!target(t)||t.version!=installed)return result(false,"target-identity-disagreement");
            if(!h.enabled()||!h.overlayAllowed())return result(false,"helper-not-enabled-or-overlay-denied");
            Ledger l=ledger(h);
            if(l.attempts.contains(id))return result(live(h,l)&&l.watch.kind.equals("update")&&id.equals(l.watch.attempt)
                &&l.watch.installed==installed&&l.watch.target==expected,"attempt-replayed");
            if(live(h,l)&&l.watch.kind.equals("update"))return result(false,"watch-busy");
            if(l.attempts.size()>=MAX_ATTEMPTS)return result(false,"attempt-ledger-full");
            Set<String> used=new HashSet<String>(l.attempts);used.add(id);
            Watch w=new Watch("update",id,installed,expected,h.now()+WATCH_MS,0,0,0,0);
            return h.save(new Ledger(l.boot,used,w))?result(true,"armed"):result(false,"persistence-unavailable");
        }catch(RuntimeException unavailable){return result(false,"helper-state-unavailable");}
    }
    /** Internal boot path only, called after BootPolicy's durable once-per-boot admission. */
    public static synchronized Result armBoot(Host h,BootPolicy.Target t){
        try {
            if(!target(t)||t.version==SAFE_INTEGER||!h.enabled()||!h.overlayAllowed())return result(false,"legacy-or-unavailable-target");
            Ledger l=ledger(h);
            if(l.watch!=null)return result(false,"boot-watch-already-recorded");
            Watch w=new Watch("boot","boot",t.version,t.version+1,h.now()+WATCH_MS,0,0,0,0);
            return h.save(l.with(w))?result(true,"armed"):result(false,"persistence-unavailable");
        }catch(RuntimeException unavailable){return result(false,"helper-state-unavailable");}
    }
    public static synchronized Result ack(Host h,String method,long loadedCode){
        try {
            BootPolicy.Target t=h.resolveCurrentTarget();
            if(!target(t)||!code(loadedCode)||t.version!=loadedCode)return result(false,"loaded-package-identity-disagreement");
            if(!method.equals("launch-started")&&!method.equals("launch-ready")&&!method.equals("cancel-watch"))return result(false,"unsupported-method");
            Ledger l=ledger(h);
            if(method.equals("cancel-watch"))return h.save(l.with(null))?result(true,"watch-cancelled"):result(false,"persistence-unavailable");
            if(!live(h,l))return result(false,"no-live-watch");
            Watch w=l.watch;
            if(w.kind.equals("update")&&loadedCode<w.target)return result(false,"expected-newer-package-not-started");
            // Original boot startup stays unconfirmed until ready; a trusted PM-equal
            // replacement startup proves Meta already relaunched, even before focus.
            boolean complete=method.equals("launch-ready")||w.kind.equals("update")||loadedCode>=w.target;
            return h.save(l.with(complete?null:w.started(loadedCode)))?result(true,complete?"launch-acknowledged":"boot-started-unconfirmed"):result(false,"persistence-unavailable");
        }catch(RuntimeException unavailable){return result(false,"helper-state-unavailable");}
    }
    public static synchronized void cancelInternal(Host h){try{Ledger l=ledger(h);h.save(l.with(null));}catch(RuntimeException unavailable){}}
    public static synchronized void poll(Host h){
        try {
            Ledger l=ledger(h);if(!live(h,l)){if(l.watch!=null)h.save(l.with(null));return;}
            Watch w=l.watch;
            if(w.dispatchedCode!=0)return; // Stronger than per-version: one fallback total for this watch.
            BootPolicy.Target t=h.resolveCurrentTarget();
            if(!target(t)||t.version<w.target)return;
            if(w.detectedCode!=t.version){h.save(l.with(w.detected(t.version,h.now()+META_GRACE_MS)));return;}
            if(h.now()<w.notBefore)return;
            // Consumption survives process death, launch exceptions and duplicate package events.
            if(h.save(l.with(w.dispatched(t.version))))h.dispatchFallback(t);
        }catch(RuntimeException unavailable){/* no retry of a durably consumed launch */}
    }
}
