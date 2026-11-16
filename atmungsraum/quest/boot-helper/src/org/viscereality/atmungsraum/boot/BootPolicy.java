package org.viscereality.atmungsraum.boot;

public final class BootPolicy {
    public static final String TARGET_PACKAGE = "org.viscereality.atmungsraum.developer";
    public static final class Target {
        public final String packageName, activity;
        public final boolean sameSigner, main, launcher;
        public Target(String packageName, String activity, boolean sameSigner, boolean main, boolean launcher) {
            this.packageName = packageName; this.activity = activity; this.sameSigner = sameSigner;
            this.main = main; this.launcher = launcher;
        }
    }
    public interface Host {
        boolean enabled(); boolean overlayAllowed(); long admittedBoot(); boolean admit(long boot);
        Target resolveCurrentTarget(); void dispatch(Target target); void record(String status);
    }
    public static synchronized String run(Host host, String action, long boot) {
        // BOOT_COMPLETED is an Android protected broadcast. Only the non-exported
        // receiver calls here; this API is not exposed to Intent extras or IPC.
        if (!"android.intent.action.BOOT_COMPLETED".equals(action)) return "IGNORED_ACTION";
        if (boot < 0) return "BOOT_COUNT_UNAVAILABLE";
        if (boot <= host.admittedBoot()) return "ALREADY_ADMITTED";
        // Commit even a denied attempt. Enabling setup, restoring permission or
        // exiting the target during this boot never creates a retry/watchdog.
        if (!host.admit(boot)) return "ADMISSION_FAILED";
        String status;
        try {
            if (!host.enabled()) status = "DISABLED";
            else if (!host.overlayAllowed()) status = "OVERLAY_REQUIRED";
            else {
                Target target = host.resolveCurrentTarget();
                if (target == null) status = "TARGET_ABSENT";
                else if (!TARGET_PACKAGE.equals(target.packageName) ||
                        target.activity == null || target.activity.isEmpty() || !target.main || !target.launcher)
                    status = "INVALID_LAUNCHER";
                else if (!target.sameSigner) status = "SIGNATURE_MISMATCH";
                else {
                    host.record("LAUNCH_DISPATCH_PENDING_UNCONFIRMED");
                    host.dispatch(target);
                    status = "LAUNCH_REQUESTED_UNCONFIRMED";
                }
            }
        } catch (RuntimeException failure) {
            status = "LAUNCH_FAILED";
        }
        host.record(status);
        return status;
    }
}
