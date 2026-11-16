package org.viscereality.atmungsraum.boot;

/** No Android stubs: exercises the production admission/dispatch boundary. */
public final class BootHelperPolicyTest {
    private static int checks;
    private static final class Host implements BootPolicy.Host {
        long admitted = -1;
        boolean enabled = true, overlay = true, commit = true, failLaunch;
        int resolves, launches;
        String status, launched;
        BootPolicy.Target target = target("current.First", true, true, true);
        public boolean enabled() { return enabled; }
        public boolean overlayAllowed() { return overlay; }
        public long admittedBoot() { return admitted; }
        public boolean admit(long boot) { if (commit) admitted = boot; return commit; }
        public BootPolicy.Target resolveCurrentTarget() { resolves++; return target; }
        public void dispatch(BootPolicy.Target value) {
            if (admitted < 0) throw new AssertionError("dispatch precedes durable admission");
            launches++; launched = value.activity;
            if (failLaunch) throw new SecurityException("blocked by OS");
        }
        public void record(String value) { status = value; }
    }
    private static BootPolicy.Target target(String activity, boolean signer, boolean main, boolean launcher) {
        return new BootPolicy.Target("org.viscereality.atmungsraum.developer", activity, signer, main, launcher);
    }
    private static void expect(String expected, String actual) {
        checks++; if (!expected.equals(actual)) throw new AssertionError(expected + " != " + actual);
    }
    private static String boot(Host host, long count) { return BootPolicy.run(host, "android.intent.action.BOOT_COMPLETED", count); }
    public static void main(String[] args) {
        Host h = new Host();
        expect("IGNORED_ACTION", BootPolicy.run(h, "synthetic", 5));
        expect("-1:0:0", h.admitted + ":" + h.resolves + ":" + h.launches);
        expect("BOOT_COUNT_UNAVAILABLE", boot(h, -1));
        expect("-1", "" + h.admitted);
        h.enabled = false; expect("DISABLED", boot(h, 1));
        h.enabled = true; expect("ALREADY_ADMITTED", boot(h, 1));
        h.overlay = false; expect("OVERLAY_REQUIRED", boot(h, 2));
        expect("0", "" + h.resolves);
        h.overlay = true; h.target = null; expect("TARGET_ABSENT", boot(h, 3));
        h.target = target("current.First", false, true, true); expect("SIGNATURE_MISMATCH", boot(h, 4));
        h.target = new BootPolicy.Target("other.package", "Other", true, true, true);
        expect("INVALID_LAUNCHER", boot(h, 5));
        h.target = target("current.First", true, false, true); expect("INVALID_LAUNCHER", boot(h, 6));
        h.target = target("current.First", true, true, false); expect("INVALID_LAUNCHER", boot(h, 7));
        h.target = target("", true, true, true); expect("INVALID_LAUNCHER", boot(h, 8));
        h.target = target("current.First", true, true, true);
        h.commit = false; expect("ADMISSION_FAILED", boot(h, 9)); expect("0", "" + h.launches);
        h.commit = true; expect("LAUNCH_REQUESTED_UNCONFIRMED", boot(h, 9));
        expect("current.First", h.launched); expect("ALREADY_ADMITTED", boot(h, 9));
        expect("ALREADY_ADMITTED", boot(h, 8)); expect("1", "" + h.launches);
        h.target = target("updated.Second", true, true, true);
        expect("LAUNCH_REQUESTED_UNCONFIRMED", boot(h, 10)); expect("updated.Second", h.launched);
        h.failLaunch = true; expect("LAUNCH_FAILED", boot(h, 11));
        expect("ALREADY_ADMITTED", boot(h, 11)); expect("3", "" + h.launches);
        expect("LAUNCH_FAILED", h.status);
        System.out.println("PASS BootHelperPolicyTest: " + checks + " checks; fixed target, current launcher, durable once-per-boot, failure barriers");
    }
}
