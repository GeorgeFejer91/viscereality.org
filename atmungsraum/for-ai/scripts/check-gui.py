"""Run the production smartphone rules against the exact branch being pushed."""
from pathlib import Path
from subprocess import check_output, run
from sys import executable, stdin
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import argparse

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent


def verify(project):
    return run([executable, "-B", str(project / "tools" / "test-smartphone-gui.py")],
               cwd=project).returncode


def verify_ref(ref):
    revision = check_output(["git", "rev-parse", "--verify", ref + "^{commit}"], cwd=REPO, text=True).strip()
    with TemporaryDirectory(prefix="atmungsraum-gui-") as directory:
        snapshot = Path(directory)
        archive = snapshot / "candidate.zip"
        run(["git", "archive", "--format=zip", f"--output={archive}", "--prefix=atmungsraum/", revision + ":atmungsraum"],
            cwd=REPO, check=True)
        with ZipFile(archive) as source:
            source.extractall(snapshot)
        print(f"Verifying smartphone GUI at {revision}", flush=True)
        return verify(snapshot / "atmungsraum")


def changed(candidate, previous, remote):
    if not previous.strip("0"):
        comparison = run(["git", "merge-base", candidate, f"refs/remotes/{remote}/main"],
                         cwd=REPO, capture_output=True, text=True)
        if comparison.returncode:
            return True
        previous = comparison.stdout.strip()
    return run(["git", "diff", "--quiet", previous, candidate, "--", "atmungsraum",
                ":(exclude)atmungsraum/**/*.md", ":(exclude)atmungsraum/quest/**",
                ":(exclude)atmungsraum/audio-assets/**"], cwd=REPO).returncode != 0


def pre_push(remote):
    checked = set()
    for line in stdin:
        _, candidate, remote_ref, previous = line.split()
        if not remote_ref.startswith("refs/heads/") or not candidate.strip("0") or candidate in checked:
            continue
        if changed(candidate, previous, remote) and verify_ref(candidate):
            return 1
        checked.add(candidate)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pre-push", metavar="REMOTE")
    mode.add_argument("--ref")
    args = parser.parse_args()
    if args.pre_push is not None:
        raise SystemExit(pre_push(args.pre_push))
    raise SystemExit(verify_ref(args.ref) if args.ref else verify(ROOT))
