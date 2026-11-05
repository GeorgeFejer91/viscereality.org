"""Generate the shared, participant-independent NeuroKit2 demo trace."""
import hashlib
import json
from pathlib import Path

import neurokit2 as nk
import numpy as np
import scipy

target = Path(__file__).resolve().parents[1] / "assets" / "ecg"
target.mkdir(parents=True, exist_ok=True)
settings = dict(duration=1800, sampling_rate=256, heart_rate=70,
                heart_rate_std=1, noise=0.01, method="ecgsyn", random_state=42)
signal = nk.ecg_simulate(**settings).astype("<f4")
assert len(signal) == settings["duration"] * settings["sampling_rate"]
assert np.isfinite(signal).all() and np.ptp(signal) > 0
content = signal.tobytes()
(target / "mock-ecg.f32").write_bytes(content)
manifest = dict(schema_version=1, source="mock-neurokit2", unit="a.u.",
                generator="neurokit2.ecg_simulate", neurokit2_version=nk.__version__,
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                settings=settings, samples=len(signal), encoding="float32-le",
                file="mock-ecg.f32", sha256=hashlib.sha256(content).hexdigest(),
                playback="Continuous page-clock replay; wraps after 1800 seconds.")
(target / "mock-ecg.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Generated {len(signal)} NeuroKit2 samples; SHA-256 {manifest['sha256']}")
