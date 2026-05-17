from __future__ import annotations

import math
from dataclasses import dataclass
from math import gcd


GCE_PREFIX = b"\x21\xf9\x04"


@dataclass(frozen=True)
class GifTimingNormalization:
    frame_count: int
    duration_before_cs: int
    duration_after_cs: int
    quantum_cs: int
    added_cs: int

    def to_dict(self) -> dict[str, int | float]:
        return {
            "frame_count": self.frame_count,
            "duration_before_cs": self.duration_before_cs,
            "duration_after_cs": self.duration_after_cs,
            "duration_before_s": round(self.duration_before_cs / 100.0, 3),
            "duration_after_s": round(self.duration_after_cs / 100.0, 3),
            "quantum_cs": self.quantum_cs,
            "quantum_s": round(self.quantum_cs / 100.0, 3),
            "added_cs": self.added_cs,
        }


def export_safe_gif_quantum_cs(fps: int) -> int:
    """
    Return the smallest duration quantum that both GIF centisecond timing and an
    integer-fps video export can represent exactly.

    For 30 fps, the common grid is 10 cs == 0.1 s == 3 frames.
    """
    fps = int(fps)
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return 100 // gcd(100, fps)


def normalize_gif_duration_to_export_grid(
    data: bytes,
    *,
    fps: int,
    policy: str = "export-grid",
) -> tuple[bytes, GifTimingNormalization | None]:
    """
    Extend a GIF onto an export-safe timing grid without touching image payloads.

    Only Graphic Control Extension delay words are patched. Pixel/frame bytes are
    preserved verbatim; when slack is needed, centiseconds are distributed across
    frames as evenly as possible so the retiming remains subtle.
    """
    delay_offsets = _graphic_control_delay_offsets(data)
    if not delay_offsets:
        return data, None

    delays_cs = [_read_u16le(data, offset) for offset in delay_offsets]
    duration_before_cs = sum(delays_cs)
    if duration_before_cs <= 0:
        return data, None

    normalized_policy = str(policy or "export-grid").strip().lower()
    if normalized_policy == "export-grid":
        quantum_cs = export_safe_gif_quantum_cs(fps)
        duration_after_cs = int(math.ceil(duration_before_cs / float(quantum_cs))) * quantum_cs
    elif normalized_policy == "ceil-second":
        quantum_cs = 100
        duration_after_cs = int(math.ceil(duration_before_cs / float(quantum_cs))) * quantum_cs
    elif normalized_policy == "nearest-second":
        quantum_cs = 100
        duration_after_cs = max(
            quantum_cs,
            int(math.floor((duration_before_cs / float(quantum_cs)) + 0.5)) * quantum_cs,
        )
    else:
        raise ValueError("policy must be 'export-grid', 'ceil-second', or 'nearest-second'")
    added_cs = duration_after_cs - duration_before_cs
    if added_cs == 0:
        return (
            data,
            GifTimingNormalization(
                frame_count=len(delays_cs),
                duration_before_cs=duration_before_cs,
                duration_after_cs=duration_after_cs,
                quantum_cs=quantum_cs,
                added_cs=0,
            ),
        )

    normalized_delays = _retime_delays_by_centiseconds(delays_cs, added_cs)
    patched = bytearray(data)
    for offset, delay_cs in zip(delay_offsets, normalized_delays, strict=True):
        patched[offset : offset + 2] = int(delay_cs).to_bytes(2, "little", signed=False)

    return (
        bytes(patched),
        GifTimingNormalization(
            frame_count=len(delays_cs),
            duration_before_cs=duration_before_cs,
            duration_after_cs=duration_after_cs,
            quantum_cs=quantum_cs,
            added_cs=added_cs,
        ),
    )


def _graphic_control_delay_offsets(data: bytes) -> list[int]:
    if len(data) < 13 or data[:6] not in {b"GIF87a", b"GIF89a"}:
        return []

    # Header (6) + logical screen descriptor (7)
    cursor = 13
    packed = data[10]
    if packed & 0x80:
        gct_size = 3 * (2 ** ((packed & 0x07) + 1))
        cursor += gct_size

    offsets: list[int] = []
    while cursor < len(data):
        marker = data[cursor]
        cursor += 1
        if marker == 0x3B:  # trailer
            break
        if marker == 0x21:  # extension
            if cursor >= len(data):
                break
            label = data[cursor]
            cursor += 1
            if label == 0xF9:  # graphic control extension
                if cursor + 6 > len(data):
                    break
                block_size = data[cursor]
                if block_size != 0x04:
                    break
                # GCE payload: [packed][delay_lo][delay_hi][transparent]
                delay_offset = cursor + 2
                terminator_offset = cursor + 5
                if terminator_offset >= len(data) or data[terminator_offset] != 0:
                    break
                offsets.append(delay_offset)
                cursor = terminator_offset + 1
            else:
                cursor = _skip_subblocks(data, cursor)
        elif marker == 0x2C:  # image descriptor
            if cursor + 9 > len(data):
                break
            packed = data[cursor + 8]
            cursor += 9
            if packed & 0x80:
                lct_size = 3 * (2 ** ((packed & 0x07) + 1))
                cursor += lct_size
            if cursor >= len(data):
                break
            cursor += 1  # LZW minimum code size
            cursor = _skip_subblocks(data, cursor)
        else:
            # Malformed/unsupported block marker.
            break
    return offsets


def _read_u16le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 2], "little", signed=False)


def _retime_delays_by_centiseconds(delays_cs: list[int], added_cs: int) -> list[int]:
    out = list(delays_cs)
    if added_cs == 0 or not out:
        return out

    frame_count = len(out)
    if added_cs > 0:
        base, remainder = divmod(added_cs, frame_count)
        if base:
            out = [delay + base for delay in out]
        if remainder:
            for i in range(remainder):
                idx = int(round((i * (frame_count - 1)) / max(1, remainder - 1)))
                out[idx] += 1
        return out

    remaining = abs(added_cs)
    candidates = [idx for idx, delay in enumerate(out) if delay > 1]
    if remaining > sum(max(0, out[idx] - 1) for idx in candidates):
        raise ValueError("Cannot shorten GIF timing without dropping a frame delay below 1 cs.")
    cursor = 0
    while remaining > 0 and candidates:
        idx = candidates[cursor % len(candidates)]
        if out[idx] > 1:
            out[idx] -= 1
            remaining -= 1
        cursor += 1
        if cursor % len(candidates) == 0:
            candidates = [i for i in candidates if out[i] > 1]
    return out


def _skip_subblocks(data: bytes, cursor: int) -> int:
    while cursor < len(data):
        block_size = data[cursor]
        cursor += 1
        if block_size == 0:
            return cursor
        cursor += block_size
    return cursor
