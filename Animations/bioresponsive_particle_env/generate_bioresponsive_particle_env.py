#!/usr/bin/env python3
"""
Bioresponsive Particle Environment GIF Generator

Generates an animated GIF showing an icosphere particle orb with ECG-triggered
visual effects including pulse displacement, color waves, and synchronization.

This version is fully self-contained on Windows:
- Uses Pillow for rendering (no Cairo dependency)
- Uses direct GIF export (no ffmpeg dependency)
- Converts HSL to explicit RGB to preserve particle color in GIF output

Usage:
    py -3 generate_bioresponsive_particle_env.py

Optional flags:
    --ecg-csv <path>
    --output <path>
    --fps <int>
    --duration <seconds>
    --subdivisions <int>
    --canvas-size <int>
"""

import argparse
import colorsys
import csv
import math
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ECG_CSV = SCRIPT_DIR / "synthetic_ecg_waveform.csv"
DEFAULT_OUTPUT_GIF = SCRIPT_DIR / "bioresponsive_particle_env.gif"

# Default visual parameters
DEFAULT_CANVAS_SIZE = 500
DEFAULT_ANIMATION_DURATION = 6.0
DEFAULT_BEAT_INTERVAL = 2.0
DEFAULT_FPS = 30
DEFAULT_ICOSPHERE_SUBDIVISIONS = 2  # 2 -> 162 particles
DEFAULT_PARTICLE_BASE_SIZE = 11
DEFAULT_RANDOM_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a bioresponsive particle environment GIF.")
    parser.add_argument("--ecg-csv", default=str(DEFAULT_ECG_CSV), help="Path to ECG waveform CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_GIF), help="Output GIF path")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second")
    parser.add_argument("--duration", type=float, default=DEFAULT_ANIMATION_DURATION, help="Animation duration in seconds")
    parser.add_argument("--beat-interval", type=float, default=DEFAULT_BEAT_INTERVAL, help="Seconds between ECG beats")
    parser.add_argument("--subdivisions", type=int, default=DEFAULT_ICOSPHERE_SUBDIVISIONS, help="Icosphere subdivisions")
    parser.add_argument("--canvas-size", type=int, default=DEFAULT_CANVAS_SIZE, help="Square canvas size in pixels")
    parser.add_argument("--particle-size", type=float, default=DEFAULT_PARTICLE_BASE_SIZE, help="Base particle radius")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for stable output")
    return parser.parse_args()


def read_ecg_data(filepath):
    """Read ECG waveform from CSV file with columns: time,value."""
    times = []
    values = []

    with open(filepath, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            raise ValueError("ECG CSV appears empty.")

        for row in reader:
            if len(row) < 2:
                continue
            times.append(float(row[0]))
            values.append(float(row[1]))

    if len(times) < 3:
        raise ValueError("ECG CSV must contain at least 3 samples.")

    return times, values


def find_r_peaks(times, values):
    """Detect simple R-peaks from an ECG signal."""
    threshold = 0.8 * max(values)
    min_distance = 0.3
    r_peaks = []
    last_peak_time = -1e9

    for i in range(1, len(values) - 1):
        if values[i] > threshold and values[i] > values[i - 1] and values[i] > values[i + 1]:
            if times[i] - last_peak_time > min_distance:
                r_peaks.append((times[i], values[i], i))
                last_peak_time = times[i]

    return r_peaks


def normalize(v):
    """Normalize a 3D vector to unit length."""
    length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return [v[0] / length, v[1] / length, v[2] / length]


def midpoint(v1, v2):
    """Compute normalized midpoint between two vertices."""
    return normalize([(v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2, (v1[2] + v2[2]) / 2])


def generate_icosphere(subdivisions):
    """Generate icosphere vertices using midpoint subdivision."""
    t = (1 + math.sqrt(5)) / 2

    vertices = [
        normalize([-1, t, 0]), normalize([1, t, 0]),
        normalize([-1, -t, 0]), normalize([1, -t, 0]),
        normalize([0, -1, t]), normalize([0, 1, t]),
        normalize([0, -1, -t]), normalize([0, 1, -t]),
        normalize([t, 0, -1]), normalize([t, 0, 1]),
        normalize([-t, 0, -1]), normalize([-t, 0, 1]),
    ]

    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    for _ in range(subdivisions):
        new_faces = []
        edge_cache = {}

        def get_midpoint(i1, i2):
            key = (min(i1, i2), max(i1, i2))
            if key not in edge_cache:
                mp = midpoint(vertices[i1], vertices[i2])
                edge_cache[key] = len(vertices)
                vertices.append(mp)
            return edge_cache[key]

        for v1, v2, v3 in faces:
            a = get_midpoint(v1, v2)
            b = get_midpoint(v2, v3)
            c = get_midpoint(v3, v1)

            new_faces.extend([
                [v1, a, c],
                [v2, b, a],
                [v3, c, b],
                [a, b, c],
            ])

        faces = new_faces

    return vertices


def rotate_y(v, angle):
    """Rotate a 3D vector around Y axis."""
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return [v[0] * cos_a + v[2] * sin_a, v[1], -v[0] * sin_a + v[2] * cos_a]


def hsl_to_rgb_int(h, s, l):
    """Convert HSL values to RGB 0-255 ints."""
    # colorsys uses HLS where lightness is the second arg.
    r, g, b = colorsys.hls_to_rgb((h % 360) / 360.0, max(0.0, min(1.0, l / 100.0)), max(0.0, min(1.0, s / 100.0)))
    return int(r * 255), int(g * 255), int(b * 255)


def draw_filled_circle(draw, cx, cy, radius, fill_rgba):
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill_rgba)


def draw_horizontal_sweep(draw, x_left, y_top, width, height):
    """Draw a soft horizontal alpha gradient sweep rectangle."""
    for i in range(int(width)):
        t = i / max(width - 1, 1)
        # Bell-like alpha profile, darkest in middle section
        alpha = int(180 * math.exp(-((t - 0.42) ** 2) / 0.04))
        if alpha <= 0:
            continue
        x = x_left + i
        draw.line((x, y_top, x, y_top + height), fill=(0, 0, 0, alpha), width=1)


def generate_frame_image(frame, particles, ecg_coords, r_peak_times, cfg):
    fps = cfg["fps"]
    t_anim = frame / fps

    canvas_size = cfg["canvas_size"]
    sphere_center_x = cfg["sphere_center_x"]
    sphere_center_y = cfg["sphere_center_y"]
    sphere_radius = cfg["sphere_radius"]
    particle_base_size = cfg["particle_base_size"]

    ecg_x_offset = cfg["ecg_x_offset"]
    ecg_y_offset = cfg["ecg_y_offset"]
    ecg_height = cfg["ecg_height"]
    x_per_sec = cfg["x_per_sec"]

    # Compute ECG dot position
    current_x = ecg_x_offset + t_anim * x_per_sec
    dot_y = ecg_y_offset + ecg_height / 2
    for i in range(1, len(ecg_coords)):
        if ecg_coords[i][0] >= current_x:
            x1, y1 = ecg_coords[i - 1]
            x2, y2 = ecg_coords[i]
            if x2 != x1:
                frac = (current_x - x1) / (x2 - x1)
                dot_y = y1 + frac * (y2 - y1)
            break

    # Beat-driven effects
    pulse_intensity = 0.0
    color_shift = 0.0
    color_wave_progress = -1.0
    sync_intensity = 0.0

    for peak_idx, peak_time in enumerate(r_peak_times):
        time_since_peak = t_anim - peak_time

        if 0 <= time_since_peak < 0.4:
            pulse_intensity = max(pulse_intensity, 1 - (time_since_peak / 0.4))

        if peak_idx == 0 and 0 <= time_since_peak < 0.6:
            color_shift = (1 - time_since_peak / 0.6) * 120
        elif peak_idx == 1 and 0 <= time_since_peak < 0.8:
            color_wave_progress = time_since_peak / 0.8
        elif peak_idx == 2 and 0 <= time_since_peak < 0.7:
            sync_intensity = math.sin((time_since_peak / 0.7) * math.pi)

    rot_angle = t_anim * 0.25

    # Base image
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # Title
    draw.text((canvas_size / 2 - 128, 14), "Bioresponsive Particle Environment", fill=(235, 235, 235, 255))

    # Stick figure
    figure_scale = cfg["figure_scale"]
    figure_center_x = cfg["figure_center_x"]
    figure_center_y = cfg["figure_center_y"]
    stick_col = (150, 150, 150, int(255 * 0.42))

    # Head
    draw.ellipse(
        (
            figure_center_x - figure_scale * 0.35,
            figure_center_y - figure_scale * 1.75,
            figure_center_x + figure_scale * 0.35,
            figure_center_y - figure_scale * 1.05,
        ),
        outline=stick_col,
        width=2,
    )

    # Body and limbs
    draw.line(
        (figure_center_x, figure_center_y - figure_scale * 1.0, figure_center_x, figure_center_y + figure_scale * 0.3),
        fill=stick_col,
        width=2,
    )
    draw.line(
        (
            figure_center_x - figure_scale * 0.6,
            figure_center_y - figure_scale * 0.5,
            figure_center_x + figure_scale * 0.6,
            figure_center_y - figure_scale * 0.5,
        ),
        fill=stick_col,
        width=2,
    )
    draw.line(
        (
            figure_center_x,
            figure_center_y + figure_scale * 0.3,
            figure_center_x - figure_scale * 0.4,
            figure_center_y + figure_scale * 1.0,
        ),
        fill=stick_col,
        width=2,
    )
    draw.line(
        (
            figure_center_x,
            figure_center_y + figure_scale * 0.3,
            figure_center_x + figure_scale * 0.4,
            figure_center_y + figure_scale * 1.0,
        ),
        fill=stick_col,
        width=2,
    )

    # Rotate and depth-sort particles
    rotated_particles = []
    for p in particles:
        rotated_pos = rotate_y(p["base_pos"], rot_angle)
        rotated_particles.append({**p, "pos": rotated_pos})

    sorted_particles = sorted(rotated_particles, key=lambda p: p["pos"][2])

    # Stable per-frame jitter randomness
    random.seed(frame * 1000)

    for particle in sorted_particles:
        pos = particle["pos"]
        depth = (pos[2] + 1) / 2

        base_px = sphere_center_x + pos[0] * sphere_radius
        base_py = sphere_center_y + pos[1] * sphere_radius

        if pulse_intensity > 0:
            jitter_x = (random.random() - 0.5) * 22 * pulse_intensity
            jitter_y = (random.random() - 0.5) * 22 * pulse_intensity
        else:
            jitter_x, jitter_y = 0.0, 0.0

        px = base_px + jitter_x
        py = base_py + jitter_y

        size = particle_base_size * (0.5 + 0.55 * depth)

        # Particle hue cycling
        phase_offset = particle["color_phase_offset"]
        color_speed = particle["color_speed"]
        cycling_hue = (t_anim * color_speed * 360 + phase_offset * 180) % 360

        final_hue = (cycling_hue + color_shift) % 360

        # Expanding color wave
        if color_wave_progress >= 0:
            dist = math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)
            wave_hit = abs(dist - color_wave_progress * 1.2)
            if wave_hit < 0.25:
                wave_strength = 1 - wave_hit / 0.25
                final_hue = (final_hue + wave_strength * 180) % 360

        # Synchronization toward shared hue
        if sync_intensity > 0:
            target_hue = (t_anim * 60) % 360
            final_hue = final_hue * (1 - sync_intensity * 0.85) + target_hue * sync_intensity * 0.85

        sat = 85
        light = 34 + 33 * depth
        if pulse_intensity > 0:
            light += pulse_intensity * 14
        light = min(light, 85)

        r, g, b = hsl_to_rgb_int(final_hue, sat, light)

        opacity = max(0.0, min(1.0, 0.55 + 0.4 * depth))
        a_core = int(255 * opacity)
        a_glow = int(255 * opacity * 0.32)

        # Outer glow
        draw_filled_circle(draw, px, py, size * 1.35, (r, g, b, a_glow))
        # Core
        draw_filled_circle(draw, px, py, size, (r, g, b, a_core))
        # Highlight
        hl_size = size * 0.28
        hl_opacity = int(255 * 0.5 * depth * opacity)
        draw_filled_circle(draw, px - size * 0.28, py - size * 0.28, hl_size, (255, 255, 255, hl_opacity))

    # ECG baseline and label
    draw.line((30, 400, 470, 400), fill=(58, 58, 58, 255), width=1)
    draw.text((canvas_size / 2 - 24, 410), "ECG Signal", fill=(110, 110, 110, 255))

    # ECG trace
    if len(ecg_coords) > 1:
        for i in range(1, len(ecg_coords)):
            draw.line((ecg_coords[i - 1][0], ecg_coords[i - 1][1], ecg_coords[i][0], ecg_coords[i][1]), fill=(235, 235, 235, 255), width=2)

    # Sweep overlay
    draw_horizontal_sweep(draw, current_x - 35, 422, 45, 65)

    # Current ECG dot
    draw_filled_circle(draw, current_x, dot_y, 3.5, (255, 255, 255, 255))

    # Time labels
    draw.text((24, 486), "0s", fill=(86, 86, 86, 255))
    draw.text((170, 486), "2s", fill=(86, 86, 86, 255))
    draw.text((317, 486), "4s", fill=(86, 86, 86, 255))
    draw.text((464, 486), "6s", fill=(86, 86, 86, 255), anchor="rs")

    return img


def build_ecg_display(times, values, beat_interval, animation_duration):
    r_peaks_raw = find_r_peaks(times, values)
    if len(r_peaks_raw) < 2:
        raise ValueError(
            "Could not detect at least two R-peaks in ECG data. "
            "Please verify synthetic_ecg_waveform.csv format/content."
        )

    first_r_idx = r_peaks_raw[0][2]
    second_r_idx = r_peaks_raw[1][2]

    start_offset = 10
    cycle_start = max(0, first_r_idx - start_offset)
    cycle_end = max(cycle_start + 2, second_r_idx - start_offset)

    cycle_times = times[cycle_start:cycle_end]
    cycle_values = values[cycle_start:cycle_end]
    original_duration = cycle_times[-1] - cycle_times[0]
    if original_duration <= 0:
        raise ValueError("ECG cycle duration is zero. Check ECG input data.")

    ecg_width = 440
    ecg_height = 50
    ecg_x_offset = 30
    ecg_y_offset = 432

    y_min, y_max = min(values), max(values)
    y_range = max(y_max - y_min, 1e-6)

    y_scale = ecg_height / y_range
    time_scale = beat_interval / original_duration
    x_per_sec = ecg_width / animation_duration

    ecg_coords = []
    for cycle in range(3):
        cycle_offset = cycle * beat_interval
        for i in range(0, len(cycle_times), 2):
            t = (cycle_times[i] - cycle_times[0]) * time_scale + cycle_offset
            x = ecg_x_offset + t * x_per_sec
            y = ecg_y_offset + ecg_height - (cycle_values[i] - y_min) * y_scale
            ecg_coords.append((x, y))

    r_peak_fraction = (start_offset / max(len(cycle_times), 1)) * time_scale
    r_peak_times = [r_peak_fraction + i * beat_interval for i in range(3)]

    return {
        "ecg_coords": ecg_coords,
        "r_peak_times": r_peak_times,
        "ecg_x_offset": ecg_x_offset,
        "ecg_y_offset": ecg_y_offset,
        "ecg_height": ecg_height,
        "x_per_sec": x_per_sec,
        "source_peak_count": len(r_peaks_raw),
    }


def main():
    args = parse_args()

    ecg_csv_path = Path(args.ecg_csv)
    output_gif = Path(args.output)

    print("Bioresponsive Particle Environment GIF Generator")
    print("=" * 50)
    print(f"ECG CSV: {ecg_csv_path}")
    print(f"Output GIF: {output_gif}")

    if not ecg_csv_path.exists():
        raise FileNotFoundError(
            f"ECG CSV not found: {ecg_csv_path}\n"
            "Pass --ecg-csv with a valid path."
        )

    times, values = read_ecg_data(ecg_csv_path)
    ecg_display = build_ecg_display(times, values, args.beat_interval, args.duration)

    print(f"Detected {ecg_display['source_peak_count']} R peaks in source data")
    print(f"Animation R peaks: {[f'{t:.2f}s' for t in ecg_display['r_peak_times']]}")

    print(f"Generating icosphere (subdivisions={args.subdivisions})...")
    vertices = generate_icosphere(args.subdivisions)
    print(f"Vertices/particles: {len(vertices)}")

    random.seed(args.seed)
    particles = []
    for i, v in enumerate(vertices):
        particles.append(
            {
                "idx": i,
                "base_pos": v,
                "color_phase_offset": random.random() * 2 * math.pi,
                "color_speed": 0.3 + random.random() * 0.2,
            }
        )

    cfg = {
        "fps": args.fps,
        "canvas_size": args.canvas_size,
        "sphere_center_x": args.canvas_size / 2,
        "sphere_center_y": 190,
        "sphere_radius": 115,
        "particle_base_size": args.particle_size,
        "figure_scale": 35,
        "figure_center_x": args.canvas_size / 2,
        "figure_center_y": 195,
        "ecg_x_offset": ecg_display["ecg_x_offset"],
        "ecg_y_offset": ecg_display["ecg_y_offset"],
        "ecg_height": ecg_display["ecg_height"],
        "x_per_sec": ecg_display["x_per_sec"],
    }

    num_frames = int(round(args.fps * args.duration))
    print(f"Rendering {num_frames} frames at {args.fps} fps...")

    frames = []
    for frame in range(num_frames):
        img = generate_frame_image(
            frame=frame,
            particles=particles,
            ecg_coords=ecg_display["ecg_coords"],
            r_peak_times=ecg_display["r_peak_times"],
            cfg=cfg,
        )
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=256))

        if frame % max(1, args.fps) == 0:
            print(f"  Frame {frame}/{num_frames}")

    output_gif.parent.mkdir(parents=True, exist_ok=True)

    print("Saving GIF...")
    frame_duration_ms = int(round(1000 / args.fps))
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    print(f"GIF created: {output_gif}")
    print("Done.")


if __name__ == "__main__":
    main()
