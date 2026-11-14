#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer StaticSpherePositions {
    vec4 rows[];
} sphere_positions;

layout(set = 0, binding = 1) readonly buffer StaticSphereNormals {
    vec4 rows[];
} sphere_normals;

layout(set = 0, binding = 2) buffer PrivateParticleRows {
    vec4 rows[];
} particle_output;

layout(set = 0, binding = 3) readonly buffer PhaseSourceRows {
    vec4 rows[];
} phase_source;

layout(set = 0, binding = 4) buffer PhaseTargetRows {
    vec4 rows[];
} phase_target;

layout(set = 0, binding = 5) readonly buffer NeighborAux0Rows {
    uvec4 rows[];
} neighbor_aux0;

layout(set = 0, binding = 8) readonly buffer PrivateParticleDriverRows {
    vec4 rows[];
} particle_drivers;

layout(push_constant) uniform PrivateParticlePush {
    vec4 params0;
    vec4 params1;
    vec4 transparency_params;
    vec4 tracer_params;
    vec4 world_center_scale;
    vec4 eye_position;
    vec4 eye_orientation_xyzw;
    vec4 fov_tangents;
} pc;

// Existing products retain the 26-row driver ABI and never read the extension.
// Only an explicitly selected product shader/buffer pair enables these rows.
#ifndef PRIVATE_PARTICLE_FRAME_EXTENSION_V1
#define PRIVATE_PARTICLE_FRAME_EXTENSION_V1 0
#endif

float showcase_radius_weight01(float p) {
    float centered = 2.0 * clamp(p, 0.0, 1.0) - 1.0;
    return centered * centered;
}

float private_coupling_gain() {
#if PRIVATE_PARTICLE_FRAME_EXTENSION_V1
    vec4 meta = particle_drivers.rows[26];
    // Exact tagged, versioned opt-in. Unknown modes safely bypass modulation.
    if (meta.x != 7601.0 || meta.y != 1.0 || meta.z != 1.0 || meta.w != 1.0) {
        return 1.0;
    }
    vec4 controls = particle_drivers.rows[27];
    if (any(isnan(controls)) || any(isinf(controls))) {
        return 1.0;
    }
    float endpoint_gain = clamp(controls.y, 0.0, 8.0);
    float midpoint_gain = clamp(controls.z, 0.0, 8.0);
    return mix(midpoint_gain, endpoint_gain, showcase_radius_weight01(controls.x));
#else
    return 1.0;
#endif
}

#ifndef PRIVATE_PARTICLE_ANCHOR_ECHO_MAX_COUNT
#define PRIVATE_PARTICLE_ANCHOR_ECHO_MAX_COUNT 0
#endif
#ifndef PRIVATE_PARTICLE_ANCHOR_ECHO_DRAW_ECHO_COUNT
#define PRIVATE_PARTICLE_ANCHOR_ECHO_DRAW_ECHO_COUNT 0
#endif
#ifndef PRIVATE_PARTICLE_ANCHOR_ECHO_LIFETIME_SECONDS
#define PRIVATE_PARTICLE_ANCHOR_ECHO_LIFETIME_SECONDS 0.05
#endif
#ifndef PRIVATE_PARTICLE_ANCHOR_ECHO_COPIES_PER_SECOND
#define PRIVATE_PARTICLE_ANCHOR_ECHO_COPIES_PER_SECOND 20.0
#endif
#ifndef PRIVATE_PARTICLE_ANCHOR_ECHO_RADIUS_M
#define PRIVATE_PARTICLE_ANCHOR_ECHO_RADIUS_M 0.08
#endif
#ifndef PRIVATE_PARTICLE_ANCHOR_ECHO_ALPHA
#define PRIVATE_PARTICLE_ANCHOR_ECHO_ALPHA 0.09803922
#endif
#ifndef PRIVATE_PARTICLE_ANCHOR_ECHO_ROTATION_RADIANS
#define PRIVATE_PARTICLE_ANCHOR_ECHO_ROTATION_RADIANS 3.14159265358979323846
#endif

const float TAU = 6.28318530717958647692;
const float TRACER_FADE_IN_FRACTION = 0.10;
const uint AKD_AUX0_MAGIC = 0x32444b41u;
const uint AKD_AUX0_LAYOUT_VERSION = 9u;
const uint AKD_PHASE_DIMENSIONS = 6u;
const uint AKD_VISUAL_DRIVER_POLICY_OSCILLATOR = 0u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER0 = 1u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER1 = 2u;
const uint AKD_VISUAL_DRIVER_POLICY_FUTURE_SLOT_OSCILLATOR_FALLBACK = 3u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER2 = 4u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER3 = 5u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER4 = 6u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER5 = 7u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER6 = 8u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER7 = 9u;
const uint AKD_VISUAL_DRIVER_POLICY_DRIVER_BANK_SLOT_COUNT = 8u;
const uint AKD_VISUAL_DRIVER_POLICY_NO_DRIVER = 0xffffffffu;
const uint PANEL_DRIVER_BANK_VALUE_VEC4_ROWS = 2u;
const uint PANEL_DRIVER_CONTROL_ROWS_PER_SLOT = 3u;
const uint PANEL_DRIVER_MODE_OSCILLATOR = 0u;
const uint PANEL_DRIVER_MODE_MANUAL = 1u;
const uint PANEL_DRIVER_MODE_INPUT_SLOT = 2u;
const uint PANEL_DRIVER_MODE_DIRECT = 3u;
const uint PANEL_DRIVER_MODE_PRIVATE_HEARTBEAT_ORBIT = 4u;
const uint PANEL_CURVE_LINEAR = 0u;
const uint PANEL_CURVE_AKD_HUMP = 1u;
const uint PANEL_CURVE_SMOOTHSTEP = 2u;
const uint PANEL_CURVE_REVERSE_LINEAR = 3u;
const uint PANEL_CURVE_HOLD_LOW = 4u;
const uint PANEL_CURVE_HOLD_HIGH = 5u;
const uint PANEL_CONTROL_NO_SLOT = 0xffffffffu;
const uint PANEL_PARTICLE_SIZE_MODE_SPHERE_PERCENT = 1u;
const uint PANEL_PARTICLE_SIZE_MODE_WORLD_METERS = 2u;
const uint AKD_TRACER_STATE_ROWS_PER_SLOT = 4u;
const uint AKD_ANCHOR_ECHO_STATE_ROWS_PER_SLOT = 4u;

vec4 safe_normalize_quat(vec4 quat) {
    float length_sq = max(dot(quat, quat), 0.000000000001);
    return quat * inversesqrt(length_sq);
}

vec3 rotate_by_quat(vec4 quat, vec3 vector) {
    vec4 q = safe_normalize_quat(quat);
    vec3 uv = cross(q.xyz, vector);
    vec3 uuv = cross(q.xyz, uv);
    return vector + uv * (2.0 * q.w) + uuv * 2.0;
}

float hash01(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return float(value & 0x00ffffffu) / float(0x01000000u);
}

float wrap_tau(float phase) {
    return phase - floor(phase / TAU) * TAU;
}

float phase01(float phase) {
    return fract(phase / TAU);
}

float triangle01(float value01) {
    float x = fract(value01);
    return 1.0 - abs(x * 2.0 - 1.0);
}

float cubic_hermite(float p0, float m0, float p1, float m1, float t, float dt) {
    float t2 = t * t;
    float t3 = t2 * t;
    float h00 = (2.0 * t3) - (3.0 * t2) + 1.0;
    float h10 = t3 - (2.0 * t2) + t;
    float h01 = (-2.0 * t3) + (3.0 * t2);
    float h11 = t3 - t2;
    return (h00 * p0) + (h10 * m0 * dt) + (h01 * p1) + (h11 * m1 * dt);
}

float akd_hump_envelope(float value01) {
    float t = clamp(value01, 0.0, 1.0);
    float value = t <= 0.5
        ? cubic_hermite(0.0, 2.0, 1.0, 0.0, t / 0.5, 0.5)
        : cubic_hermite(1.0, 0.0, 0.0, -2.0, (t - 0.5) / 0.5, 0.5);
    return clamp(value, 0.0, 1.0);
}

float resolve_cycle_phase01(float base_phase01, uint cycle_multiplier) {
    float phase = clamp(base_phase01, 0.0, 1.0);
    if (cycle_multiplier == 0u) {
        return 0.0;
    }
    if (cycle_multiplier == 1u) {
        return phase;
    }
    return fract(phase * float(cycle_multiplier));
}

float to_half_open_phase01(float value01) {
    return min(clamp(value01, 0.0, 1.0), 0.99902344);
}

uint component_at(uvec4 row, uint component) {
    if (component == 0u) { return row.x; }
    if (component == 1u) { return row.y; }
    if (component == 2u) { return row.z; }
    return row.w;
}

float component_at(vec4 row, uint component) {
    if (component == 0u) { return row.x; }
    if (component == 1u) { return row.y; }
    if (component == 2u) { return row.z; }
    return row.w;
}

float generic_driver_value01(uint index, float fallback) {
    if (index >= 8u) {
        return clamp(fallback, 0.0, 1.0);
    }
    vec4 row = particle_drivers.rows[index >> 2u];
    return clamp(component_at(row, index & 3u), 0.0, 1.0);
}

float header_f32_or(bool valid, uint row_index, uint component, float fallback);

uint panel_u32(float value, uint max_value) {
    return uint(clamp(floor(value + 0.5), 0.0, float(max_value)));
}

uint panel_control_slot_for_curve_index(uint curve_index) {
    if (curve_index == 1u) { return 2u; }
    if (curve_index == 2u) { return 3u; }
    if (curve_index == 6u) { return 4u; }
    if (curve_index == 7u) { return 5u; }
    if (curve_index == 8u) { return 6u; }
    if (curve_index == 9u) { return 7u; }
    return PANEL_CONTROL_NO_SLOT;
}

vec4 panel_control_meta(uint slot) {
    uint row = PANEL_DRIVER_BANK_VALUE_VEC4_ROWS + slot * PANEL_DRIVER_CONTROL_ROWS_PER_SLOT;
    return particle_drivers.rows[row];
}

vec4 panel_control_value_row(uint slot) {
    uint row = PANEL_DRIVER_BANK_VALUE_VEC4_ROWS
        + slot * PANEL_DRIVER_CONTROL_ROWS_PER_SLOT
        + 1u;
    return particle_drivers.rows[row];
}

vec4 panel_control_aux_row(uint slot) {
    uint row = PANEL_DRIVER_BANK_VALUE_VEC4_ROWS
        + slot * PANEL_DRIVER_CONTROL_ROWS_PER_SLOT
        + 2u;
    return particle_drivers.rows[row];
}

// ORBIT_CONTROL_NUMERIC_BEGIN
// Negative, disjoint two-unit intervals carry source0..3 and a [0,1]
// envelope in the existing phase_hi.z word. Nonnegative values are legacy.
// This preserves phase_hi.xy (dimensions4/5) and phase_hi.w (rotation).
float orbit_control_pack_state(uint source, float envelope01) {
    return -(2.0 * float(source + 1u) + clamp(envelope01, 0.0, 1.0));
}

float orbit_control_envelope(float state, uint source) {
    float tag = 2.0 * float(source + 1u);
    float tagged_value = -state;
    return tagged_value >= tag && tagged_value <= tag + 1.0
        ? clamp(tagged_value - tag, 0.0, 1.0) : 0.0;
}

float orbit_control_next_state(
    uint source, float previous_state, bool reset_state,
    float event01, float dt) {
    float previous01 = reset_state ? 0.0 : orbit_control_envelope(previous_state, source);
    float envelope01 = source >= 2u
        ? (event01 >= 0.5 ? 1.0 : previous01 * exp(-dt / 0.25))
        : 0.0;
    return orbit_control_pack_state(source, envelope01);
}

float orbit_control_weight(uint source, float oscillator_weight01, float envelope01) {
    if (source == 0u) { return 0.0; }
    return source == 1u ? oscillator_weight01 : envelope01;
}

float orbit_control_radius(
    bool enabled, uint source, bool scale_with_particle_size, float reference_radius_m,
    float radius_m, float weight01, float minimum, float maximum,
    bool legacy_cycle_disabled) {
    float orbit_radius_multiplier = mix(minimum, maximum, weight01);
    if ((!enabled || source == 1u) && legacy_cycle_disabled) {
        orbit_radius_multiplier = 0.0;
    }
    float orbit_reference_m = enabled && !scale_with_particle_size
        ? reference_radius_m : radius_m;
    return orbit_radius_multiplier * orbit_reference_m;
}
// ORBIT_CONTROL_NUMERIC_END

#if PRIVATE_PARTICLE_FRAME_EXTENSION_V1
float showcase_heartbeat_pulse(float previous01, float event01, float dt) {
    return event01 >= 0.5 ? 1.0 : 0.1 + max(previous01 - 0.1, 0.0) * exp(-dt / 0.25);
}

float showcase_heartbeat_orbit_weight(float pulse01) {
    // Unweighted Hermite keys at 0, 0.5 and 1 have slopes 2, 0 and -2.
    float x = 2.0 * min(pulse01, 1.0 - pulse01);
    return x + x * x * (1.0 - x);
}
#endif

uint panel_control_mode(uint slot) {
    return panel_u32(panel_control_meta(slot).x, PANEL_DRIVER_MODE_PRIVATE_HEARTBEAT_ORBIT);
}

bool panel_control_active(uint curve_index) {
    uint slot = panel_control_slot_for_curve_index(curve_index);
    if (slot == PANEL_CONTROL_NO_SLOT) {
        return false;
    }
    return panel_control_mode(slot) != PANEL_DRIVER_MODE_DIRECT;
}

float panel_curve_value01(uint curve_code, float t01) {
    float t = clamp(t01, 0.0, 1.0);
    if (curve_code == PANEL_CURVE_AKD_HUMP) {
        return akd_hump_envelope(t);
    }
    if (curve_code == PANEL_CURVE_SMOOTHSTEP) {
        return t * t * (3.0 - 2.0 * t);
    }
    if (curve_code == PANEL_CURVE_REVERSE_LINEAR) {
        return 1.0 - t;
    }
    if (curve_code == PANEL_CURVE_HOLD_LOW) {
        return 0.0;
    }
    if (curve_code == PANEL_CURVE_HOLD_HIGH) {
        return 1.0;
    }
    return t;
}

float panel_range_endpoint_or(
    bool valid_layout,
    uint curve_index,
    uint row_index,
    uint component,
    float fallback,
    bool max_endpoint) {
    uint slot = panel_control_slot_for_curve_index(curve_index);
    if (slot != PANEL_CONTROL_NO_SLOT && panel_control_mode(slot) != PANEL_DRIVER_MODE_DIRECT) {
        vec4 value_row = panel_control_value_row(slot);
        return max_endpoint ? value_row.y : value_row.x;
    }
    return header_f32_or(valid_layout, row_index, component, fallback);
}

uint panel_cycle_or(uint curve_index, uint fallback_cycle) {
    uint slot = panel_control_slot_for_curve_index(curve_index);
    if (slot == PANEL_CONTROL_NO_SLOT || panel_control_mode(slot) == PANEL_DRIVER_MODE_DIRECT) {
        return fallback_cycle;
    }
    return panel_u32(panel_control_meta(slot).w, 10u);
}

uint packed_u32(uint row_start, uint element_index) {
    uvec4 row = neighbor_aux0.rows[row_start + (element_index >> 2u)];
    return component_at(row, element_index & 3u);
}

float packed_f32(uint row_start, uint element_index) {
    return uintBitsToFloat(packed_u32(row_start, element_index));
}

float header_f32(uint row_index, uint component) {
    return uintBitsToFloat(component_at(neighbor_aux0.rows[row_index], component));
}

uint header_u32_or(bool valid, uint row_index, uint component, uint fallback) {
    return valid ? component_at(neighbor_aux0.rows[row_index], component) : fallback;
}

float header_f32_or(bool valid, uint row_index, uint component, float fallback) {
    return valid ? header_f32(row_index, component) : fallback;
}

uint visual_row_start() {
    return neighbor_aux0.rows[2].w;
}

uint visual_u32(uint row_offset, uint component) {
    return component_at(neighbor_aux0.rows[visual_row_start() + row_offset], component);
}

float visual_f32(uint row_offset, uint component) {
    return uintBitsToFloat(visual_u32(row_offset, component));
}

bool aux0_layout_valid(uint particle_count) {
    uvec4 header = neighbor_aux0.rows[0];
    return header.x == AKD_AUX0_MAGIC
        && header.y == AKD_AUX0_LAYOUT_VERSION
        && header.z == particle_count
        && header.w == AKD_PHASE_DIMENSIONS;
}

float phase_component(vec4 lo, vec4 hi, uint dim);

vec3 normalized_or(vec3 value, vec3 fallback) {
    float len_sq = dot(value, value);
    return len_sq > 0.00000001 ? value * inversesqrt(len_sq) : fallback;
}

vec3 private_world_anchor_center() {
    return pc.world_center_scale.xyz;
}

void private_sim_to_world_basis(out vec3 right, out vec3 up, out vec3 forward) {
    right = normalized_or(pc.eye_position.xyz, vec3(1.0, 0.0, 0.0));
    up = normalized_or(pc.eye_orientation_xyzw.xyz, vec3(0.0, 1.0, 0.0));
    forward = normalized_or(pc.fov_tangents.xyz, normalized_or(cross(up, right), vec3(0.0, 0.0, -1.0)));
}

vec3 private_sim_offset_to_world(vec3 sim_offset, vec3 right, vec3 up, vec3 forward) {
    return right * sim_offset.x + up * sim_offset.y - forward * sim_offset.z;
}

float sample_deformation_curve01(bool valid_layout, uint curve_index, float t01) {
    if (!valid_layout) {
        return clamp(t01, 0.0, 1.0);
    }

    uint sample_count = header_u32_or(valid_layout, 21u, 2u, 0u);
    uint row_start = header_u32_or(valid_layout, 21u, 3u, 0u);
    if (sample_count < 2u || row_start == 0u) {
        return clamp(t01, 0.0, 1.0);
    }

    float sample_position = clamp(t01, 0.0, 1.0) * float(sample_count - 1u);
    uint lo = uint(floor(sample_position));
    uint hi = min(lo + 1u, sample_count - 1u);
    float f = sample_position - float(lo);
    uint curve_base = curve_index * sample_count;
    return mix(
        packed_f32(row_start, curve_base + lo),
        packed_f32(row_start, curve_base + hi),
        f);
}

float sample_visual_curve01(bool valid_layout, uint curve_index, float t01, float fallback01) {
    uint panel_slot = panel_control_slot_for_curve_index(curve_index);
    if (panel_slot != PANEL_CONTROL_NO_SLOT
        && panel_control_mode(panel_slot) != PANEL_DRIVER_MODE_DIRECT) {
        uint curve_code = panel_u32(panel_control_meta(panel_slot).z, PANEL_CURVE_HOLD_HIGH);
        return panel_curve_value01(curve_code, t01);
    }
    if (!valid_layout) {
        return clamp(fallback01, 0.0, 1.0);
    }

    uint sample_count = header_u32_or(valid_layout, 22u, 0u, 0u);
    uint curve_count = header_u32_or(valid_layout, 22u, 1u, 0u);
    uint row_start = header_u32_or(valid_layout, 22u, 2u, 0u);
    if (sample_count < 2u || row_start == 0u || curve_index >= curve_count) {
        return clamp(fallback01, 0.0, 1.0);
    }

    float sample_position = clamp(t01, 0.0, 1.0) * float(sample_count - 1u);
    uint lo = uint(floor(sample_position));
    uint hi = min(lo + 1u, sample_count - 1u);
    float f = sample_position - float(lo);
    uint curve_base = curve_index * sample_count;
    return mix(
        packed_f32(row_start, curve_base + lo),
        packed_f32(row_start, curve_base + hi),
        f);
}

float sample_coupling_curve01(bool valid_layout, uint curve_index, float t01, float fallback01) {
    if (!valid_layout) {
        return clamp(fallback01, 0.0, 1.0);
    }

    uint sample_count = header_u32_or(valid_layout, 23u, 0u, 0u);
    uint curve_count = header_u32_or(valid_layout, 23u, 1u, 0u);
    uint row_start = header_u32_or(valid_layout, 23u, 2u, 0u);
    if (sample_count < 2u || row_start == 0u || curve_index >= curve_count) {
        return clamp(fallback01, 0.0, 1.0);
    }

    float sample_position = clamp(t01, 0.0, 1.0) * float(sample_count - 1u);
    uint lo = uint(floor(sample_position));
    uint hi = min(lo + 1u, sample_count - 1u);
    float f = sample_position - float(lo);
    uint curve_base = curve_index * sample_count;
    return mix(
        packed_f32(row_start, curve_base + lo),
        packed_f32(row_start, curve_base + hi),
        f);
}

uint tuner_coupling_start() {
#if PRIVATE_PARTICLE_FRAME_EXTENSION_V1
    uint rows = uint(neighbor_aux0.rows.length());
    if (rows > 91u) {
        uint start = rows - 91u;
        uvec4 footer = neighbor_aux0.rows[rows - 1u];
        if (all(equal(footer, uvec4(0x31434e54u, 1u, start, 5u)))) {
            return start;
        }
    }
#endif
    return 0u;
}

#if PRIVATE_PARTICLE_FRAME_EXTENSION_V1
float tuner_coupling_weight(uint row, float showcase_driver01) {
    uvec4 meta = neighbor_aux0.rows[row];
    vec4 controls = uintBitsToFloat(neighbor_aux0.rows[row + 1u]);
    float radius = generic_driver_value01(0u, clamp(pc.params0.w, 0.0, 1.0));
    float beat = generic_driver_value01(5u, 0.0);
    float phase = fract(max(pc.params1.z, 0.0) * 0.1);
    float input01 = showcase_driver01;
    switch (meta.x) {
        case 1u: input01 = controls.x; break;
        case 2u: input01 = meta.z == 1u ? 1.0 - abs(2.0 * radius - 1.0) : radius; break;
        case 3u: input01 = phase; break;
        case 4u: input01 = beat; break;
        case 5u: {
            float total = controls.y + controls.z + controls.w;
            input01 = total > 0.0 ? dot(controls.yzw, vec3(radius, beat, phase)) / total : 0.0;
            break;
        }
    }
    float x = clamp(input01, 0.0, 1.0);
    uint key_count = clamp(meta.y, 2u, 16u);
    vec4 a = uintBitsToFloat(neighbor_aux0.rows[row + 2u]);
    if (x <= a.x) return a.y;
    for (uint i = 1u; i < key_count; ++i) {
        vec4 b = uintBitsToFloat(neighbor_aux0.rows[row + 2u + i]);
        if (x <= b.x && b.x > a.x) {
            float width = b.x - a.x;
            float t = (x - a.x) / width;
            float t2 = t * t;
            float t3 = t2 * t;
            return clamp((2.0 * t3 - 3.0 * t2 + 1.0) * a.y
                + (t3 - 2.0 * t2 + t) * width * a.w
                + (-2.0 * t3 + 3.0 * t2) * b.y
                + (t3 - t2) * width * b.z, 0.0, 1.0);
        }
        a = b;
    }
    return a.y;
}
#endif

vec3 compute_deformed_surface_offset(
    vec3 normal,
    float base_radius,
    vec3 deformation_axis,
    float radius_progress01,
    bool valid_layout) {
    if (!valid_layout || header_f32(20u, 0u) < 0.5) {
        return normal * base_radius;
    }

    vec3 axis_dir = normalized_or(deformation_axis, vec3(0.0, 1.0, 0.0));
    float u = dot(normal, axis_dir);
    float sin2 = clamp(1.0 - u * u, 0.0, 1.0);
    float sin_theta = sqrt(sin2);
    if (sin_theta < 0.00001) {
        return normal * base_radius;
    }

    vec3 radial_dir = normalized_or(normal - u * axis_dir, normal);
    float axis_offset = base_radius * u;
    float radial_base = base_radius * sin_theta;

    // AKD stores normalized curve shape separately from endpoint values. The
    // current Sussex payload samples linear curves, but the shader consumes the
    // payload rows so later config-defined curve shape stays data-owned.
    float oblate_curve01 = sample_deformation_curve01(valid_layout, 0u, radius_progress01);
    float oblate_scale = mix(header_f32(20u, 2u), header_f32(20u, 3u), oblate_curve01);
    float axis01 = abs(u);
    float baseline = 1.0 - clamp(axis01, 0.0, 1.0);
    float axis_curve01 = sample_deformation_curve01(valid_layout, 1u, axis01);
    float raw_axis_profile = mix(header_f32(21u, 0u), header_f32(21u, 1u), axis_curve01);
    float axis_profile_scale = max(0.0, 1.0 + (raw_axis_profile - baseline));
    float radial_offset = radial_base * max(0.0, oblate_scale * axis_profile_scale);
    return axis_dir * axis_offset + radial_dir * radial_offset;
}

float visual_phase01(vec4 phase_lo, vec4 phase_hi, uint dim, uint cycle_multiplier) {
    return resolve_cycle_phase01(phase01(phase_component(phase_lo, phase_hi, dim)), cycle_multiplier);
}

uint visual_driver_policy_or(bool valid_layout, uint curve_index, uint fallback_policy) {
    if (!valid_layout) {
        return fallback_policy;
    }

    uint policy_count = header_u32_or(valid_layout, 25u, 0u, 0u);
    uint row_start = header_u32_or(valid_layout, 25u, 1u, 0u);
    if (row_start == 0u || curve_index >= policy_count) {
        return fallback_policy;
    }
    return packed_u32(row_start, curve_index);
}

uint visual_driver_policy_driver_index(uint policy) {
    if (policy == AKD_VISUAL_DRIVER_POLICY_DRIVER0) { return 0u; }
    if (policy == AKD_VISUAL_DRIVER_POLICY_DRIVER1) { return 1u; }
    if (policy >= AKD_VISUAL_DRIVER_POLICY_DRIVER2
        && policy <= AKD_VISUAL_DRIVER_POLICY_DRIVER7) {
        return 2u + (policy - AKD_VISUAL_DRIVER_POLICY_DRIVER2);
    }
    return AKD_VISUAL_DRIVER_POLICY_NO_DRIVER;
}

float visual_driver_phase01(
    bool valid_layout,
    uint curve_index,
    vec4 phase_lo,
    vec4 phase_hi,
    uint dim,
    uint cycle_multiplier) {
    float raw_oscillator_phase01 = phase01(phase_component(phase_lo, phase_hi, dim));
    uint panel_slot = panel_control_slot_for_curve_index(curve_index);
    if (panel_slot != PANEL_CONTROL_NO_SLOT) {
        vec4 meta = panel_control_meta(panel_slot);
        uint mode = panel_u32(meta.x, PANEL_DRIVER_MODE_PRIVATE_HEARTBEAT_ORBIT);
        if (mode != PANEL_DRIVER_MODE_DIRECT) {
            uint cycle = panel_u32(meta.w, 10u);
            float source01 = raw_oscillator_phase01;
            if (mode == PANEL_DRIVER_MODE_MANUAL) {
                source01 = panel_control_value_row(panel_slot).z;
            } else if (mode == PANEL_DRIVER_MODE_INPUT_SLOT) {
                uint source_slot = panel_u32(meta.y, AKD_VISUAL_DRIVER_POLICY_DRIVER_BANK_SLOT_COUNT - 1u);
                source01 = generic_driver_value01(source_slot, raw_oscillator_phase01);
            } else if (mode == PANEL_DRIVER_MODE_PRIVATE_HEARTBEAT_ORBIT
                && curve_index == 7u) {
                // The combined same-APK panel can select the existing private
                // RR envelope for orbit radius. Its baseline, peak, decay, and
                // range remain payload/shader-owned; the panel supplies no
                // timing or amplitude knobs and the default direct mode is
                // unchanged.
                source01 = clamp(phase_hi.z, 0.0, 1.0);
            }
            return resolve_cycle_phase01(source01, cycle);
        }
    }

    float oscillator_phase01 = resolve_cycle_phase01(raw_oscillator_phase01, cycle_multiplier);
    uint policy = visual_driver_policy_or(
        valid_layout,
        curve_index,
        AKD_VISUAL_DRIVER_POLICY_OSCILLATOR);
    uint driver_index = visual_driver_policy_driver_index(policy);
    if (driver_index < AKD_VISUAL_DRIVER_POLICY_DRIVER_BANK_SLOT_COUNT) {
        // The isolated heartbeat payload opts into a private temporal envelope.
        // Rusty Quest supplies only a generic, one-frame bounded event in slot 5;
        // the baseline, peak, decay, and orbit-radius interpretation stay here.
        if (curve_index == 7u
            && driver_index == 5u
            && header_u32_or(valid_layout, 22u, 3u, 0u) == 1u) {
            return resolve_cycle_phase01(clamp(phase_hi.z, 0.0, 1.0), cycle_multiplier);
        }
        float driver_value01 = generic_driver_value01(driver_index, oscillator_phase01);
        return resolve_cycle_phase01(driver_value01, cycle_multiplier);
    }
    // Active Sussex routes orbit radius from heartbeat. The public substrate
    // exposes generic normalized driver scalars, but this private payload keeps
    // that AKD assignment unbound until the default, range, and later signal
    // driver behavior are verified.
    return oscillator_phase01;
}

float initial_phase_component(uint particle, uint dim) {
    uint seed = (particle * AKD_PHASE_DIMENSIONS + dim) * 747796405u + 2891336453u;
    return hash01(seed) * TAU;
}

vec4 initial_phase_lo(uint index) {
    return vec4(
        initial_phase_component(index, 0u),
        initial_phase_component(index, 1u),
        initial_phase_component(index, 2u),
        initial_phase_component(index, 3u)
    );
}

vec4 initial_phase_hi(uint index) {
    return vec4(
        initial_phase_component(index, 4u),
        initial_phase_component(index, 5u),
        0.0,
        0.0
    );
}

float phase_component(vec4 lo, vec4 hi, uint dim) {
    if (dim == 0u) { return lo.x; }
    if (dim == 1u) { return lo.y; }
    if (dim == 2u) { return lo.z; }
    if (dim == 3u) { return lo.w; }
    if (dim == 4u) { return hi.x; }
    return hi.y;
}

void set_phase_component(inout vec4 lo, inout vec2 hi, uint dim, float value) {
    if (dim == 0u) { lo.x = value; return; }
    if (dim == 1u) { lo.y = value; return; }
    if (dim == 2u) { lo.z = value; return; }
    if (dim == 3u) { lo.w = value; return; }
    if (dim == 4u) { hi.x = value; return; }
    hi.y = value;
}

void read_phase_pair(uint particle, bool reset_state, out vec4 lo, out vec4 hi) {
    if (reset_state) {
        lo = initial_phase_lo(particle);
        hi = initial_phase_hi(particle);
        return;
    }
    uint base = particle * 2u;
    lo = phase_source.rows[base];
    hi = phase_source.rows[base + 1u];
}

float natural_frequency_hz(uint particle, uint dim) {
    uint natural_start = neighbor_aux0.rows[2].y;
    uint row = natural_start + particle * 2u + (dim >= 4u ? 1u : 0u);
    uint component = dim >= 4u ? dim - 4u : dim;
    return uintBitsToFloat(component_at(neighbor_aux0.rows[row], component));
}

bool is_static_phase_dimension(bool valid_layout, uint dim) {
    uint static_mask = header_u32_or(valid_layout, 23u, 3u, 0u);
    return dim < 32u && (static_mask & (1u << dim)) != 0u;
}

float cross_coupling_weight(bool valid_layout, uint target_dim, uint source_dim) {
    uint cross_start = neighbor_aux0.rows[2].z;
    uint element = target_dim * AKD_PHASE_DIMENSIONS + source_dim;
    return packed_f32(cross_start, element);
}

void accumulate_tier(
    uint dim,
    float phase,
    uint count,
    uint offset,
    uint index_row_start,
    float kernel,
    bool reset_state,
    inout float sum,
    inout uint valid) {
    if (kernel == 0.0 || count == 0u) {
        return;
    }

    for (uint slot = 0u; slot < count; ++slot) {
        uint neighbor = packed_u32(index_row_start, offset + slot);
        vec4 neighbor_lo;
        vec4 neighbor_hi;
        read_phase_pair(neighbor, reset_state, neighbor_lo, neighbor_hi);
        float dphi = phase_component(neighbor_lo, neighbor_hi, dim) - phase;
        sum += sin(dphi) * kernel;
        valid += 1u;
    }
}

struct CouplingFrame {
    vec4 kernels;
    float cross_strength;
    float nat_freq_multiplier;
    float gain;
};

CouplingFrame resolve_coupling_frame(bool valid_layout, float driver01) {
    float base_coupling_strength = header_f32(6u, 0u);
    float cross_coupling_strength = header_f32(6u, 1u);
    float nat_freq_curve01 = sample_coupling_curve01(valid_layout, 0u, driver01, 1.0);
    float tier1_curve01 = sample_coupling_curve01(valid_layout, 1u, driver01, driver01);
    float tier2_curve01 = sample_coupling_curve01(valid_layout, 2u, driver01, driver01);
    float tier3_curve01 = sample_coupling_curve01(valid_layout, 3u, driver01, driver01);
    float small_curve01 = sample_coupling_curve01(valid_layout, 4u, driver01, driver01);
    uint tuner_row = tuner_coupling_start();
#if PRIVATE_PARTICLE_FRAME_EXTENSION_V1
    if (tuner_row != 0u) {
        tier1_curve01 = tuner_coupling_weight(tuner_row, driver01);
        tier2_curve01 = tuner_coupling_weight(tuner_row + 18u, driver01);
        tier3_curve01 = tuner_coupling_weight(tuner_row + 36u, driver01);
        small_curve01 = tuner_coupling_weight(tuner_row + 54u, driver01);
        nat_freq_curve01 = tuner_coupling_weight(tuner_row + 72u, driver01);
    }
#endif
    float nat_freq_multiplier = mix(header_f32(24u, 0u), header_f32(24u, 1u), nat_freq_curve01);
    float tier1_kernel = mix(header_f32(6u, 2u), header_f32(6u, 3u), tier1_curve01) * base_coupling_strength;
    float tier2_kernel = mix(header_f32(7u, 0u), header_f32(7u, 1u), tier2_curve01) * base_coupling_strength;
    float tier3_kernel = mix(header_f32(7u, 2u), header_f32(7u, 3u), tier3_curve01) * base_coupling_strength;
    float small_kernel = header_f32(8u, 2u) > 0.5
        ? mix(header_f32(8u, 0u), header_f32(8u, 1u), small_curve01) * base_coupling_strength
        : 0.0;
    return CouplingFrame(
        vec4(tier1_kernel, tier2_kernel, tier3_kernel, small_kernel),
        cross_coupling_strength, nat_freq_multiplier,
        tuner_row != 0u ? 1.0 : private_coupling_gain());
}

float integrate_dimension(
    uint particle,
    uint dim,
    vec4 phase_lo,
    vec4 phase_hi,
    uvec4 tier_meta0,
    uvec4 tier_meta1,
    bool reset_state,
    float dt,
    bool valid_layout,
    CouplingFrame coupling) {
    float phase = phase_component(phase_lo, phase_hi, dim);

    uint tier1_start = neighbor_aux0.rows[1].y;
    uint tier2_start = neighbor_aux0.rows[1].z;
    uint tier3_start = neighbor_aux0.rows[1].w;
    uint small_start = neighbor_aux0.rows[2].x;

    float neighbor_sum = 0.0;
    uint valid = 0u;
    accumulate_tier(dim, phase, tier_meta0.x, tier_meta0.y, tier1_start, coupling.kernels.x, reset_state, neighbor_sum, valid);
    accumulate_tier(dim, phase, tier_meta0.z, tier_meta0.w, tier2_start, coupling.kernels.y, reset_state, neighbor_sum, valid);
    accumulate_tier(dim, phase, tier_meta1.x, tier_meta1.y, tier3_start, coupling.kernels.z, reset_state, neighbor_sum, valid);
    accumulate_tier(dim, phase, tier_meta1.z, tier_meta1.w, small_start, coupling.kernels.w, reset_state, neighbor_sum, valid);

    float coupling_gain = coupling.gain;
    float neighbor_coupling = valid > 0u ? neighbor_sum / float(valid) : 0.0;
    neighbor_coupling *= coupling_gain;
    float cross_coupling = 0.0;
    for (uint source_dim = 0u; source_dim < AKD_PHASE_DIMENSIONS; ++source_dim) {
        if (source_dim == dim) {
            continue;
        }
        float raw_weight = cross_coupling_weight(valid_layout, dim, source_dim);
        float source_participation = is_static_phase_dimension(valid_layout, source_dim) ? 0.0 : 1.0;
        float phase_delta = phase_component(phase_lo, phase_hi, source_dim) - phase;
        cross_coupling += raw_weight * sin(phase_delta) * source_participation;
    }
    cross_coupling *= coupling.cross_strength * coupling_gain;

    float frequency = natural_frequency_hz(particle, dim) * coupling.nat_freq_multiplier;
    return wrap_tau(phase + dt * (frequency + neighbor_coupling + cross_coupling) * TAU);
}

vec3 hsv_to_rgb(float hue, float saturation, float value) {
    vec3 p = abs(fract(vec3(hue) + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    vec3 rgb = clamp(p - 1.0, vec3(0.0), vec3(1.0));
    return value * mix(vec3(1.0), rgb, saturation);
}

vec3 color_stop_or(bool valid_layout, uint stop_index, vec3 fallback) {
    if (!valid_layout) {
        return fallback;
    }
    uint row_offset = 2u + stop_index;
    return vec3(
        visual_f32(row_offset, 0u),
        visual_f32(row_offset, 1u),
        visual_f32(row_offset, 2u));
}

vec3 sample_akd_gradient(float value01, bool valid_layout) {
    vec3 c0 = color_stop_or(valid_layout, 0u, vec3(0.0, 0.50980395, 0.15888166));
    vec3 c1 = color_stop_or(valid_layout, 1u, vec3(0.7169812, 0.7169812, 0.0));
    vec3 c2 = color_stop_or(valid_layout, 2u, vec3(0.6603774, 0.0, 0.0));
    vec3 c3 = color_stop_or(valid_layout, 3u, vec3(0.026636988, 0.0, 0.6792453));
    vec3 c4 = color_stop_or(valid_layout, 4u, vec3(0.0, 0.50980395, 0.16078432));
    float t = clamp(value01, 0.0, 1.0) * 4.0;
    if (t < 1.0) { return mix(c0, c1, t); }
    if (t < 2.0) { return mix(c1, c2, t - 1.0); }
    if (t < 3.0) { return mix(c2, c3, t - 2.0); }
    return mix(c3, c4, t - 3.0);
}

vec3 apply_saturation_brightness(vec3 rgb, float saturation, float brightness) {
    float luma = dot(rgb, vec3(0.299, 0.587, 0.114));
    vec3 gray = vec3(luma);
    return clamp(mix(gray, rgb, clamp(saturation, 0.0, 1.0)) * max(brightness, 0.0), vec3(0.0), vec3(1.0));
}

bool saturation_brightness_clips(vec3 rgb, float saturation, float brightness) {
    float luma = dot(rgb, vec3(0.299, 0.587, 0.114));
    vec3 gray = vec3(luma);
    vec3 mapped = mix(gray, rgb, clamp(saturation, 0.0, 1.0)) * max(brightness, 0.0);
    return any(lessThan(mapped, vec3(0.0))) || any(greaterThan(mapped, vec3(1.0)));
}

void build_tangent_basis(vec3 normal, out vec3 tangent, out vec3 bitangent) {
    vec3 up = abs(normal.y) < 0.98 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    tangent = normalize(cross(up, normal));
    bitangent = normalize(cross(normal, tangent));
}

void write_tracer_slots(
    uint index,
    uint particle_count,
    uint tracer_max_count,
    uint tracer_draw_count,
    bool reset_state,
    float dt,
    vec3 world_position,
    float radius_m,
    vec3 color,
    float alpha,
    vec3 normal,
    float rotation,
    float animation_frame01,
    float local_coherence) {
    if (tracer_max_count == 0u || particle_count == 0u) {
        return;
    }

    float lifetime_s = max(pc.tracer_params.z, dt);
    float copies_per_second = max(pc.tracer_params.w, 0.0);
    uint state_per_oscillator = tracer_max_count / particle_count;
    if (state_per_oscillator == 0u) {
        return;
    }
    uint draw_per_oscillator = min(tracer_draw_count / particle_count, state_per_oscillator);
    uint slot_base = index * state_per_oscillator;
    if (slot_base >= tracer_max_count) {
        return;
    }

    for (uint draw_local_slot = 0u; draw_local_slot < draw_per_oscillator; ++draw_local_slot) {
        uint output_tracer_slot = index * draw_per_oscillator + draw_local_slot;
        if (output_tracer_slot >= tracer_draw_count) {
            break;
        }
        uint output_base = (particle_count + output_tracer_slot) * 4u;
        particle_output.rows[output_base] = vec4(0.0);
        particle_output.rows[output_base + 1u] = vec4(0.0);
        particle_output.rows[output_base + 2u] = vec4(0.0);
        particle_output.rows[output_base + 3u] = vec4(0.0);
    }

    float previous_time_s = max(pc.params1.z - dt, 0.0);
    uint previous_spawn_tick = uint(floor(previous_time_s * copies_per_second));
    uint current_spawn_tick = uint(floor(max(pc.params1.z, 0.0) * copies_per_second));
    bool spawn = !reset_state && copies_per_second > 0.0 && current_spawn_tick > previous_spawn_tick;
    uint spawn_local_slot = spawn
        ? ((current_spawn_tick - 1u) % state_per_oscillator)
        : 0u;

    for (uint local_slot = 0u; local_slot < state_per_oscillator; ++local_slot) {
        uint tracer_slot = slot_base + local_slot;
        if (tracer_slot >= tracer_max_count) {
            break;
        }

        uint tracer_state_base = particle_count * 2u + tracer_slot * AKD_TRACER_STATE_ROWS_PER_SLOT;
        vec4 previous_position_radius = phase_source.rows[tracer_state_base];
        vec4 previous_color_alpha = phase_source.rows[tracer_state_base + 1u];
        vec4 previous_normal_flags = phase_source.rows[tracer_state_base + 2u];
        vec4 previous_aux_age = phase_source.rows[tracer_state_base + 3u];
        float previous_age = reset_state ? lifetime_s : previous_aux_age.w;
        bool spawn_this_slot = spawn && local_slot == spawn_local_slot;

        vec4 tracer_position_radius = previous_position_radius;
        vec4 tracer_color_alpha = previous_color_alpha;
        vec4 tracer_normal_flags = previous_normal_flags;
        vec4 tracer_aux_age = vec4(previous_aux_age.xyz, previous_age + dt);
        if (spawn_this_slot) {
            tracer_position_radius = vec4(world_position, max(radius_m, 0.0005));
            tracer_color_alpha = vec4(color, alpha);
            tracer_normal_flags = vec4(normal, 1.0);
            tracer_aux_age = vec4(rotation, animation_frame01, local_coherence, 0.0);
        }

        bool alive = tracer_aux_age.w < lifetime_s;
        phase_target.rows[tracer_state_base] = alive ? tracer_position_radius : vec4(0.0);
        phase_target.rows[tracer_state_base + 1u] = alive
            ? tracer_color_alpha
            : vec4(0.0);
        phase_target.rows[tracer_state_base + 2u] = alive
            ? tracer_normal_flags
            : vec4(0.0);
        phase_target.rows[tracer_state_base + 3u] = alive
            ? tracer_aux_age
            : vec4(0.0, 0.0, 0.0, lifetime_s);

        if (!alive) {
            continue;
        }

        if (draw_per_oscillator == 0u || current_spawn_tick == 0u) {
            continue;
        }
        uint recency_rank = (current_spawn_tick - 1u + state_per_oscillator - local_slot)
            % state_per_oscillator;
        if (recency_rank >= draw_per_oscillator) {
            continue;
        }
        uint output_tracer_slot = index * draw_per_oscillator + recency_rank;
        if (output_tracer_slot >= tracer_draw_count) {
            continue;
        }
        uint output_base = (particle_count + output_tracer_slot) * 4u;

        float age01 = clamp(tracer_aux_age.w / lifetime_s, 0.0, 1.0);
        float life01 = 1.0 - age01;
        float fade_in01 = smoothstep(0.0, TRACER_FADE_IN_FRACTION, age01);
        float snapshot_coherence = tracer_aux_age.z;
        float tracer_alpha = clamp(
            tracer_color_alpha.a * life01 * fade_in01,
            0.0,
            1.0);

        particle_output.rows[output_base] = vec4(tracer_position_radius.xyz, tracer_position_radius.w);
        particle_output.rows[output_base + 1u] = vec4(tracer_color_alpha.rgb, tracer_alpha);
        particle_output.rows[output_base + 2u] = tracer_normal_flags;
        particle_output.rows[output_base + 3u] = vec4(tracer_aux_age.x, tracer_aux_age.y, 2.0, life01 * snapshot_coherence);
    }
}

uint anchor_echo_max_count() {
    return uint(PRIVATE_PARTICLE_ANCHOR_ECHO_MAX_COUNT);
}

uint anchor_echo_draw_echo_count() {
    return min(
        uint(PRIVATE_PARTICLE_ANCHOR_ECHO_DRAW_ECHO_COUNT),
        anchor_echo_max_count());
}

uint anchor_echo_draw_count() {
    uint max_count = anchor_echo_max_count();
    return max_count == 0u ? 0u : 1u + anchor_echo_draw_echo_count();
}

float retention_ping_pong_frame01(float progress01) {
    float value = clamp(progress01, 0.0, 1.0);
    return to_half_open_phase01(1.0 - abs(value * 2.0 - 1.0));
}

void write_current_anchor_output(
    uint particle_count,
    uint tracer_draw_count,
    vec3 anchor_position,
    vec3 normal) {
    uint total_draw_count = anchor_echo_draw_count();
    if (total_draw_count == 0u) {
        return;
    }

    uint output_base_index = particle_count + tracer_draw_count;
    float radius_m = max(float(PRIVATE_PARTICLE_ANCHOR_ECHO_RADIUS_M), 0.0005);
    float alpha = clamp(float(PRIVATE_PARTICLE_ANCHOR_ECHO_ALPHA), 0.0, 1.0);
    uint anchor_output_base = output_base_index * 4u;
    particle_output.rows[anchor_output_base] = vec4(anchor_position, radius_m);
    particle_output.rows[anchor_output_base + 1u] = vec4(1.0, 1.0, 1.0, alpha);
    particle_output.rows[anchor_output_base + 2u] = vec4(normal, 1.0);
    particle_output.rows[anchor_output_base + 3u] = vec4(0.0, 0.0, 3.0, 1.0);
}

void main() {
    uint index = gl_GlobalInvocationID.x;
    uint particle_count = uint(max(pc.params0.x, 0.0));
    if (index >= particle_count) {
        return;
    }

    bool valid_layout = aux0_layout_valid(particle_count);
    bool reset_state = pc.params1.x < 1.5 || !valid_layout;
    float dt = clamp(pc.params1.y, 0.001, 0.033);
    // Generic private-particle driver1 is the low-rate control input for the
    // AKD tier-coupling progress and active Sussex coherence-routed visual
    // parameters. Ranges and per-parameter policy remain payload-owned.
    float coupling_driver01 = generic_driver_value01(1u, clamp(pc.params1.w, 0.0, 1.0));
    float coupling_strength = valid_layout ? header_f32(6u, 0u) : 1.0;

    vec4 phase_lo;
    vec4 phase_hi;
    read_phase_pair(index, reset_state, phase_lo, phase_hi);

    uint particle_meta_start = valid_layout ? neighbor_aux0.rows[1].x : 0u;
    uvec4 tier_meta0 = valid_layout
        ? neighbor_aux0.rows[particle_meta_start + index * 2u]
        : uvec4(0u);
    uvec4 tier_meta1 = valid_layout
        ? neighbor_aux0.rows[particle_meta_start + index * 2u + 1u]
        : uvec4(0u);

    vec4 next_lo = phase_lo;
    vec2 next_hi = phase_hi.xy;
    if (valid_layout) {
        CouplingFrame coupling = resolve_coupling_frame(valid_layout, coupling_driver01);
        for (uint dim = 0u; dim < AKD_PHASE_DIMENSIONS; ++dim) {
            float integrated_phase = integrate_dimension(
                index,
                dim,
                phase_lo,
                phase_hi,
                tier_meta0,
                tier_meta1,
                reset_state,
                dt,
                valid_layout,
                coupling);
            float static_dimension = is_static_phase_dimension(valid_layout, dim) ? 1.0 : 0.0;
            float next_phase = mix(integrated_phase, 0.0, static_dimension);
            set_phase_component(next_lo, next_hi, dim, next_phase);
        }
    }


    vec2 order_dim0 = vec2(cos(next_lo.x), sin(next_lo.x));
    uint order_count = 1u;
    uint tier1_start = valid_layout ? neighbor_aux0.rows[1].y : 0u;
    for (uint slot = 0u; slot < tier_meta0.x; ++slot) {
        uint neighbor = packed_u32(tier1_start, tier_meta0.y + slot);
        vec4 neighbor_lo;
        vec4 neighbor_hi;
        read_phase_pair(neighbor, reset_state, neighbor_lo, neighbor_hi);
        order_dim0 += vec2(cos(neighbor_lo.x), sin(neighbor_lo.x));
        order_count += 1u;
    }
    float local_coherence = clamp(length(order_dim0) / float(max(order_count, 1u)), 0.0, 1.0);

    vec4 static_normal = sphere_normals.rows[index];
    vec3 normal = normalized_or(static_normal.xyz, vec3(0.0, 1.0, 0.0));
    float sphere_scale_m = max(pc.world_center_scale.w, 0.01);
    float alpha = 1.0;

    vec4 next_phase_hi = vec4(next_hi, 0.0, 0.0);
    // Slot5 aux ABI: enabled, source0..3, scale-with-particle-size, reference m.
    vec4 explicit_orbit = panel_control_aux_row(5u);
    bool explicit_orbit_enabled = explicit_orbit.x > 0.5;
    uint explicit_orbit_source = panel_u32(explicit_orbit.y, 3u);
    bool showcase_heartbeat_orbit = false;
    float heartbeat_baseline01 = header_f32_or(valid_layout, 24u, 2u, 0.1);
    float heartbeat_peak01 = header_f32_or(valid_layout, 24u, 3u, 1.0);
    float heartbeat_decay_seconds = max(header_f32_or(valid_layout, 25u, 3u, 0.25), dt);
    float previous_heartbeat_envelope01 = reset_state
        ? heartbeat_baseline01
        : clamp(phase_hi.z, heartbeat_baseline01, heartbeat_peak01);
    float heartbeat_event01 = generic_driver_value01(5u, 0.0);
    float heartbeat_envelope01 = heartbeat_event01 >= 0.5
        ? heartbeat_peak01
        : heartbeat_baseline01
            + (previous_heartbeat_envelope01 - heartbeat_baseline01)
                * exp(-dt / heartbeat_decay_seconds);
    next_phase_hi.z = clamp(heartbeat_envelope01, heartbeat_baseline01, heartbeat_peak01);
    if (explicit_orbit_enabled) {
        next_phase_hi.z = orbit_control_next_state(
            explicit_orbit_source, phase_hi.z, reset_state, heartbeat_event01, dt);
    }
#if PRIVATE_PARTICLE_FRAME_EXTENSION_V1
    showcase_heartbeat_orbit = explicit_orbit_enabled && explicit_orbit_source >= 2u;
    if (showcase_heartbeat_orbit) {
        float previous = reset_state ? 0.1 : orbit_control_envelope(phase_hi.z, explicit_orbit_source);
        next_phase_hi.z = orbit_control_pack_state(explicit_orbit_source,
            showcase_heartbeat_pulse(previous, heartbeat_event01, dt));
    }
#endif
    uint color_dim = header_u32_or(valid_layout, 9u, 0u, 0u);
    uint size_dim = header_u32_or(valid_layout, 9u, 1u, 1u);
    uint rotation_dim = header_u32_or(valid_layout, 9u, 2u, 2u);
    uint orbit_radius_dim = header_u32_or(valid_layout, 9u, 3u, 3u);
    uint orbit_angle_dim = header_u32_or(valid_layout, 10u, 0u, 3u);
    uint wave_dim = header_u32_or(valid_layout, 10u, 1u, 4u);
    uint animation_dim = header_u32_or(valid_layout, 10u, 2u, 5u);
    uint alpha_dim = header_u32_or(valid_layout, 10u, 3u, 0u);
    uint saturation_dim = header_u32_or(valid_layout, 11u, 0u, 0u);
    uint brightness_dim = header_u32_or(valid_layout, 11u, 1u, 0u);

    uint color_cycle = header_u32_or(valid_layout, 11u, 2u, 2u);
    uint size_cycle = panel_cycle_or(1u, header_u32_or(valid_layout, 11u, 3u, 1u));
    uint wave_cycle = panel_cycle_or(2u, header_u32_or(valid_layout, 12u, 0u, 3u));
    uint alpha_cycle = header_u32_or(valid_layout, 12u, 1u, 1u);
    uint saturation_cycle = header_u32_or(valid_layout, 12u, 2u, 1u);
    uint brightness_cycle = header_u32_or(valid_layout, 12u, 3u, 1u);
    uint orbit_radius_cycle = panel_cycle_or(7u, header_u32_or(valid_layout, 13u, 0u, 1u));
    if (showcase_heartbeat_orbit) {
        orbit_radius_cycle = panel_u32(panel_control_meta(5u).w, 10u);
    }
    uint orbit_angle_cycle = panel_cycle_or(8u, header_u32_or(valid_layout, 13u, 1u, 1u));
    uint animation_cycle = panel_cycle_or(9u, header_u32_or(valid_layout, 13u, 2u, 1u));
    uint spin_cycle = panel_cycle_or(6u, header_u32_or(valid_layout, 13u, 3u, 0u));

    float deformation_radius_progress01 = generic_driver_value01(0u, clamp(pc.params0.w, 0.0, 1.0));
    float color_phase01 = visual_driver_phase01(valid_layout, 0u, next_lo, next_phase_hi, color_dim, color_cycle);
    float size_phase01 = visual_driver_phase01(valid_layout, 1u, next_lo, next_phase_hi, size_dim, size_cycle);
    float wave_phase01 = visual_driver_phase01(valid_layout, 2u, next_lo, next_phase_hi, wave_dim, wave_cycle);
    float alpha_phase01 = visual_driver_phase01(valid_layout, 3u, next_lo, next_phase_hi, alpha_dim, alpha_cycle);
    float saturation_phase01 = visual_driver_phase01(valid_layout, 4u, next_lo, next_phase_hi, saturation_dim, saturation_cycle);
    float brightness_phase01 = visual_driver_phase01(valid_layout, 5u, next_lo, next_phase_hi, brightness_dim, brightness_cycle);
    float spin_phase01 = visual_driver_phase01(valid_layout, 6u, next_lo, next_phase_hi, rotation_dim, spin_cycle);
    float orbit_radius_phase01 = visual_driver_phase01(valid_layout, 7u, next_lo, next_phase_hi, orbit_radius_dim, orbit_radius_cycle);
    if (explicit_orbit_enabled) {
        // The radius source selection leaves the independent angle channel intact.
        orbit_radius_phase01 = visual_phase01(next_lo, next_phase_hi, orbit_radius_dim, orbit_radius_cycle);
    }
    float orbit_angle_phase01 = visual_driver_phase01(valid_layout, 8u, next_lo, next_phase_hi, orbit_angle_dim, orbit_angle_cycle);
    float animation_phase01 = visual_driver_phase01(valid_layout, 9u, next_lo, next_phase_hi, animation_dim, animation_cycle);

    float color_driver01 = sample_visual_curve01(valid_layout, 0u, color_phase01, color_phase01);
    float size_weight01 = sample_visual_curve01(valid_layout, 1u, size_phase01, akd_hump_envelope(size_phase01));
    float wave_weight01 = sample_visual_curve01(valid_layout, 2u, wave_phase01, akd_hump_envelope(wave_phase01));
    float alpha_weight01 = sample_visual_curve01(valid_layout, 3u, alpha_phase01, alpha_phase01);
    float saturation_weight01 = sample_visual_curve01(valid_layout, 4u, saturation_phase01, saturation_phase01);
    float brightness_weight01 = sample_visual_curve01(valid_layout, 5u, brightness_phase01, brightness_phase01);
    float spin_weight01 = sample_visual_curve01(valid_layout, 6u, spin_phase01, akd_hump_envelope(spin_phase01));
    float orbit_radius_weight01 = sample_visual_curve01(valid_layout, 7u, orbit_radius_phase01, akd_hump_envelope(orbit_radius_phase01));
    if (explicit_orbit_enabled) {
        orbit_radius_weight01 = orbit_control_weight(
            explicit_orbit_source, orbit_radius_weight01,
            orbit_control_envelope(next_phase_hi.z, explicit_orbit_source));
    }
#if PRIVATE_PARTICLE_FRAME_EXTENSION_V1
    if (showcase_heartbeat_orbit) {
        float pulse = orbit_control_envelope(next_phase_hi.z, explicit_orbit_source);
        orbit_radius_weight01 = showcase_heartbeat_orbit_weight(resolve_cycle_phase01(pulse, orbit_radius_cycle));
    }
#endif
    float orbit_angle_weight01 = sample_visual_curve01(valid_layout, 8u, orbit_angle_phase01, orbit_angle_phase01);
    float animation_weight01 = sample_visual_curve01(valid_layout, 9u, animation_phase01, akd_hump_envelope(animation_phase01));

    vec4 explicit_size = panel_control_aux_row(2u);
    bool explicit_size_enabled = explicit_size.x > 0.5;
    uint explicit_size_mode = panel_u32(explicit_size.y, PANEL_PARTICLE_SIZE_MODE_WORLD_METERS);
    float explicit_size_base = max(explicit_size.z, 0.0);
    float explicit_size_reciprocal = 1.0 - clamp(explicit_size.w / 100.0, 0.0, 0.9);
    float explicit_size_min = explicit_size_base * explicit_size_reciprocal;
    float explicit_size_max = explicit_size_base / max(explicit_size_reciprocal, 0.1);
    float size_value = explicit_size_enabled
        ? mix(explicit_size_min, explicit_size_max, size_weight01)
        : mix(
            panel_range_endpoint_or(valid_layout, 1u, 14u, 0u, 0.04, false),
            panel_range_endpoint_or(valid_layout, 1u, 14u, 1u, 0.115, true),
            size_weight01);
    if (size_cycle == 0u) {
        size_value = explicit_size_enabled ? explicit_size_base : 0.01;
    }
    float particle_visual_scale = clamp(pc.params0.y, 0.05, 1.0);
    vec3 sim_right;
    vec3 sim_up;
    vec3 sim_forward;
    private_sim_to_world_basis(sim_right, sim_up, sim_forward);
    vec3 deformation_axis = vec3(0.0, 0.0, -1.0);
    vec3 deformed_surface_offset_sim = compute_deformed_surface_offset(
        normal,
        sphere_scale_m,
        deformation_axis,
        deformation_radius_progress01,
        valid_layout);
    float local_surface_radius = max(length(deformed_surface_offset_sim), 0.0001);
    vec3 depth_dir_sim = -deformed_surface_offset_sim / local_surface_radius;
    float radius_m = explicit_size_enabled
        ? (explicit_size_mode == PANEL_PARTICLE_SIZE_MODE_WORLD_METERS
            ? max(size_value, 0.001)
            : max((size_value / 100.0) * local_surface_radius, 0.001))
        : max(size_value * local_surface_radius * particle_visual_scale, 0.001);

    float depth_percent = mix(
        panel_range_endpoint_or(valid_layout, 2u, 14u, 2u, 0.0, false),
        panel_range_endpoint_or(valid_layout, 2u, 14u, 3u, 0.1, true),
        wave_weight01);
    if (wave_cycle == 0u) {
        depth_percent = 0.0;
    }
    float wave = sin(wave_phase01 * TAU * 2.0) * local_surface_radius * max(depth_percent, 0.0);

    vec3 tangent;
    vec3 bitangent;
    build_tangent_basis(normal, tangent, bitangent);
    float orbit_minimum = panel_range_endpoint_or(valid_layout, 7u, 17u, 0u, 0.2, false);
    float orbit_maximum = panel_range_endpoint_or(valid_layout, 7u, 17u, 1u, 1.5, true);
    if (showcase_heartbeat_orbit) {
        vec4 range = panel_control_value_row(5u);
        orbit_minimum = range.x;
        orbit_maximum = range.y;
    }
    float signed_orbit_radius = orbit_control_radius(
        explicit_orbit_enabled, explicit_orbit_source, explicit_orbit.z > 0.5, explicit_orbit.w, radius_m,
        orbit_radius_weight01, orbit_minimum, orbit_maximum, orbit_radius_cycle == 0u);
    if (showcase_heartbeat_orbit && orbit_radius_cycle == 0u) {
        signed_orbit_radius = 0.0;
    }
    float orbit_sign = signed_orbit_radius < 0.0 ? -1.0 : 1.0;
    float orbit_radius = abs(signed_orbit_radius);
    float orbit_angle = mix(
        panel_range_endpoint_or(valid_layout, 8u, 17u, 2u, 0.0, false),
        panel_range_endpoint_or(valid_layout, 8u, 17u, 3u, TAU, true),
        orbit_angle_weight01);
    if (orbit_angle_cycle == 0u) {
        orbit_angle = 0.0;
    } else {
        orbit_angle *= orbit_sign;
    }
    vec3 orbit_offset_sim = (tangent * cos(orbit_angle) + bitangent * sin(orbit_angle)) * orbit_radius;
    vec3 surface_offset_sim = deformed_surface_offset_sim
        + depth_dir_sim * wave
        + orbit_offset_sim;
    vec3 world_surface_offset =
        private_sim_offset_to_world(surface_offset_sim, sim_right, sim_up, sim_forward);
    vec3 world_deformed_surface_offset =
        private_sim_offset_to_world(deformed_surface_offset_sim, sim_right, sim_up, sim_forward);
    vec3 world_normal = normalized_or(
        private_sim_offset_to_world(normal, sim_right, sim_up, sim_forward),
        vec3(0.0, 1.0, 0.0));
    vec3 world_anchor_center = private_world_anchor_center();
    vec3 world_position = world_anchor_center + world_surface_offset;

    float previous_rotation = reset_state ? 0.0 : phase_hi.w;
    float spin_speed = mix(
        panel_range_endpoint_or(valid_layout, 6u, 16u, 2u, 0.1, false),
        panel_range_endpoint_or(valid_layout, 6u, 16u, 3u, 0.5, true),
        spin_weight01);
    if (spin_cycle == 0u) {
        spin_speed = 0.0;
    }
    float computed_rotation = wrap_tau(previous_rotation + TAU * max(spin_speed, 0.0) * dt);
    float rotation = mix(
        computed_rotation,
        0.0,
        is_static_phase_dimension(valid_layout, rotation_dim) ? 1.0 : 0.0);
    float animation_value01 = animation_weight01;
    uint animation_panel_slot = panel_control_slot_for_curve_index(9u);
    if (animation_panel_slot != PANEL_CONTROL_NO_SLOT
        && panel_control_mode(animation_panel_slot) != PANEL_DRIVER_MODE_DIRECT) {
        vec4 animation_range = panel_control_value_row(animation_panel_slot);
        animation_value01 = mix(animation_range.x, animation_range.y, animation_weight01);
    }
    float computed_animation_frame01 = to_half_open_phase01(animation_value01);
    float animation_frame01 = mix(
        computed_animation_frame01,
        0.0,
        is_static_phase_dimension(valid_layout, animation_dim) || animation_cycle == 0u ? 1.0 : 0.0);

    vec3 color = sample_akd_gradient(color_driver01, valid_layout);
    float saturation = mix(
        header_f32_or(valid_layout, 15u, 2u, 0.3),
        header_f32_or(valid_layout, 15u, 3u, 1.0),
        saturation_weight01);
    if (saturation_cycle == 0u) {
        saturation = 1.0;
    }
    float brightness = mix(
        header_f32_or(valid_layout, 16u, 0u, 0.3),
        header_f32_or(valid_layout, 16u, 1u, 1.0),
        brightness_weight01);
    if (brightness_cycle == 0u) {
        brightness = 1.0;
    }
    color = apply_saturation_brightness(color, saturation, brightness);
    float transparency = mix(
        header_f32_or(valid_layout, 15u, 0u, 1.0),
        header_f32_or(valid_layout, 15u, 1u, 1.0),
        alpha_weight01);
    if (alpha_cycle == 0u) {
        transparency = 1.0;
    }
    alpha *= clamp(transparency, 0.0, 1.0) * clamp(pc.transparency_params.x, 0.0, 1.0);

    uint state_base = index * 2u;
    phase_target.rows[state_base] = next_lo;
    phase_target.rows[state_base + 1u] = vec4(next_hi, next_phase_hi.z, rotation);

    uint base = index * 4u;
    particle_output.rows[base] = vec4(world_position, radius_m);
    particle_output.rows[base + 1u] = vec4(color, alpha);
    particle_output.rows[base + 2u] = vec4(world_normal, 1.0);
    particle_output.rows[base + 3u] = vec4(rotation, animation_frame01, 1.0, local_coherence);

    uint tracer_max_count = uint(max(pc.tracer_params.y, 0.0));
    uint total_draw_count = uint(max(pc.tracer_params.x, 0.0));
    uint retention_draw_count = anchor_echo_draw_count();
    uint tracer_draw_count = total_draw_count > particle_count + retention_draw_count
        ? total_draw_count - particle_count - retention_draw_count
        : 0u;
    write_tracer_slots(
        index,
        particle_count,
        tracer_max_count,
        tracer_draw_count,
        reset_state,
        dt,
        world_position,
        radius_m,
        color,
        alpha,
        world_normal,
        rotation,
        animation_frame01,
        local_coherence);

    uint special_point_index = min(header_u32_or(valid_layout, 19u, 0u, 0u), particle_count - 1u);
    if (index == special_point_index) {
        write_current_anchor_output(
            particle_count,
            tracer_draw_count,
            world_anchor_center + world_deformed_surface_offset,
            world_normal);
    }
}
