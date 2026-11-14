#version 450

// This pass owns exactly one existing anchor-echo state slot per invocation.
// The primary pass has already written the stable anchor rows in particle_output.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 2) buffer PrivateParticleRows { vec4 rows[]; } particle_output;
layout(set = 0, binding = 3) readonly buffer PhaseSourceRows { vec4 rows[]; } phase_source;
layout(set = 0, binding = 4) buffer PhaseTargetRows { vec4 rows[]; } phase_target;
layout(set = 0, binding = 5) readonly buffer NeighborAux0Rows { uvec4 rows[]; } neighbor_aux0;

layout(push_constant) uniform PrivateParticlePush {
    vec4 params0; vec4 params1; vec4 transparency_params; vec4 tracer_params;
    vec4 world_center_scale; vec4 eye_position; vec4 eye_orientation_xyzw; vec4 fov_tangents;
} pc;

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

const uint AKD_AUX0_MAGIC = 0x32444b41u;
const uint AKD_AUX0_LAYOUT_VERSION = 9u;
const uint AKD_PHASE_DIMENSIONS = 6u;
const uint AKD_TRACER_STATE_ROWS_PER_SLOT = 4u;
const uint AKD_ANCHOR_ECHO_STATE_ROWS_PER_SLOT = 4u;

bool valid_aux0_layout(uint particle_count) {
    uvec4 header = neighbor_aux0.rows[0];
    return header.x == AKD_AUX0_MAGIC && header.y == AKD_AUX0_LAYOUT_VERSION
        && header.z == particle_count && header.w == AKD_PHASE_DIMENSIONS;
}

void clear_echo_output(uint output_base) {
    particle_output.rows[output_base] = vec4(0.0);
    particle_output.rows[output_base + 1u] = vec4(0.0);
    particle_output.rows[output_base + 2u] = vec4(0.0);
    particle_output.rows[output_base + 3u] = vec4(0.0);
}

void main() {
    uint local_slot = gl_GlobalInvocationID.x;
    uint max_count = uint(PRIVATE_PARTICLE_ANCHOR_ECHO_MAX_COUNT);
    if (local_slot >= max_count) return;

    uint particle_count = uint(max(pc.params0.x, 0.0));
    uint draw_echo_count = min(uint(PRIVATE_PARTICLE_ANCHOR_ECHO_DRAW_ECHO_COUNT), max_count);
    uint total_draw_count = uint(max(pc.tracer_params.x, 0.0));
    uint anchor_draw_count = max_count == 0u ? 0u : 1u + draw_echo_count;
    uint tracer_draw_count = total_draw_count > particle_count + anchor_draw_count
        ? total_draw_count - particle_count - anchor_draw_count : 0u;
    uint output_base_index = particle_count + tracer_draw_count;
    // Recency rank is a bijection over slots; each invocation clears only the
    // output row it can subsequently fill.
    if (local_slot < draw_echo_count) clear_echo_output((output_base_index + 1u + local_slot) * 4u);
    if (draw_echo_count == 0u) return;

    // Must exactly match the primary integration clamp so resets and age
    // advancement remain equivalent across the dispatch boundary.
    float dt = clamp(pc.params1.y, 0.001, 0.033);
    float lifetime_s = max(float(PRIVATE_PARTICLE_ANCHOR_ECHO_LIFETIME_SECONDS), dt);
    float copies_per_second = max(float(PRIVATE_PARTICLE_ANCHOR_ECHO_COPIES_PER_SECOND), 0.0);
    bool reset_state = pc.params1.x < 1.5 || !valid_aux0_layout(particle_count);
    float previous_time_s = max(pc.params1.z - dt, 0.0);
    uint previous_tick = uint(floor(previous_time_s * copies_per_second));
    uint current_tick = uint(floor(max(pc.params1.z, 0.0) * copies_per_second));
    bool spawn = !reset_state && copies_per_second > 0.0 && current_tick > previous_tick;
    uint spawn_slot = spawn ? ((current_tick - 1u) % max_count) : 0u;
    uint state_base = particle_count * 2u + uint(max(pc.tracer_params.y, 0.0))
        * AKD_TRACER_STATE_ROWS_PER_SLOT + local_slot * AKD_ANCHOR_ECHO_STATE_ROWS_PER_SLOT;
    vec4 previous_aux_age = phase_source.rows[state_base + 3u];
    float previous_age = reset_state ? lifetime_s : previous_aux_age.w;
    bool spawn_this_slot = spawn && local_slot == spawn_slot;
    float age = spawn_this_slot ? 0.0 : previous_age + dt;
    bool alive = age < lifetime_s;
    uint anchor_output_base = output_base_index * 4u;
    vec4 anchor_position_radius = particle_output.rows[anchor_output_base];
    vec4 anchor_color_alpha = particle_output.rows[anchor_output_base + 1u];
    vec4 anchor_normal_flags = particle_output.rows[anchor_output_base + 2u];
    phase_target.rows[state_base] = alive ? anchor_position_radius : vec4(0.0);
    phase_target.rows[state_base + 1u] = alive ? anchor_color_alpha : vec4(0.0);
    phase_target.rows[state_base + 2u] = alive ? anchor_normal_flags : vec4(0.0);
    phase_target.rows[state_base + 3u] = alive ? vec4(0.0, 0.0, 1.0, age) : vec4(0.0, 0.0, 0.0, lifetime_s);
    // Output ownership is by recency rank, never by the state slot above.
    // This rank lane is the sole clearer and writer for its four output rows.
    if (local_slot >= draw_echo_count || current_tick == 0u) return;
    uint output_rank = local_slot;
    uint source_slot = (current_tick - 1u + max_count - output_rank) % max_count;
    uint source_state_base = particle_count * 2u + uint(max(pc.tracer_params.y, 0.0))
        * AKD_TRACER_STATE_ROWS_PER_SLOT + source_slot * AKD_ANCHOR_ECHO_STATE_ROWS_PER_SLOT;
    float source_previous_age = reset_state ? lifetime_s : phase_source.rows[source_state_base + 3u].w;
    float source_age = spawn && source_slot == spawn_slot ? 0.0 : source_previous_age + dt;
    if (source_age >= lifetime_s) return;
    uint output_base = (output_base_index + 1u + output_rank) * 4u;
    float life01 = 1.0 - clamp(source_age / lifetime_s, 0.0, 1.0);
    particle_output.rows[output_base] = anchor_position_radius;
    particle_output.rows[output_base + 1u] = anchor_color_alpha;
    particle_output.rows[output_base + 2u] = anchor_normal_flags;
    particle_output.rows[output_base + 3u] = vec4(0.0, 0.0, 4.0, life01);
}
