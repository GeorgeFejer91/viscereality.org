#version 450

// Detailed-only, read-only observer. Semantic compute shaders deliberately do
// not declare the diagnostic binding, so basic and detailed retain identical
// primary/auxiliary SPIR-V and command endpoints.
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 2) readonly buffer PrivateParticleRows {
    vec4 rows[];
} particle_output;
layout(set = 0, binding = 9) buffer PrivateParticleDiagnosticRows {
    int values[];
} diagnostics;

// Independent 16-byte observer push at offset zero: exact frame low/high,
// population, schema. Never extend the 128-byte semantic particle ABI.
layout(push_constant) uniform PrivateParticleDiagnosticPush {
    uvec4 diagnostic_frame;
} pc;

const uint DIAGNOSTIC_SCHEMA_V2 = 2u;
const uint DIAGNOSTIC_VALIDITY_PARTICLE_COUNT = 1u;
const uint DIAGNOSTIC_VALIDITY_FRAME_ID = 2u;
const uint DIAGNOSTIC_VALIDITY_LANES_VISITED = 4u;

void main() {
    if (pc.diagnostic_frame.w != DIAGNOSTIC_SCHEMA_V2) return;
    uint index = gl_GlobalInvocationID.x;
    uint particle_count = pc.diagnostic_frame.z;
    if (index < particle_count) {
        // Reading the established output preserves a data dependency while this
        // generic transport refuses to expose private metric interpretation.
        vec4 observed = particle_output.rows[index * 4u];
        if (observed.w >= 0.0) atomicAdd(diagnostics.values[22], 1);
    }
    if (index == 0u) {
        diagnostics.values[0] = int(particle_count);
        diagnostics.values[20] = int(pc.diagnostic_frame.x);
        diagnostics.values[21] = int(pc.diagnostic_frame.y);
        diagnostics.values[23] = int(DIAGNOSTIC_SCHEMA_V2 |
            ((DIAGNOSTIC_VALIDITY_PARTICLE_COUNT | DIAGNOSTIC_VALIDITY_FRAME_ID |
              DIAGNOSTIC_VALIDITY_LANES_VISITED) << 16u));
    }
}
