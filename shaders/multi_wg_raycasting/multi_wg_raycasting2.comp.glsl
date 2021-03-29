// Blelloch parallel prefix sum/scan
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#version 450 core

#define CHUNK_X -1337
#define CHUNK_Y -1337
#define CHUNK_Z -1337
#define CHUNK_SIZE -1337
#define N -1337
#define B -1337
#define N_OVER_B (N / B)

layout(local_size_x = B / 2, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) coherent readonly buffer InputData {
  uint chunk[CHUNK_SIZE];
  uvec4 ray_start;
}
input_data;

layout(std430, binding = 1) coherent buffer OutputData {
  uint sums[N_OVER_B];
  uint offsets[N];
  uint has_hit[N];
  uvec4 hits[N];
  uvec4 compact_hits[N];
}
output_data;

void main() {
  uint W = gl_WorkGroupID.x;
  uint T = gl_LocalInvocationID.x;

  uint ix0 = (W * B) + (2 * T);
  uint ix1 = (W * B) + (2 * T + 1);

  output_data.offsets[ix0] += output_data.sums[W];
  output_data.offsets[ix1] += output_data.sums[W];

  barrier();
  memoryBarrier();

  if (bool(output_data.has_hit[ix0])) {
    output_data.compact_hits[output_data.offsets[ix0]] = output_data.hits[ix0];
  }

  if (bool(output_data.has_hit[ix1])) {
    output_data.compact_hits[output_data.offsets[ix1]] = output_data.hits[ix1];
  }
}
