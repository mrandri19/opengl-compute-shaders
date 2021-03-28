// Blelloch parallel prefix sum/scan
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#version 450 core

// A single thread operates on two items at a time

#define N -1337
#define B -1337
#define N_OVER_B (N / B)

layout(local_size_x = B / 2, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) coherent buffer InputData { uint data[N]; }
input_data;

layout(std430, binding = 1) coherent buffer OutputData {
  uint sums[N_OVER_B];
  uint offsets[N];
  uint results[N];
  uint data[N];
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

  if (bool(output_data.results[ix0])) {
    output_data.data[output_data.offsets[ix0]] = input_data.data[ix0];
  }

  if (bool(output_data.results[ix1])) {
    output_data.data[output_data.offsets[ix1]] = input_data.data[ix1];
  }
}
