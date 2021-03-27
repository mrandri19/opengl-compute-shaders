// Blelloch parallel prefix sum/scan
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#version 460 core

// A single thread operates on two items at a time
#define N -1337
#define THREADS (N / 2)

layout(local_size_x = THREADS, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) coherent buffer InputData { float data[]; }
input_data;

void main() {
  uint tid = gl_LocalInvocationID.x;

  // **************************************************************************
  // Reduce (or Up-Sweep)
  uint offset = 1;

  for (uint d = N / 2; d > 0; d /= 2) {
    barrier();

    if (tid < d) {
      uint ai = offset * (2 * tid + 1) - 1;
      uint bi = offset * (2 * tid + 2) - 1;
      input_data.data[bi] += input_data.data[ai];
    }

    offset *= 2;
  }

  // **************************************************************************
  // Down-sweep
  if (tid == 0) {
    input_data.data[N - 1] = 0;
  }

  for (uint d = 1; d < N; d *= 2) {
    offset /= 2;

    barrier();

    if (tid < d) {
      uint ai = offset * (2 * tid + 1) - 1;
      uint bi = offset * (2 * tid + 2) - 1;

      float t = input_data.data[ai];

      input_data.data[ai] = input_data.data[bi];
      input_data.data[bi] += t;
    }
  }
}
