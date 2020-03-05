// Blelloch parallel prefix sum/scan
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#version 450 core

// A single thread operates on two items at a time

#define DATA_LEN 32
#define WORK_GROUPS 2
#define N (DATA_LEN / WORK_GROUPS)
#define THREADS (N / 2)

layout(local_size_x = THREADS, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) coherent buffer InputData { float data[N]; }
input_data;

layout(std430, binding = 1) coherent buffer OutputData {
  float sums[WORK_GROUPS];
  float data[N];
}
output_data;

float prefix_sum_return_total(inout float offsets[N], uint T) {
  float sum = -100.;
  // **************************************************************************
  // Reduce
  uint offset = 1;

  for (uint d = N >> 1; d > 0; d >>= 1) {
    barrier();
    memoryBarrier(); // TODO: do I really need both of these? A. yes but maybe
                     // not when I use shared memory instead

    if (T < d) {
      uint ai = offset * (2 * T + 1) - 1;
      uint bi = offset * (2 * T + 2) - 1;
      offsets[bi] += offsets[ai];
    }

    offset *= 2;
  }

  // **************************************************************************
  // Down-sweep
  if (T == 0) {
    sum = offsets[N - 1];
    offsets[N - 1] = 0;
  }

  for (uint d = 1; d < N; d *= 2) {
    offset >>= 1;

    barrier();
    memoryBarrier();

    if (T < d) {
      uint ai = offset * (2 * T + 1) - 1;
      uint bi = offset * (2 * T + 2) - 1;

      float t = offsets[ai];

      offsets[ai] = offsets[bi];
      offsets[bi] += t;
    }
  }

  return sum;
}

void prefix_sum(inout float sums[WORK_GROUPS], uint T) {
  // **************************************************************************
  // Reduce
  uint offset = 1;

  for (uint d = WORK_GROUPS >> 1; d > 0; d >>= 1) {
    barrier();
    memoryBarrier(); // TODO: do I really need both of these? A. yes but maybe
                     // not when I use shared memory instead

    if (T < d) {
      uint ai = offset * (2 * T + 1) - 1;
      uint bi = offset * (2 * T + 2) - 1;
      sums[bi] += sums[ai];
    }

    offset *= 2;
  }

  // **************************************************************************
  // Down-sweep
  if (T == 0) {
    sums[WORK_GROUPS - 1] = 0;
  }

  for (uint d = 1; d < WORK_GROUPS; d *= 2) {
    offset >>= 1;

    barrier();
    memoryBarrier();

    if (T < d) {
      uint ai = offset * (2 * T + 1) - 1;
      uint bi = offset * (2 * T + 2) - 1;

      float t = sums[ai];

      sums[ai] = sums[bi];
      sums[bi] += t;
    }
  }
}

shared float block[N];
shared float increments[WORK_GROUPS];

void main() {
  uint W = gl_WorkGroupID.x;
  uint T = gl_LocalInvocationID.x;

  // Copy global memory data into wg-shared data
  block[2 * T] = input_data.data[(W * N) + (2 * T)];
  block[2 * T + 1] = input_data.data[(W * N) + (2 * T + 1)];
  // Wait for the copy to be done
  barrier();
  memoryBarrier();

  // Perform the prefix sum on the input data, returning the total sum before
  // replacing the last element with 0
  float sum = prefix_sum_return_total(block, T);
  if (T == 0) {
    output_data.sums[W] = sum;
  }
  // Wait for the prefix sum and the write to be done
  barrier();
  memoryBarrier();

  // Copy wg-shared data back to global memory
  output_data.data[(W * N) + (2 * T)] = block[2 * T];
  output_data.data[(W * N) + (2 * T + 1)] = block[2 * T + 1];
  // Wait for the data to be copied
  barrier();
  memoryBarrier();
}
