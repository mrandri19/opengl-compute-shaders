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

uint prefix_sum_return_total(inout uint offsets[B], uint T) {
  uint sum = -100;
  // **************************************************************************
  // Reduce
  uint offset = 1;

  for (uint d = B >> 1; d > 0; d >>= 1) {
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
    sum = offsets[B - 1];
    offsets[B - 1] = 0;
  }

  for (uint d = 1; d < B; d *= 2) {
    offset >>= 1;

    barrier();
    memoryBarrier();

    if (T < d) {
      uint ai = offset * (2 * T + 1) - 1;
      uint bi = offset * (2 * T + 2) - 1;

      uint t = offsets[ai];

      offsets[ai] = offsets[bi];
      offsets[bi] += t;
    }
  }

  return sum;
}

void prefix_sum(inout uint sums[N_OVER_B], uint T) {
  // **************************************************************************
  // Reduce
  uint offset = 1;

  for (uint d = N_OVER_B >> 1; d > 0; d >>= 1) {
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
    sums[N_OVER_B - 1] = 0;
  }

  for (uint d = 1; d < N_OVER_B; d *= 2) {
    offset >>= 1;

    barrier();
    memoryBarrier();

    if (T < d) {
      uint ai = offset * (2 * T + 1) - 1;
      uint bi = offset * (2 * T + 2) - 1;

      uint t = sums[ai];

      sums[ai] = sums[bi];
      sums[bi] += t;
    }
  }
}

shared bool results[B];
shared uint offsets[B];

bool predicate(uint x) { return x % 2 == 0; }

void main() {
  uint W = gl_WorkGroupID.x;
  uint T = gl_LocalInvocationID.x;

  results[2 * T] = predicate(input_data.data[(W * B) + (2 * T)]);
  results[2 * T + 1] = predicate(input_data.data[(W * B) + (2 * T + 1)]);
  barrier();
  memoryBarrier();

  offsets[2 * T] = uint(results[2 * T]);
  offsets[2 * T + 1] = uint(results[2 * T + 1]);
  barrier();
  memoryBarrier();

  // Perform the prefix sum on the input data, returning the total sum
  // replacing the last element with 0
  uint sum = prefix_sum_return_total(offsets, T);
  if (T == 0) {
    output_data.sums[W] = sum;
  }
  // Wait for the prefix sum and the write to be done
  barrier();
  memoryBarrier();

  // Copy wg-shared data back to global memory
  output_data.offsets[(W * B) + (2 * T)] = offsets[2 * T];
  output_data.offsets[(W * B) + (2 * T + 1)] = offsets[2 * T + 1];
  barrier();
  memoryBarrier();

  output_data.results[(W * B) + (2 * T)] = uint(results[2 * T]);
  output_data.results[(W * B) + (2 * T + 1)] = uint(results[2 * T + 1]);
  // Wait for the data to be copied
  barrier();
  memoryBarrier();
}
