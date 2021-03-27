// Blelloch parallel prefix sum/scan
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#version 460 core

// A single thread operates on two items at a time
#define N -1337
#define THREADS (N / 2)

layout(local_size_x = THREADS, local_size_y = 1, local_size_z = 1) in;

// These two need to stay synchronized with the program, of course
layout(std430, binding = 0) restrict coherent buffer InputData {
  float data[N];
}
input_data;

layout(std430, binding = 1) restrict coherent buffer OutputData {
  uint length;
  float data[N];
}
output_data;

// TODO: Verify that this compiles to a pass-by-reference and not a deepcopy. Or
// at least gets inlined
void prefix_sum2(inout uint offsets[N], uint tid) {
  // **************************************************************************
  // Reduce
  uint offset = 1;

  for (uint d = N >> 1; d > 0; d >>= 1) {
    barrier();
    memoryBarrier(); // TODO: do I really need both of these? A. yes but maybe
                     // not when I use shared memory instead

    if (tid < d) {
      uint ai = offset * (2 * tid + 1) - 1;
      uint bi = offset * (2 * tid + 2) - 1;
      offsets[bi] += offsets[ai];
    }

    offset *= 2;
  }

  // **************************************************************************
  // Down-sweep
  if (tid == 0) {
    offsets[N - 1] = 0;
  }

  for (uint d = 1; d < N; d *= 2) {
    offset >>= 1;

    barrier();
    memoryBarrier();

    if (tid < d) {
      uint ai = offset * (2 * tid + 1) - 1;
      uint bi = offset * (2 * tid + 2) - 1;

      uint t = offsets[ai];

      offsets[ai] = offsets[bi];
      offsets[bi] += t;
    }
  }

  barrier();
  memoryBarrier();
}

shared bool results[N];
shared uint offsets[N];

bool predicate(float x) { return int(x) % 2 == 0; }

// FIXME(Andrea): there is some kind of bug in the first element

void main() {
  uint tid = gl_LocalInvocationID.x;

  results[2 * tid] = predicate(input_data.data[2 * tid]);
  results[2 * tid + 1] = predicate(input_data.data[2 * tid + 1]);

  offsets[2 * tid] = uint(results[2 * tid]);
  offsets[2 * tid + 1] = uint(results[2 * tid + 1]);

  prefix_sum2(offsets, tid);

  if (results[2 * tid]) {
    output_data.data[offsets[2 * tid]] = input_data.data[2 * tid];
  }
  if (results[2 * tid + 1]) {
    output_data.data[offsets[2 * tid + 1]] = input_data.data[2 * tid + 1];
  }

  barrier();
  memoryBarrier();

  if (tid == 0) {
    // TODO(Andrea): Could this be done better? Maybe with an atomic?
    for (uint i = N - 1; i >= 0; i--) {
      if (results[i]) {
        output_data.length = offsets[i] + 1;
        break;
      }
    }
  }
}
