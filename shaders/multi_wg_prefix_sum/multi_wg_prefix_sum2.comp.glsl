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

void main() {
  uint W = gl_WorkGroupID.x;
  uint T = gl_LocalInvocationID.x;

  output_data.data[(W * N) + (2 * T)] += output_data.sums[W - 1];
  output_data.data[(W * N) + (2 * T + 1)] += output_data.sums[W - 1];
}
