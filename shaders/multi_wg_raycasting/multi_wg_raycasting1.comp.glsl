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
#define MAX_ITERS 100

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

shared bool has_hit[B];
shared uint offsets[B];

uvec4 raycast(vec3 ray_start, vec3 ray_direction_, out bool has_hit,
              in uint chunk[CHUNK_SIZE]) {
  vec3 ray_direction = normalize(ray_direction_ + vec3(1e-8, 1e-8, 1e-8));
  vec3 ray_voxel = floor(ray_start);
  vec3 step_ = sign(ray_direction);

  vec3 t_max = ((ray_voxel + step_) - ray_start) / (ray_direction);
  vec3 t_delta = (vec3(1., 1., 1.) / ray_direction) * step_;

  has_hit = false;
  for (int i = 0; i < MAX_ITERS; i++) {
    // Traverse
    if (t_max.x < t_max.y) {
      if (t_max.x < t_max.z) {
        ray_voxel.x += step_.x;
        t_max.x += t_delta.x;
      } else {
        ray_voxel.z += step_.z;
        t_max.z += t_delta.z;
      }
    } else {
      if (t_max.y < t_max.z) {
        ray_voxel.y += step_.y;
        t_max.y += t_delta.y;
      } else {
        ray_voxel.z += step_.z;
        t_max.z += t_delta.z;
      }
    }

    // Check bounds
    if (ray_voxel.x >= CHUNK_X || ray_voxel.x < 0)
      break;
    if (ray_voxel.y >= CHUNK_Y || ray_voxel.y < 0)
      break;
    if (ray_voxel.z >= CHUNK_Z || ray_voxel.z < 0)
      break;

    // Check if we are in a voxel full of data
    int x = int(ray_voxel.x);
    int y = int(ray_voxel.y);
    int z = int(ray_voxel.z);
    if (input_data.chunk[CHUNK_X * CHUNK_Y * z + CHUNK_X * y + x] == 1) {
      // If we are, return the hit position
      has_hit = true;
      return uvec4(x, y, z, 1337);
    }
  }

  return uvec4(-1, -1, -1, -1);
}

void main() {
  uint W = gl_WorkGroupID.x;
  uint T = gl_LocalInvocationID.x;

  uint ix0 = (W * B) + (2 * T);
  uint ix1 = (W * B) + (2 * T + 1);

  vec3 ray_start = vec3(3., 0., 0.);
  vec3 ray_direction0 = vec3(2.0 * (-0.5 + float(ix0) / 7.0), 1.0, 0.0);
  vec3 ray_direction1 = vec3(2.0 * (-0.5 + float(ix1) / 7.0), 1.0, 0.0);

  output_data.hits[ix0] =
      raycast(ray_start, ray_direction0, has_hit[2 * T], input_data.chunk);
  output_data.hits[ix1] =
      raycast(ray_start, ray_direction1, has_hit[2 * T + 1], input_data.chunk);
  barrier();
  memoryBarrier();

  offsets[2 * T] = uint(has_hit[2 * T]);
  offsets[2 * T + 1] = uint(has_hit[2 * T + 1]);
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
  output_data.offsets[ix0] = offsets[2 * T];
  output_data.offsets[ix1] = offsets[2 * T + 1];
  barrier();
  memoryBarrier();

  output_data.has_hit[ix0] = uint(has_hit[2 * T]);
  output_data.has_hit[ix1] = uint(has_hit[2 * T + 1]);
  // Wait for the data to be copied
  barrier();
  memoryBarrier();
}
