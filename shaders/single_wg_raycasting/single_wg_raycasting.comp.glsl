#version 460 core

#define CHUNK_ROWS -1337
#define CHUNK_COLS -1337
#define CHUNK_SIZE (CHUNK_ROWS * CHUNK_COLS)
#define MAX_ITERS -1337
#define THREADS -1337

layout(local_size_x = THREADS, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) restrict coherent buffer InputData {
  uint chunk[CHUNK_SIZE];
  vec4 ray_start;
  vec4 ray_direction;
}
input_data;

layout(std430, binding = 1) restrict coherent buffer OutputData {
  vec4 hit;
  bool has_hit;
}
output_data;

void main() {
  uint tid = gl_LocalInvocationID.x;

  vec2 ray_start = vec2(input_data.ray_start);
  vec2 ray_direction = vec2(input_data.ray_direction);
  vec2 ray_voxel = vec2(floor(ray_start));
  vec2 step = sign(ray_direction);
  vec2 t_max = ((ray_voxel + step) - ray_start) / ray_direction;

  vec2 t_delta = vec2(1., 1.) / ray_direction * step;

  output_data.has_hit = false;
  for (int i = 0; i < MAX_ITERS; i++) {
    if (t_max.x < t_max.y) {
      ray_voxel.x += step.x;
      t_max.x += t_delta.x;
    } else {
      ray_voxel.y += step.y;
      t_max.y += t_delta.y;
    }
    float x = ray_voxel.x;
    float y = ray_voxel.y;

    if (x >= CHUNK_COLS)
      break;
    if (y >= CHUNK_ROWS)
      break;

    int x_ = int(x);
    int y_ = int(y);
    if (input_data.chunk[CHUNK_COLS * y_ + x_] == 1) {
      input_data.chunk[CHUNK_COLS * y_ + x_] = 3;
      output_data.has_hit = true;
      output_data.hit = vec4(x, y, 0., 0.);
      break;
    }
    input_data.chunk[CHUNK_COLS * y_ + x_] = 2;
  }
}
