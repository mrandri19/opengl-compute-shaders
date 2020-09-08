# opengl-compute-shaders
> **⚠️WARNING⚠️**: Highly experimental🧪, hobby code. Don't even think about using this in production!

Experiments with OpenGL 4.5 Compute Shaders using Rust.

These gpu programs were written while following [CS344 Introduction to Parallel Programming](https://github.com/udacity/cs344). The class uses CUDA, which would
have made this work much easier (or at least, I would have worked with much better documentation), but I have decided to use OpenGL instead.
This is because the aim of this exploration was to develop a compute shader for blazing fast occlusion detection, which I needed for a game I was developing.

Many of the implementation details come from [GPU Gems 3 - Chapter 39](https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda).

## Algorithms Implemented

- ### [Single workgroup, Blelloch prefix sum](https://github.com/mrandri19/opengl-compute-shaders/blob/master/shaders/prefix_sum.comp)

- ### [Single workgroup, Blelloch prefix sum with final parallel compaction](https://github.com/mrandri19/opengl-compute-shaders/blob/master/shaders/compaction.comp)

- ### [Multi-workgroup, Blelloch prefix sum](https://github.com/mrandri19/opengl-compute-shaders/tree/master/shaders/multi_wg_prefix_sum_float)

- ### [Multi-workgroup, Blelloch prefix sum with final parallel compaction](https://github.com/mrandri19/opengl-compute-shaders/tree/master/shaders/multi_wg_compaction)

- ### [Raycasting for occlusion detection](https://github.com/mrandri19/opengl-compute-shaders/tree/master/shaders/multi_wg_raycasting)
