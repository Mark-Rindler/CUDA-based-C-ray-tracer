# CUDA-based-C-ray-tracer
Real-time ray tracer capable of rendering hundreds of thousands of polygons at ~100 fps. Implements bounding volume hierarchies (BVH), triangle-based ray tracing, and physically based materials. Includes an additional experimental build for visualizing ray-tracing internals.


IMPORTANT NOTES:
If you plan on building this project you will need an NVIDIA GPU on the following list. Furthermore, you will need to go into the build file and change the flag "-gencode arch=compute_86,code=sm_86" to reflect your GPU's correct flag. Also, ensure that you have downloaded CUDA onto your system and that your drivers are up to date.

-gencode arch=compute_86,code=sm_86 \       # for RTX 30xx / Ampere
  -gencode arch=compute_89,code=sm_89 \       # for Ada (RTX 40xx etc.)
  -gencode arch=compute_90,code=sm_90 \       # for Hopper series
  -gencode arch=compute_100,code=sm_100 \     # for Blackwell
  -gencode arch=compute_86,code=compute_86 \   # PTX fallback
  -gencode arch=compute_89,code=compute_89 \   # etc.
