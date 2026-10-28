/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "rasterize_points.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/backward_cpu.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("set_backward_timing_enabled", &CudaRasterizer::Rasterizer::set_backward_timing_enabled);
  m.def("get_backward_timing", &CudaRasterizer::Rasterizer::get_backward_timing);
  m.def("set_cpu_backward_profile_enabled", &BACKWARD::set_cpu_profile_enabled);
  m.def("get_cpu_backward_profile", []() {
    const BACKWARD::CpuRenderProfile p = BACKWARD::get_cpu_profile();
    py::dict result;
    result["d2h_ms"] = p.d2h_ms;
    result["host_init_ms"] = p.host_init_ms;
    result["render_cpu_ms"] = p.render_cpu_ms;
    result["h2d_ms"] = p.h2d_ms;
    result["total_ms"] = p.total_ms;
    result["buckets"] = p.buckets;
    result["buckets_skipped"] = p.buckets_skipped;
    result["splats"] = p.splats;
    result["gaussian_pixel_tests"] = p.gaussian_pixel_tests;
    result["skipped_invalid_pixel"] = p.skipped_invalid_pixel;
    result["skipped_last_contributor"] = p.skipped_last_contributor;
    result["pixel_checks_passed"] = p.pixel_checks_passed;
    result["power_evaluations"] = p.power_evaluations;
    result["skipped_power"] = p.skipped_power;
    result["exp_evaluations"] = p.exp_evaluations;
    result["skipped_alpha"] = p.skipped_alpha;
    result["gradient_evaluations"] = p.gradient_evaluations;
    return result;
  });
  m.def("mark_visible", &markVisible);
  m.def("adamUpdate", &adamUpdate);
}
