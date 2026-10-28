/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 */

#ifndef CUDA_RASTERIZER_BACKWARD_CPU_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_CPU_H_INCLUDED

#include <cstdint>

namespace BACKWARD
{
	// Sequential CPU implementation of the per-Gaussian rendering backward
	// pass. All pointers passed to this function must refer to host memory.
	void render_cpu(
		const std::uint32_t* ranges,
		const std::uint32_t* point_list,
		int P, int W, int H, int B, int num_tiles,
		const std::uint32_t* per_tile_bucket_offset,
		const std::uint32_t* bucket_to_tile,
		const float* sampled_T,
		const float* sampled_ar,
		const float* bg_color,
		const float* means2D,
		const float* conic_opacity,
		const float* colors,
		const float* final_Ts,
		const std::uint32_t* n_contrib,
		const std::uint32_t* max_contrib,
		const float* pixel_colors,
		const float* dL_dpixels,
		float* dL_dmean2D,
		float* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors);
}

#endif
