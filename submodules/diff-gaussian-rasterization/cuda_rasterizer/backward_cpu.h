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

#include <array>
#include <cstddef>
#include <cstdint>

namespace BACKWARD
{
	enum class CpuRenderSection : std::size_t
	{
		TilePixelPreparation,
		GaussianPreparation,
		PixelEligibilityCheck,
		GaussianExponent,
		GaussianWeight,
		AlphaCalculation,
		AlphaThreshold,
		ColorGradient,
		ForegroundAlphaGradient,
		BackgroundAlphaGradient,
		TransmittanceUpdate,
		MeanGradient,
		ConicOpacityGradient,
		OutputAccumulation,
		Count
	};

	constexpr std::size_t kCpuRenderSectionCount =
		static_cast<std::size_t>(CpuRenderSection::Count);

	struct CpuRenderProfile
	{
		double d2h_ms = 0.0;
		double host_init_ms = 0.0;
		double render_cpu_ms = 0.0;
		double h2d_ms = 0.0;
		double total_ms = 0.0;
		std::uint64_t buckets = 0;
		std::uint64_t buckets_skipped = 0;
		std::uint64_t splats = 0;
		std::uint64_t gaussian_pixel_tests = 0;
		std::uint64_t skipped_invalid_pixel = 0;
		std::uint64_t skipped_last_contributor = 0;
		std::uint64_t pixel_checks_passed = 0;
		std::uint64_t power_evaluations = 0;
		std::uint64_t skipped_power = 0;
		std::uint64_t exp_evaluations = 0;
		std::uint64_t skipped_alpha = 0;
		std::uint64_t gradient_evaluations = 0;
		std::array<std::uint64_t, kCpuRenderSectionCount> section_ticks{};
		std::array<std::uint64_t, kCpuRenderSectionCount> section_intervals{};
		std::array<std::uint64_t, kCpuRenderSectionCount> section_zero_tick_intervals{};
		double timer_overhead_ticks = 0.0;
		double timer_full_overhead_ticks = 0.0;
		std::uint64_t timer_frequency_hz = 0;
	};

	const char* cpu_render_section_name(CpuRenderSection section);

	void set_cpu_profile_enabled(bool enabled);
	bool cpu_profile_enabled();
	CpuRenderProfile get_cpu_profile();
	void store_cpu_profile(const CpuRenderProfile& profile);

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
		float* dL_dcolors,
		CpuRenderProfile* profile = nullptr);
}

#endif
