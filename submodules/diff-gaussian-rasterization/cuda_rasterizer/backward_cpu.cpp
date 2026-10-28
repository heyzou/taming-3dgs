/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 */

#include "backward_cpu.h"
#include "config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace
{
	constexpr int kChannels = NUM_CHAFFELS;
	constexpr int kBucketSize = 32;
	constexpr int kBlockSize = BLOCK_X * BLOCK_Y;

	inline std::size_t channel_pixel_index(int channel, int pixel_id, int image_size)
	{
		return static_cast<std::size_t>(channel) * image_size + pixel_id;
	}
}

void BACKWARD::render_cpu(
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
	float* dL_dcolors)
{
	const int image_size = W * H;
	const int horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const float ddelx_dx = 0.5f * W;
	const float ddely_dy = 0.5f * H;

	// State entering the current Gaussian for every pixel in a tile. Keeping
	// this state across the lane loop reproduces the CUDA warp shuffles in a
	// direct, sequential form.
	std::array<float, kBlockSize> transmittance{};
	std::array<float, kBlockSize> final_transmittance{};
	std::array<std::uint32_t, kBlockSize> last_contributor{};
	std::array<bool, kBlockSize> valid_pixel{};
	std::array<int, kBlockSize> pixel_ids{};
	std::array<std::array<float, kBlockSize>, kChannels> accumulated_remainder{};
	std::array<std::array<float, kBlockSize>, kChannels> pixel_gradients{};

	for (int global_bucket_idx = 0; global_bucket_idx < B; ++global_bucket_idx)
	{
		const std::uint32_t tile_id = bucket_to_tile[global_bucket_idx];
		if (tile_id >= static_cast<std::uint32_t>(num_tiles))
			throw std::runtime_error("CPU backward render received an invalid tile index");

		const std::uint32_t range_start = ranges[2 * tile_id];
		const std::uint32_t range_end = ranges[2 * tile_id + 1];
		const int num_splats_in_tile = static_cast<int>(range_end - range_start);
		const std::uint32_t buckets_before_tile =
			tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
		const int bucket_idx_in_tile =
			global_bucket_idx - static_cast<int>(buckets_before_tile);
		const int first_splat_in_bucket = bucket_idx_in_tile * kBucketSize;

		// This is the same whole-bucket early exit used by
		// PerGaussianRenderCUDA.
		if (first_splat_in_bucket >= static_cast<int>(max_contrib[tile_id]))
			continue;

		const int tile_x = static_cast<int>(tile_id) % horizontal_blocks;
		const int tile_y = static_cast<int>(tile_id) / horizontal_blocks;
		const int pixel_min_x = tile_x * BLOCK_X;
		const int pixel_min_y = tile_y * BLOCK_Y;

		for (int local_pixel = 0; local_pixel < kBlockSize; ++local_pixel)
		{
			const int pixel_x = pixel_min_x + local_pixel % BLOCK_X;
			const int pixel_y = pixel_min_y + local_pixel / BLOCK_X;
			const bool inside = pixel_x < W && pixel_y < H;
			valid_pixel[local_pixel] = inside;
			if (!inside)
				continue;

			const int pixel_id = W * pixel_y + pixel_x;
			pixel_ids[local_pixel] = pixel_id;
			transmittance[local_pixel] =
				sampled_T[static_cast<std::size_t>(global_bucket_idx) * kBlockSize + local_pixel];
			final_transmittance[local_pixel] = final_Ts[pixel_id];
			last_contributor[local_pixel] = n_contrib[pixel_id];

			for (int channel = 0; channel < kChannels; ++channel)
			{
				const std::size_t sample_index =
					static_cast<std::size_t>(global_bucket_idx) * kBlockSize * kChannels +
					channel * kBlockSize + local_pixel;
				const std::size_t pixel_index =
					channel_pixel_index(channel, pixel_id, image_size);
				accumulated_remainder[channel][local_pixel] =
					-pixel_colors[pixel_index] + sampled_ar[sample_index];
				pixel_gradients[channel][local_pixel] = dL_dpixels[pixel_index];
			}
		}
		for (int lane = 0; lane < kBucketSize; ++lane)
		{
			const int splat_idx_in_tile = first_splat_in_bucket + lane;
			if (splat_idx_in_tile >= num_splats_in_tile)
				break;

			const std::uint32_t gaussian_idx = point_list[range_start + splat_idx_in_tile];
			if (gaussian_idx >= static_cast<std::uint32_t>(P))
				throw std::runtime_error("CPU backward render received an invalid Gaussian index");

			const float mean_x = means2D[2 * gaussian_idx];
			const float mean_y = means2D[2 * gaussian_idx + 1];
			const float conic_x = conic_opacity[4 * gaussian_idx];
			const float conic_y = conic_opacity[4 * gaussian_idx + 1];
			const float conic_z = conic_opacity[4 * gaussian_idx + 2];
			const float opacity = conic_opacity[4 * gaussian_idx + 3];

			float register_mean_x = 0.0f;
			float register_mean_y = 0.0f;
			float register_conic_x = 0.0f;
			float register_conic_y = 0.0f;
			float register_conic_w = 0.0f;
			float register_opacity = 0.0f;
			std::array<float, kChannels> register_colors{};

			for (int local_pixel = 0; local_pixel < kBlockSize; ++local_pixel)
			{
				if (!valid_pixel[local_pixel])
					continue;
				if (splat_idx_in_tile >= static_cast<int>(last_contributor[local_pixel]))
					continue;

				const int pixel_id = pixel_ids[local_pixel];
				const int pixel_x = pixel_id % W;
				const int pixel_y = pixel_id / W;
				const float dx = mean_x - static_cast<float>(pixel_x);
				const float dy = mean_y - static_cast<float>(pixel_y);
				const float power =
					-0.5f * (conic_x * dx * dx + conic_z * dy * dy) - conic_y * dx * dy;
				if (power > 0.0f)
					continue;

				const float gaussian = std::exp(power);
				const float alpha = std::min(0.99f, opacity * gaussian);
				if (alpha < 1.0f / 255.0f)
					continue;

				float& T = transmittance[local_pixel];
				const float dchannel_dcolor = alpha * T;
				const float one_minus_alpha_reciprocal = 1.0f / (1.0f - alpha);
				float dL_dalpha = 0.0f;

				for (int channel = 0; channel < kChannels; ++channel)
				{
					const float color = colors[gaussian_idx * kChannels + channel];
					float& ar = accumulated_remainder[channel][local_pixel];
					ar += dchannel_dcolor * color;
					const float dL_dchannel = pixel_gradients[channel][local_pixel];
					register_colors[channel] += dchannel_dcolor * dL_dchannel;
					dL_dalpha +=
						(color * T + one_minus_alpha_reciprocal * ar) * dL_dchannel;
				}

				float bg_dot_dpixel = 0.0f;
				for (int channel = 0; channel < kChannels; ++channel)
					bg_dot_dpixel += bg_color[channel] * pixel_gradients[channel][local_pixel];
				dL_dalpha +=
					(-final_transmittance[local_pixel] * one_minus_alpha_reciprocal) *
					bg_dot_dpixel;
				T *= 1.0f - alpha;

				const float dL_dG = opacity * dL_dalpha;
				const float gdx = gaussian * dx;
				const float gdy = gaussian * dy;
				const float dG_ddelx = -gdx * conic_x - gdy * conic_y;
				const float dG_ddely = -gdy * conic_z - gdx * conic_y;

				register_mean_x += dL_dG * dG_ddelx * ddelx_dx;
				register_mean_y += dL_dG * dG_ddely * ddely_dy;
				register_conic_x += -0.5f * gdx * dx * dL_dG;
				register_conic_y += -0.5f * gdx * dy * dL_dG;
				register_conic_w += -0.5f * gdy * dy * dL_dG;
				register_opacity += gaussian * dL_dalpha;
			}

			// A Gaussian may occur in several tiles. Sequential execution makes
			// normal additions sufficient; CUDA atomics are not needed here.
			dL_dmean2D[3 * gaussian_idx] += register_mean_x;
			dL_dmean2D[3 * gaussian_idx + 1] += register_mean_y;
			dL_dconic2D[4 * gaussian_idx] += register_conic_x;
			dL_dconic2D[4 * gaussian_idx + 1] += register_conic_y;
			dL_dconic2D[4 * gaussian_idx + 3] += register_conic_w;
			dL_dopacity[gaussian_idx] += register_opacity;
			for (int channel = 0; channel < kChannels; ++channel)
				dL_dcolors[gaussian_idx * kChannels + channel] += register_colors[channel];
		}
	}
}
