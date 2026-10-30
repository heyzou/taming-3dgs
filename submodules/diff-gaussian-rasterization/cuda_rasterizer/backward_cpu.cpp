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
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{
	constexpr int kChannels = NUM_CHAFFELS;
	constexpr int kBucketSize = 32;
	constexpr int kBlockSize = BLOCK_X * BLOCK_Y;
	bool cpu_profile_is_enabled = false;
	BACKWARD::CpuRenderProfile last_cpu_profile;

	inline std::uint64_t read_profile_counter()
	{
#if defined(__aarch64__)
		std::uint64_t value;
		asm volatile("isb; mrs %0, cntvct_el0" : "=r"(value) : : "memory");
		return value;
#else
		return static_cast<std::uint64_t>(
			std::chrono::steady_clock::now().time_since_epoch().count());
#endif
	}

	inline std::uint64_t profile_counter_frequency()
	{
#if defined(__aarch64__)
		std::uint64_t value;
		asm volatile("mrs %0, cntfrq_el0" : "=r"(value));
		return value;
#else
		using Period = std::chrono::steady_clock::period;
		return static_cast<std::uint64_t>(Period::den / Period::num);
#endif
	}

	class ScopedSectionTimer
	{
	public:
		ScopedSectionTimer(
			BACKWARD::CpuRenderProfile* profile,
			BACKWARD::CpuRenderSection section)
			: profile_(profile), section_(static_cast<std::size_t>(section)),
			  start_(profile ? read_profile_counter() : 0)
		{
		}

		~ScopedSectionTimer()
		{
			if (!profile_)
				return;
			const std::uint64_t end = read_profile_counter();
			const std::uint64_t elapsed_ticks = end - start_;
			profile_->section_ticks[section_] += elapsed_ticks;
			++profile_->section_intervals[section_];
			if (elapsed_ticks == 0)
				++profile_->section_zero_tick_intervals[section_];
		}

		ScopedSectionTimer(const ScopedSectionTimer&) = delete;
		ScopedSectionTimer& operator=(const ScopedSectionTimer&) = delete;

	private:
		BACKWARD::CpuRenderProfile* profile_;
		std::size_t section_;
		std::uint64_t start_;
	};

	struct TimerOverhead
	{
		double interval_ticks;
		double full_ticks;
	};

	double median(std::vector<double> values)
	{
		std::sort(values.begin(), values.end());
		const std::size_t middle = values.size() / 2;
		if (values.size() % 2 == 0)
			return 0.5 * (values[middle - 1] + values[middle]);
		return values[middle];
	}

	TimerOverhead measure_profile_timer_overhead()
	{
		constexpr int kBatches = 20;
		constexpr int kIntervalsPerBatch = 100000;
		constexpr std::size_t kCalibrationSection =
			static_cast<std::size_t>(BACKWARD::CpuRenderSection::AlphaThreshold);
		std::vector<double> interval_means;
		std::vector<double> full_means;
		interval_means.reserve(kBatches);
		full_means.reserve(kBatches);

		// Warm the timer path and the profile storage before collecting batches.
		BACKWARD::CpuRenderProfile warmup_profile;
		for (int iteration = 0; iteration < 1000; ++iteration)
		{
			ScopedSectionTimer timer(
				&warmup_profile, BACKWARD::CpuRenderSection::AlphaThreshold);
		}

		for (int batch = 0; batch < kBatches; ++batch)
		{
			BACKWARD::CpuRenderProfile calibration_profile;
			const std::uint64_t timed_start = read_profile_counter();
			for (int iteration = 0; iteration < kIntervalsPerBatch; ++iteration)
			{
				ScopedSectionTimer timer(
					&calibration_profile, BACKWARD::CpuRenderSection::AlphaThreshold);
			}
			const std::uint64_t timed_end = read_profile_counter();

			// Measure an equally long loop without the timer. The empty asm keeps
			// the compiler from deleting the baseline loop.
			const std::uint64_t baseline_start = read_profile_counter();
			for (int iteration = 0; iteration < kIntervalsPerBatch; ++iteration)
				asm volatile("" : : "r"(iteration) : "memory");
			const std::uint64_t baseline_end = read_profile_counter();

			const double intervals = static_cast<double>(
				calibration_profile.section_intervals[kCalibrationSection]);
			interval_means.push_back(
				calibration_profile.section_ticks[kCalibrationSection] / intervals);

			const double timed_ticks = static_cast<double>(timed_end - timed_start);
			const double baseline_ticks = static_cast<double>(baseline_end - baseline_start);
			full_means.push_back(
				std::max(0.0, timed_ticks - baseline_ticks) / kIntervalsPerBatch);
		}

		return TimerOverhead{median(interval_means), median(full_means)};
	}

	inline std::size_t channel_pixel_index(int channel, int pixel_id, int image_size)
	{
		return static_cast<std::size_t>(channel) * image_size + pixel_id;
	}
}

const char* BACKWARD::cpu_render_section_name(CpuRenderSection section)
{
	static constexpr std::array<const char*, kCpuRenderSectionCount> names = {
		"tile_pixel_preparation",
		"gaussian_preparation",
		"pixel_eligibility_check",
		"gaussian_exponent",
		"gaussian_weight",
		"alpha_calculation",
		"alpha_threshold",
		"color_gradient",
		"foreground_alpha_gradient",
		"background_alpha_gradient",
		"transmittance_update",
		"mean_gradient",
		"conic_opacity_gradient",
		"output_accumulation",
	};
	return names.at(static_cast<std::size_t>(section));
}

void BACKWARD::set_cpu_profile_enabled(bool enabled)
{
	cpu_profile_is_enabled = enabled;
	if (!enabled)
		last_cpu_profile = CpuRenderProfile{};
}

bool BACKWARD::cpu_profile_enabled()
{
	return cpu_profile_is_enabled;
}

BACKWARD::CpuRenderProfile BACKWARD::get_cpu_profile()
{
	return last_cpu_profile;
}

void BACKWARD::store_cpu_profile(const CpuRenderProfile& profile)
{
	last_cpu_profile = profile;
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
	float* dL_dcolors,
	CpuRenderProfile* profile)
{
	if (profile)
	{
		profile->timer_frequency_hz = profile_counter_frequency();
		const TimerOverhead timer_overhead = measure_profile_timer_overhead();
		profile->timer_overhead_ticks = timer_overhead.interval_ticks;
		profile->timer_full_overhead_ticks = timer_overhead.full_ticks;
	}

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

	// PROFILE_SECTION: Tile / pixel preparation
	for (int global_bucket_idx = 0; global_bucket_idx < B; ++global_bucket_idx)
	{
		if (profile)
			++profile->buckets;
		std::uint32_t range_start = 0;
		int num_splats_in_tile = 0;
		int first_splat_in_bucket = 0;
		{
			ScopedSectionTimer section_timer(profile, CpuRenderSection::TilePixelPreparation);
			const std::uint32_t tile_id = bucket_to_tile[global_bucket_idx];
			if (tile_id >= static_cast<std::uint32_t>(num_tiles))
				throw std::runtime_error("CPU backward render received an invalid tile index");

			range_start = ranges[2 * tile_id];
			const std::uint32_t range_end = ranges[2 * tile_id + 1];
			num_splats_in_tile = static_cast<int>(range_end - range_start);
			const std::uint32_t buckets_before_tile =
				tile_id == 0 ? 0 : per_tile_bucket_offset[tile_id - 1];
			const int bucket_idx_in_tile =
				global_bucket_idx - static_cast<int>(buckets_before_tile);
			first_splat_in_bucket = bucket_idx_in_tile * kBucketSize;

			// This is the same whole-bucket early exit used by
			// PerGaussianRenderCUDA.
			if (first_splat_in_bucket >= static_cast<int>(max_contrib[tile_id]))
			{
				if (profile)
					++profile->buckets_skipped;
				continue;
			}

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
		}
		for (int lane = 0; lane < kBucketSize; ++lane)
		{
			const int splat_idx_in_tile = first_splat_in_bucket + lane;
			if (splat_idx_in_tile >= num_splats_in_tile)
				break;

			const std::uint32_t gaussian_idx = point_list[range_start + splat_idx_in_tile];
			if (gaussian_idx >= static_cast<std::uint32_t>(P))
				throw std::runtime_error("CPU backward render received an invalid Gaussian index");
			if (profile)
			{
				++profile->splats;
				profile->gaussian_pixel_tests += kBlockSize;
			}

			// PROFILE_SECTION: Gaussian preparation
			float mean_x;
			float mean_y;
			float conic_x;
			float conic_y;
			float conic_z;
			float opacity;
			float register_mean_x;
			float register_mean_y;
			float register_conic_x;
			float register_conic_y;
			float register_conic_w;
			float register_opacity;
			std::array<float, kChannels> register_colors;
			{
				ScopedSectionTimer section_timer(profile, CpuRenderSection::GaussianPreparation);
				mean_x = means2D[2 * gaussian_idx];
				mean_y = means2D[2 * gaussian_idx + 1];
				conic_x = conic_opacity[4 * gaussian_idx];
				conic_y = conic_opacity[4 * gaussian_idx + 1];
				conic_z = conic_opacity[4 * gaussian_idx + 2];
				opacity = conic_opacity[4 * gaussian_idx + 3];
				register_mean_x = 0.0f;
				register_mean_y = 0.0f;
				register_conic_x = 0.0f;
				register_conic_y = 0.0f;
				register_conic_w = 0.0f;
				register_opacity = 0.0f;
				register_colors.fill(0.0f);
			}

			// PROFILE_SECTION: Pixel eligibility check
			for (int local_pixel = 0; local_pixel < kBlockSize; ++local_pixel)
			{
				bool invalid_pixel;
				bool beyond_last_contributor = false;
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::PixelEligibilityCheck);
					invalid_pixel = !valid_pixel[local_pixel];
					if (!invalid_pixel)
						beyond_last_contributor =
							splat_idx_in_tile >= static_cast<int>(last_contributor[local_pixel]);
				}
				if (invalid_pixel)
				{
					if (profile)
						++profile->skipped_invalid_pixel;
					continue;
				}
				if (beyond_last_contributor)
				{
					if (profile)
						++profile->skipped_last_contributor;
					continue;
				}
				if (profile)
					++profile->pixel_checks_passed;

				// PROFILE_SECTION: Gaussian exponent
				if (profile)
					++profile->power_evaluations;
				int pixel_id;
				float dx;
				float dy;
				float power;
				bool skip_power;
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::GaussianExponent);
					pixel_id = pixel_ids[local_pixel];
					const int pixel_x = pixel_id % W;
					const int pixel_y = pixel_id / W;
					dx = mean_x - static_cast<float>(pixel_x);
					dy = mean_y - static_cast<float>(pixel_y);
					power =
						-0.5f * (conic_x * dx * dx + conic_z * dy * dy) - conic_y * dx * dy;
					skip_power = power > 0.0f;
				}
				if (skip_power)
				{
					if (profile)
						++profile->skipped_power;
					continue;
				}

				// PROFILE_SECTION: Gaussian weight
				if (profile)
					++profile->exp_evaluations;
				float gaussian;
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::GaussianWeight);
					gaussian = std::exp(power);
				}

				// PROFILE_SECTION: Alpha calculation
				float alpha;
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::AlphaCalculation);
					alpha = std::min(0.99f, opacity * gaussian);
				}

				// PROFILE_SECTION: Alpha threshold
				bool skip_alpha;
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::AlphaThreshold);
					skip_alpha = alpha < 1.0f / 255.0f;
				}
				if (skip_alpha)
				{
					if (profile)
						++profile->skipped_alpha;
					continue;
				}
				if (profile)
					++profile->gradient_evaluations;

				// PROFILE_SECTION: Foreground alpha gradient
				float& T = transmittance[local_pixel];
				float dchannel_dcolor;
				float one_minus_alpha_reciprocal;
				float dL_dalpha;
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::ForegroundAlphaGradient);
					dchannel_dcolor = alpha * T;
					one_minus_alpha_reciprocal = 1.0f / (1.0f - alpha);
					dL_dalpha = 0.0f;
				}

				for (int channel = 0; channel < kChannels; ++channel)
				{
					float color;
					float& ar = accumulated_remainder[channel][local_pixel];
					float dL_dchannel;
					{
						ScopedSectionTimer section_timer(profile, CpuRenderSection::ForegroundAlphaGradient);
						color = colors[gaussian_idx * kChannels + channel];
						ar += dchannel_dcolor * color;
						dL_dchannel = pixel_gradients[channel][local_pixel];
					}
					// PROFILE_SECTION: Color gradient
					{
						ScopedSectionTimer section_timer(profile, CpuRenderSection::ColorGradient);
						register_colors[channel] += dchannel_dcolor * dL_dchannel;
					}
					// PROFILE_SECTION: Foreground alpha gradient
					{
						ScopedSectionTimer section_timer(profile, CpuRenderSection::ForegroundAlphaGradient);
						dL_dalpha +=
							(color * T + one_minus_alpha_reciprocal * ar) * dL_dchannel;
					}
				}

				// PROFILE_SECTION: Background alpha gradient
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::BackgroundAlphaGradient);
					float bg_dot_dpixel = 0.0f;
					for (int channel = 0; channel < kChannels; ++channel)
						bg_dot_dpixel += bg_color[channel] * pixel_gradients[channel][local_pixel];
					dL_dalpha +=
						(-final_transmittance[local_pixel] * one_minus_alpha_reciprocal) *
						bg_dot_dpixel;
				}
				// PROFILE_SECTION: Transmittance update
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::TransmittanceUpdate);
					T *= 1.0f - alpha;
				}

				// PROFILE_SECTION: Mean gradient
				float dL_dG;
				float gdx;
				float gdy;
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::MeanGradient);
					dL_dG = opacity * dL_dalpha;
					gdx = gaussian * dx;
					gdy = gaussian * dy;
					const float dG_ddelx = -gdx * conic_x - gdy * conic_y;
					const float dG_ddely = -gdy * conic_z - gdx * conic_y;

					register_mean_x += dL_dG * dG_ddelx * ddelx_dx;
					register_mean_y += dL_dG * dG_ddely * ddely_dy;
				}
				// PROFILE_SECTION: Conic / opacity gradient
				{
					ScopedSectionTimer section_timer(profile, CpuRenderSection::ConicOpacityGradient);
					register_conic_x += -0.5f * gdx * dx * dL_dG;
					register_conic_y += -0.5f * gdx * dy * dL_dG;
					register_conic_w += -0.5f * gdy * dy * dL_dG;
					register_opacity += gaussian * dL_dalpha;
				}
			}

			// PROFILE_SECTION: Output accumulation
			// A Gaussian may occur in several tiles. Sequential execution makes
			// normal additions sufficient; CUDA atomics are not needed here.
			{
				ScopedSectionTimer section_timer(profile, CpuRenderSection::OutputAccumulation);
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
}
