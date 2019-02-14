#pragma once

#include <cstddef>
#include <cmath>
#include "my_utils.h"


namespace cheap_math {
	static constexpr int max_size = 1000; // NOTE: you need -ftemplate-depth=xxx option
	static constexpr int max_exp = 6;

	bool is_out_of_exp_scale(const float f) {
		return (f <= -max_exp || f >= max_exp);
	}

	template <int MAX_SIZE, int MAX_EXP>
	constexpr float sigmoid_disc(size_t i) {
		return exp((i / float(MAX_SIZE) * 2 -1) * float(MAX_EXP)) / (exp((i / float(MAX_SIZE) * 2 -1) * float(MAX_EXP)) + 1);
	}

	float sigmoid(const float f) {
		constexpr auto table = my_utils::make_array<max_size>(sigmoid_disc<max_size, max_exp>);
		const int fi = int((f + max_exp) * (float(max_size) / float(max_exp) / 2.0));
		return table[fi];
	}
}
