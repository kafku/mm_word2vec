#pragma once

#include <cmath>
#include "v.h"


namespace gu {
	using namespace v;

	// grad = d cos(x, y) / d x
	void grad_cos_wrt_vec(const Vector& x, const Vector& y, Vector& grad) {
		const float norm_x_inv = 1.0 / norm(x), norm_y_inv = 1.0 / norm(y);
		const float a = norm_x_inv * norm_y_inv; // 1.0 / (norm_x * norm_y)
		const float b = dot(x, y) * std::pow(norm_x_inv, 3) * norm_y_inv; // dot(x, y) / (norm_x^3 ^ norm_y)
		int m = x.size();

		// y / (norm_x * norm_y) - dot(x, y) x / (norm_x^3 * norm_y)
		float *gd = grad.data();
		const float *xd = x.data(), *yd = y.data();
		while (--m >= 0)
			*(gd++) = *(yd++) * a - *(xd++) * b;
	}

	// grad = d cons(y, Mx) / d x
	void grad_cos_wrt_mat(const Vector& x, const Vector& y, const LightMatrix& M, LightMatrix& grad) {
		const float norm_Mx_inv = 1.0 / sgemv_norm(M, x), norm_y_inv = 1.0 / norm(y);
		const float a = norm_Mx_inv * norm_y_inv; // 1.0 / (norm(Mx) * norm(y))
		const float b = dot_A(y, x, M) * std::pow(norm_Mx_inv, 3) * norm_y_inv; // y^T Mx / (norm(Mx)^3 * norm(y))

		// y x^T / (norm(Mx) * norm(y)) - y^T Mx Mxx^T / (norm(Mx)^3 * norm(y))
		float *gd = grad.data();
		const float *xd = x.data(), *yd = y.data();
		for (int row = 0; row < grad.n_rows; ++row) {
			for (int col = 0; col < grad.n_cols; ++col) {
				grad[row * grad.n_cols + col] = yd[row] * xd[col] * a - dot(M.row(row), x) * x[col] * b;
			}
		}
	}
} // end namespace: gu
