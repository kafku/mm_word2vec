#pragma once

#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "v.h"


namespace gu {
	using namespace v;

	template <typename T>
	struct CosSim
	{
		using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;
		using MatrixXT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

		// cos(y, Mx)
		static T call(const Vector& y, const Vector& x, const LightMatrix& M) {
			Eigen::Map<const VectorXT>(y.data(), y.size());
			Eigen::Map<const VectorXT>(x.data(), x.size());
			Eigen::Map<const MatrixXT>(M.data(), M.n_rows, M.n_cols);

			const MatrixXT Mx = M * x;
			const T norm_Mx_inv = 1.0 / Mx.squaredNorm();
			const T norm_y_inv = 1.0 / y.squaredNorm();

			return (y.transpose() * Mx) * norm_Mx_inv * norm_y_inv;
		}

		// grad += alpha * d cos(Mx, y) / d x
		static void grad_wrt_vec(const T alpha, const Vector& x, const Vector& y, const LightMatrix& M, Vector& grad) {
			Eigen::Map<const VectorXT>(y.data(), y.size());
			Eigen::Map<const VectorXT>(x.data(), x.size());
			Eigen::Map<const MatrixXT>(M.data(), M.n_rows, M.n_cols);
			Eigen::Map<VectorXT>(grad.data(), grad.size());

			const MatrixXT Mx = M * x;
			const T norm_Mx_inv = 1.0 / Mx.squaredNorm();
			const T norm_y_inv = 1.0 / y.squaredNorm();
			const T a = norm_Mx_inv * norm_y_inv; // 1.0 / (norm(x) * norm(y))
			const T b = (y.transpose() * Mx) * std::pow(norm_Mx_inv, 3) * norm_y_inv; // dot(Mx, y) / (norm(Mx)^3 ^ norm(y))

			// alpha * (M^T y / (norm(Mx) * norm(y)) - dot(x, y) M^T Mx / (norm(Mx)^3 * norm(y)))
			grad += alpha * M.transpose() * (y * a - Mx * b);
		}

		// grad += alpha * d cos(y, Mx) / d M
		static void grad_wrt_mat(const T alpha, const Vector& x, const Vector& y, const LightMatrix& M, LightMatrix& grad) {
			Eigen::Map<const Eigen::VectorX<T>>(y.data(), y.size());
			Eigen::Map<const Eigen::VectorX<T>>(x.data(), x.size());
			Eigen::Map<const Eigen::MatrixX<T>>(M.data(), M.n_rows, M.n_cols);
			Eigen::Map<Eigen::MatrixX<T>>(grad.data(), grad.n_rows, grad.n_cols);

			const Eigen::MatrixX<T> Mx = M * x;
			const T norm_Mx_inv = 1.0 / Mx.squaredNorm();
			const T norm_y_inv = 1.0 / y.squaredNorm();
			const T a = norm_Mx_inv * norm_y_inv; // 1.0 / (norm(Mx) * norm(y))
			const T b = (y.transpose() * Mx) * std::pow(norm_Mx_inv, 3) * norm_y_inv; // y^T Mx / (norm(Mx)^3 * norm(y))

			// grad += alpha * (y x^T / (norm(Mx) * norm(y)) - y^T Mx Mxx^T / (norm(Mx)^3 * norm(y)))
			grad += alpha * (y * a - Mx * b) * x.transpose();
		}
	};
} // end namespace: gu
