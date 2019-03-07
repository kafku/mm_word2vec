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
		static T call(const Vector& y_, const Vector& x_, const LightMatrix& M_) {
			const Eigen::Map<const VectorXT> y(y_.data(), y_.size());
			const Eigen::Map<const VectorXT> x(x_.data(), x_.size());
			const Eigen::Map<const MatrixXT> M(M_.data(), M_.n_rows, M_.n_cols);

			const MatrixXT Mx = M * x;
			const T norm_Mx_inv = 1.0 / Mx.squaredNorm();
			const T norm_y_inv = 1.0 / y.squaredNorm();

			return (y.transpose() * Mx)(0, 0) * norm_Mx_inv * norm_y_inv;
		}

		// grad += alpha * d cos(Mx, y) / d x
		static void grad_wrt_vec(const T alpha, const Vector& x_, const Vector& y_, const LightMatrix& M_, Vector& grad_) {
			const Eigen::Map<const VectorXT> y(y_.data(), y_.size());
			const Eigen::Map<const VectorXT> x(x_.data(), x_.size());
			const Eigen::Map<const MatrixXT> M(M_.data(), M_.n_rows, M_.n_cols);
			Eigen::Map<VectorXT> grad(grad_.data(), grad_.size());

			const MatrixXT Mx = M * x;
			const T norm_Mx_inv = 1.0 / Mx.squaredNorm();
			const T norm_y_inv = 1.0 / y.squaredNorm();
			const T a = norm_Mx_inv * norm_y_inv; // 1.0 / (norm(x) * norm(y))
			const T b = (y.transpose() * Mx)(0, 0) * std::pow(norm_Mx_inv, 3) * norm_y_inv; // dot(Mx, y) / (norm(Mx)^3 ^ norm(y))

			// alpha * (M^T y / (norm(Mx) * norm(y)) - dot(x, y) M^T Mx / (norm(Mx)^3 * norm(y)))
			grad += alpha * M.transpose() * (y * a - Mx * b);
		}

		// grad += alpha * d cos(y, Mx) / d M
		static void grad_wrt_mat(const T alpha, const Vector& x_, const Vector& y_, const LightMatrix& M_, LightMatrix& grad_) {
			const Eigen::Map<const Eigen::VectorX<T>> y(y_.data(), y_.size());
			const Eigen::Map<const Eigen::VectorX<T>> x(x_.data(), x_.size());
			const Eigen::Map<const Eigen::MatrixX<T>> M(M_.data(), M_.n_rows, M_.n_cols);
			Eigen::Map<Eigen::MatrixX<T>> grad(grad_.data(), grad_.n_rows, grad_.n_cols);

			const Eigen::MatrixX<T> Mx = M * x;
			const T norm_Mx_inv = 1.0 / Mx.squaredNorm();
			const T norm_y_inv = 1.0 / y.squaredNorm();
			const T a = norm_Mx_inv * norm_y_inv; // 1.0 / (norm(Mx) * norm(y))
			const T b = (y.transpose() * Mx)(0, 0) * std::pow(norm_Mx_inv, 3) * norm_y_inv; // y^T Mx / (norm(Mx)^3 * norm(y))

			// grad += alpha * (y x^T / (norm(Mx) * norm(y)) - y^T Mx Mxx^T / (norm(Mx)^3 * norm(y)))
			grad += alpha * (y * a - Mx * b) * x.transpose(); // FIXME
		}
	};
} // end namespace: gu
