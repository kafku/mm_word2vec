#pragma once

#include <cstddef>
#include <cmath>
#include <vector>

typedef std::vector<float> Vector;

namespace v {
	struct LightVector: public std::pair<float *, float *>
	{
	  using Parent = std::pair<float *, float *>;
	  template <typename... Arg> LightVector(Arg&& ... arg): Parent(std::forward<Arg>(arg) ...) {}

	  float *data() { return first; }
	  const float *data() const { return first; }
	  size_t size() const { return std::distance(first, second); }
	  bool empty() const  { return first == second; }

	  float& operator[](size_t i) { return *(first + i); }
	  float operator[](size_t i) const { return *(first + i); }
	};

	template <class Vector1, class Vector2> inline float dot(const Vector1&x, const Vector2& y) {
		int m = x.size(); const float *xd = x.data(), *yd = y.data();
		float sum = 0.0;
		while (--m >= 0) sum += (*xd++) * (*yd++);
		return sum;
	}

	// saxpy: x = x + g * y; x = a * x + g * y
	template <typename Vector1, typename Vector2>
	inline void saxpy(Vector1& x, float g, const Vector2& y) {

		int m = x.size(); float *xd = x.data(); const float *yd = y.data();
		while (--m >= 0) (*xd++) += g * (*yd++);
	}

	template <typename Vector1, typename Vector2>
	inline void saxpy(float a, Vector1& x, float g, const Vector2& y) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data();
		while (--m >= 0) { (*xd) = a * (*xd) + g * (*yd); ++xd; ++yd; }
	}

	inline void saxpy2(Vector& x, float g, const Vector& y, float h, const Vector& z) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data(); const float *zd = z.data();
		while (--m >= 0) { (*xd++) +=  (g * (*yd++)  + h * (*zd++)); }
	}

	inline void scale(Vector& x, float g) {
		int m = x.size(); float *xd = x.data();
		while (--m >= 0) (*xd++) *= g;
	}

#if 0
	inline void addsub(Vector& x, const Vector& y, const Vector& z) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data(); const float *zd = z.data();
		while (--m >= 0) (*xd++) += ((*yd++) - (*zd++));
	}
#endif
	inline float norm(const Vector& x) {
		return std::sqrt(dot(x, x));
	}

	inline void unit(Vector& x) {
		const float len = norm(x);
		if (len == 0) return;

		int m = x.size(); float *xd = x.data();
		while (--m >= 0) (*xd++) /= len;
	}

	inline void add(Vector& x, const Vector& y) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data();
		while (--m >= 0) (*xd++) += (*yd++);
	}

	inline void sax2(Vector& x, const Vector& y) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data();
		while (--m >= 0) { (*xd++) += (*yd) * (*yd); yd++; }
	}

	inline void multiply(Vector& x, const Vector& y) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data();
		while (--m >= 0) { (*xd++) *= (*yd++); }
	}

	inline bool isfinite(const Vector& x) {
		for(auto const& i: x) { if (! std::isfinite(i)) return false; }
		return true;
	}

	// Note: expect row-major
	struct LightMatrix: LightVector
	{
		template <typename... Arg> LightMatrix(const int _n_rows, const int _n_cols, Arg&& ... arg)
			: LightVector(std::forward<Arg>(arg) ...), n_rows(_n_rows), n_cols(_n_cols) {}

		LightMatrix() = default;
		LightVector row(size_t row_idx) const {
			float* const row_start = first + row_idx * n_cols;
			float* const row_end = row_start + n_cols;
			return std::move(LightVector(row_start, row_end));
		}
		int n_rows, n_cols;
	};

	// x^T A y
	template <typename Vector1, typename Vector2>
	float dot_A(const Vector1& x, const Vector2& y, const LightMatrix& A) {
		float sum_all = 0.0;
		const float *xd = x.data();
		for (int row = 0; row < A.n_rows; ++row, ++xd) {
			int n_cols = A.n_cols;
			const float *Ad = A.row(row).data(), *yd = y.data();
			float sum = 0.0;
			while (--n_cols >= 0) sum += (*Ad++) * (*yd++);
			sum_all = *xd * sum;
		}
		return sum_all;
	}

	// norm(Ax)
	float sgemv_norm(const LightMatrix& A, const Vector& x) {
		float sum_all = 0.0;
		for (int row = 0; row < A.n_rows; ++row) {
			int n_cols = A.n_cols;
			const float *Ad = A.row(row).data(), *xd = x.data();
			float sum = 0.0;
			while (--n_cols >= 0) sum += (*Ad++) * (*xd++);
			sum_all += sum * sum;
		}
		return std::sqrt(sum_all);
	}

#if 0
	inline void sgemv(const LightMatrix& A, const LightVector& x, LightVector& y) {
		for (int row = 0; row < A.n_rows; ++row) {
			int n_cols = A.n_cols;
			const float *Ad = A.row(row).data(), *xd = x.data();
			float sum = 0.0;
			while (--n_cols >= 0) sum += (*Ad++) * (*xd++);
			y[row] = sum;
		}
	}

	inline void sger(float alpha, const LightVector& x, const LightVector& y, LightMatrix& A) {
		const float *xd = x.data();
		for (int row = 0; row < A.n_rows; ++row) {
			int n_cols = A.n_cols;
			float *Ad = A.row(row).data();
			const float *yd = y.data();
			while (--n_cols >= 0) {
				*Ad += *xd * *yd * alpha;
				++Ad, ++yd;
			}
			++xd;
		}
	}

	inline float dot(const Vector&x, const Vector& y) {
		int m = x.size(); const float *xd = x.data(), *yd = y.data();
		float sum = 0.0;
		while (--m >= 0) sum += (*xd++) * (*yd++);
		return sum;
	}

	inline void saxpy(Vector& x, float g, const Vector& y) {
		int m = x.size(); float *xd = x.data(); const float *yd = y.data();
		while (--m >= 0) (*xd++) += g * (*yd++);
	}

	inline void unit(Vector& x) {
		float len = ::sqrt(dot(x, x));
		if (len == 0) return;

		int m = x.size(); float *xd = x.data();
		while (--m >= 0) (*xd++) /= len;
	}
#endif
}

