#pragma once

#include <cstddef>
#include <type_traits>
#include <array>

namespace my_utils {
	// define own integer_sequence
	template <size_t... N>
	struct integer_sequence
	{
		using value_type = size_t;
		static inline std::size_t size() {
			return (sizeof...(N));
		}
	};

	// define own make_index_sequence
	template<size_t N>
	class make_index_sequence
	{
	private:
		template <size_t M, size_t ...I>
		struct make_index_sequence_impl : public make_index_sequence_impl<M - 1 , I..., sizeof...(I)> {};

		template <size_t ...I>
		struct make_index_sequence_impl<0, I...>
		{
			using type = integer_sequence<I...>;
		};

	public:
		using type = typename make_index_sequence_impl<N>::type;
	};

	// define own make_array
	// NOTE: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=52892
	template <typename A, size_t... I>
	constexpr auto make_array_impl(A f(size_t), integer_sequence<I...>)
		-> std::array<A, sizeof...(I)>
	{
		return {{ f(I)... }};
	}

	template <size_t N, typename Function>
	constexpr auto make_array(Function f)
		-> std::array<typename std::result_of<Function(size_t)>::type, N>
	{
		return make_array_impl(f, typename make_index_sequence<N>::type());
	}
} // namespace my_utils
