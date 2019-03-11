#pragma once

#include <string>

template <typename T> struct Cvt;

template <> struct Cvt<std::string> {
	static const std::string& to_utf8(const std::string& s) { return s; }
	static const std::string& from_utf8(const std::string& s) { return s; }
};

#if defined(_LIBCPP_BEGIN_NAMESPACE_STD)
#include <codecvt>
template <> struct Cvt<std::u16string> {
	static std::string to_utf8(const std::u16string& in) {
	    std::wstring_convert<std::codecvt_utf8<char16_t>, char16_t> cv;
    	return cv.to_bytes(in.data());
	}

	static std::u16string from_utf8(const std::string& in) {
	    std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> cv;
    	return cv.from_bytes(in.data());
	}
};
#else // gcc has no <codecvt>
#include "utf8cpp/utf8.h"
template <> struct Cvt<std::u16string> {
	static std::string to_utf8(const std::u16string& in) {
		std::string out;
		utf8::utf16to8(in.begin(), in.end(), std::back_inserter(out));
		return out;
	}

	static std::u16string from_utf8(const std::string& in) {
		std::u16string out;
		utf8::utf8to16(in.begin(), in.end(), std::back_inserter(out));
		return out;
	}
};
#endif
