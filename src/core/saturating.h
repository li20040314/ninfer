#pragma once

#include <cstdint>
#include <limits>

namespace ninfer {

// Arithmetic for the cost model that would otherwise widen an intermediate to 128 bits, a type MSVC
// does not have. Two families share that one cause:
//
//   saturating_*            Keep the widest value the type can hold. Overflow is detected with
//                           division pre-checks rather than by widening the intermediates, and every
//                           operation is exact whenever the true result fits and returns the
//                           unsigned maximum otherwise, so the cost model reports the same numbers
//                           on both toolchains.
//   wide_multiply /         Keep the whole product. Saturating is not a substitute here: two
//   wide_compare            distinct large products would both collapse onto the unsigned maximum
//                           and compare equal, so a comparison that must stay exact carries the
//                           product as a high/low pair and orders the halves.

[[nodiscard]] constexpr std::uint64_t saturating_add(std::uint64_t left,
                                                     std::uint64_t right) noexcept {
    return right > std::numeric_limits<std::uint64_t>::max() - left
               ? std::numeric_limits<std::uint64_t>::max()
               : left + right;
}

[[nodiscard]] constexpr std::uint64_t saturating_multiply(std::uint64_t left,
                                                          std::uint64_t right) noexcept {
    if (left == 0 || right == 0) { return 0; }
    return left > std::numeric_limits<std::uint64_t>::max() / right
               ? std::numeric_limits<std::uint64_t>::max()
               : left * right;
}

// n*(n+1)/2, saturated. Halving before multiplying keeps the intermediate exact: pairing the even
// factor with the odd one produces the final value directly, so the only saturation left is the
// one the value itself would have caused.
[[nodiscard]] constexpr std::uint64_t saturating_triangular(std::uint64_t n) noexcept {
    if (n == std::numeric_limits<std::uint64_t>::max()) { return n; }
    return n % 2U == 0 ? saturating_multiply(n / 2U, n + 1U)
                       : saturating_multiply(n, (n + 1U) / 2U);
}

// An exact 64x64-bit product, split into its halves. The value is the same one
// `static_cast<unsigned __int128>(left) * right` would produce.
struct WideProduct {
    std::uint64_t high = 0;
    std::uint64_t low  = 0;
};

[[nodiscard]] constexpr WideProduct wide_multiply(std::uint64_t left,
                                                  std::uint64_t right) noexcept {
    constexpr std::uint64_t kHalf = 0xffff'ffffULL;
    const std::uint64_t left_low  = left & kHalf;
    const std::uint64_t left_high = left >> 32U;
    const std::uint64_t right_low = right & kHalf;
    const std::uint64_t right_high = right >> 32U;

    // Each partial product is below 2^64, so none of these four multiplications can overflow.
    const std::uint64_t low_low   = left_low * right_low;
    const std::uint64_t low_high  = left_low * right_high;
    const std::uint64_t high_low  = left_high * right_low;
    const std::uint64_t high_high = left_high * right_high;

    // The middle column adds three values below 2^32, so it stays below 2^34. Its low half becomes
    // the upper half of the result and the bits above it carry into the high word; the shift that
    // discards them is deliberate, not an overflow.
    const std::uint64_t middle = (low_low >> 32U) + (low_high & kHalf) + (high_low & kHalf);
    return WideProduct{(high_high + (low_high >> 32U)) + (high_low >> 32U) + (middle >> 32U),
                       (middle << 32U) | (low_low & kHalf)};
}

// Three-way comparison of two wide products: negative when `left` orders first, zero when equal.
[[nodiscard]] constexpr int wide_compare(WideProduct left, WideProduct right) noexcept {
    if (left.high != right.high) { return left.high < right.high ? -1 : 1; }
    if (left.low != right.low) { return left.low < right.low ? -1 : 1; }
    return 0;
}

} // namespace ninfer
