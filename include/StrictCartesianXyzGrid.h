#pragma once

#include <bit>
#include <charconv>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <istream>
#include <limits>
#include <optional>
#include <stop_token>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace nsktsh::io {

static_assert(sizeof(double) == 8U && std::numeric_limits<double>::is_iec559 &&
    std::numeric_limits<double>::radix == 2 &&
    std::numeric_limits<double>::digits == 53 &&
    std::numeric_limits<double>::max_exponent == 1024,
    "strict Cartesian XYZ requires IEEE 754 binary64 double");

inline constexpr std::uint32_t kStrictCartesianXyzGridReaderVersion = 1U;

struct StrictCartesianXyzGridReadLimits {
    std::uintmax_t maximum_file_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t maximum_cells{16ULL * 1024ULL * 1024ULL};
};

// Authority supplied independently by the caller, never inferred from XYZ
// values. Coordinates are metric cell centres; origin values may be zero or
// negative. Anisotropic positive x/y spacing is intentional.
struct StrictCartesianXyzGridRegistration {
    std::uint64_t columns{0U};
    std::uint64_t rows{0U};
    double first_cell_center_x{0.0};
    double first_cell_center_y{0.0};
    double cell_size_x{0.0};
    double cell_size_y{0.0};
};

struct StrictCartesianXyzGrid {
    std::uint32_t reader_version{kStrictCartesianXyzGridReaderVersion};
    std::uint64_t columns{0U};
    std::uint64_t rows{0U};
    double first_cell_center_x{0.0};
    double first_cell_center_y{0.0};
    double cell_size_x{0.0};
    double cell_size_y{0.0};
    // Row-major, x fastest, starting at the southern row. The reader verifies
    // this order; it does not sort, flip, project, regrid or change z's sign.
    std::vector<double> south_to_north_values;
};

struct StrictCartesianXyzGridReadResult {
    std::optional<StrictCartesianXyzGrid> grid;
    std::string code;
    std::string message;

    [[nodiscard]] bool Succeeded() const noexcept {
        return grid.has_value() && code == "cartesian_xyz_grid_read";
    }
};

namespace cartesian_xyz_detail {

inline bool Finite(const double value) noexcept {
    return (std::bit_cast<std::uint64_t>(value) & 0x7fffffffffffffffULL) <
        0x7ff0000000000000ULL;
}

// Integer comparison does not let DAZ collapse a parsed subnormal coordinate
// to zero. Both IEEE signed zeros denote the same registered coordinate.
inline bool SameCoordinate(const double left, const double right) noexcept {
    const auto a = std::bit_cast<std::uint64_t>(left);
    const auto b = std::bit_cast<std::uint64_t>(right);
    return a == b || ((a | b) & 0x7fffffffffffffffULL) == 0U;
}

// std::fegetround alone need not observe a processor's separate SSE rounding
// control, DAZ or FTZ. Probe the operations this reader requires without
// consuming input or leaking exception flags/control changes to its caller.
// This is a local arithmetic capability check, not a global FP certification.
inline bool FloatingPointEnvironmentSupported() noexcept {
    if (std::fegetround() != FE_TONEAREST) return false;
    std::fenv_t saved{};
    if (std::fegetenv(&saved) != 0) return false;
    std::fenv_t held{};
    if (std::feholdexcept(&held) != 0) {
        (void)std::fesetenv(&saved);
        return false;
    }

    volatile double minimum_normal = std::bit_cast<double>(0x0010000000000000ULL);
    volatile double half = 0.5;
    volatile double half_minimum_normal = minimum_normal * half;
    volatile double denormal = std::bit_cast<double>(std::uint64_t{1U});
    volatile double two_denormals = denormal + denormal;
    volatile double one = 1.0;
    volatile double half_ulp = std::bit_cast<double>(0x3ca0000000000000ULL);
    volatile double three_quarters_ulp = std::bit_cast<double>(0x3ca8000000000000ULL);
    volatile double tie = one + half_ulp;
    volatile double above_tie = one + three_quarters_ulp;
    const bool supported = std::fegetround() == FE_TONEAREST &&
        std::bit_cast<std::uint64_t>(static_cast<double>(half_minimum_normal)) == 0x0008000000000000ULL &&
        std::bit_cast<std::uint64_t>(static_cast<double>(two_denormals)) == 2U &&
        std::bit_cast<std::uint64_t>(static_cast<double>(tie)) == 0x3ff0000000000000ULL &&
        std::bit_cast<std::uint64_t>(static_cast<double>(above_tie)) == 0x3ff0000000000001ULL;
    const bool restored = std::fesetenv(&saved) == 0;
    return supported && restored;
}

// Preserve the versioned binary64 multiplication followed by binary64 addition.
// The observable intermediate prohibits contraction into an implicit FMA.
inline double AffineCoordinate(const double first, const std::uint64_t index,
                               const double step) noexcept {
    const volatile double offset = static_cast<double>(index) * step;
    return first + offset;
}

inline StrictCartesianXyzGridReadResult Failure(
    std::string code, std::string message) {
    return {std::nullopt, std::move(code), std::move(message)};
}

inline bool IsAsciiWhitespace(const int value) noexcept {
    return value == ' ' || value == '\t' || value == '\r' || value == '\n' ||
        value == '\v' || value == '\f';
}

inline bool ParseFiniteDouble(const std::string& token, double& value) noexcept {
    if (token.empty()) return false;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(),
        value, std::chars_format::general);
    return parsed.ec == std::errc{} &&
        parsed.ptr == token.data() + token.size() && Finite(value);
}

enum class TokenResult { Value, End, TooLong, SizeMismatch, ReadFailed, Cancelled };

// The declared byte count includes every whitespace byte. The one EOF peek at
// the declared boundary proves that a shorter declaration cannot hide suffix
// data; that extra byte is never consumed. Tokens never grow without a bound.
class Tokens final {
public:
    Tokens(std::istream& input, const std::uintmax_t bytes,
           const std::stop_token cancellation)
        : input_(input), bytes_(bytes), cancellation_(cancellation) {}

    TokenResult Next(std::string& token) {
        token.clear();
        for (;;) {
            const int value = Read();
            if (status_ != TokenResult::Value) return status_;
            if (IsAsciiWhitespace(value)) continue;
            token.push_back(static_cast<char>(value));
            break;
        }
        for (;;) {
            const int value = Read();
            if (status_ == TokenResult::End) return TokenResult::Value;
            if (status_ != TokenResult::Value) return status_;
            if (IsAsciiWhitespace(value)) return TokenResult::Value;
            if (token.size() == kMaximumTokenBytes) return TokenResult::TooLong;
            token.push_back(static_cast<char>(value));
        }
    }

private:
    static constexpr std::size_t kMaximumTokenBytes = 128U;
    std::istream& input_;
    std::uintmax_t bytes_;
    std::stop_token cancellation_;
    std::uintmax_t consumed_{0U};
    TokenResult status_{TokenResult::Value};

    int Read() {
        if (status_ != TokenResult::Value) return std::char_traits<char>::eof();
        if ((consumed_ & 4095U) == 0U && cancellation_.stop_requested()) {
            status_ = TokenResult::Cancelled;
            return std::char_traits<char>::eof();
        }
        if (consumed_ == bytes_) {
            const int next = input_.peek();
            status_ = next != std::char_traits<char>::eof()
                ? TokenResult::SizeMismatch
                : input_.eof() && !input_.bad()
                    ? TokenResult::End : TokenResult::ReadFailed;
            return std::char_traits<char>::eof();
        }
        const int value = input_.get();
        if (value == std::char_traits<char>::eof()) {
            status_ = input_.eof() && !input_.bad()
                ? TokenResult::SizeMismatch : TokenResult::ReadFailed;
            return value;
        }
        ++consumed_;
        return value;
    }
};

inline StrictCartesianXyzGridReadResult TokenFailure(const TokenResult result) {
    switch (result) {
        case TokenResult::End:
            return Failure("cartesian_xyz_cell_missing",
                "XYZ payload ended before the registered cell count");
        case TokenResult::TooLong:
            return Failure("cartesian_xyz_token_too_long",
                "XYZ numeric token exceeds the strict byte limit");
        case TokenResult::SizeMismatch:
            return Failure("cartesian_xyz_byte_count_mismatch",
                "XYZ stream length differs from the admitted byte count");
        case TokenResult::Cancelled:
            return Failure("cartesian_xyz_read_cancelled",
                "strict Cartesian XYZ read was cancelled");
        default:
            return Failure("cartesian_xyz_read_failed",
                "XYZ input did not reach a clean end of file");
    }
}

}  // namespace cartesian_xyz_detail

// Headerless whitespace-separated x/y/z triples, exactly columns*rows, in
// registered south-first row-major order. Binary64 coordinate equality is
// exact: no tolerance can relabel angular data or a differently registered
// grid. A verified immutable source stream is preferred to path reopening.
// Peak owned storage is O(cells*sizeof(double) + 128-byte token); the input text
// is never retained. Runtime adapters may own additional verified-source leases.
// Affine coordinates use first + double(index)*step, with distinct binary64
// multiplication and addition under round-to-nearest and gradual underflow.
// A scoped, state-preserving arithmetic probe refuses unsupported environments
// before reading input rather than changing the registered expression.
[[nodiscard]] inline StrictCartesianXyzGridReadResult ReadStrictCartesianXyzGrid(
    std::istream& input,
    const std::uintmax_t input_bytes,
    const StrictCartesianXyzGridRegistration& expected,
    const StrictCartesianXyzGridReadLimits limits = {},
    const std::stop_token cancellation = {}) noexcept {
    using namespace cartesian_xyz_detail;
    try {
        if (!FloatingPointEnvironmentSupported()) {
            return Failure("cartesian_xyz_floating_point_environment_unsupported",
                "strict Cartesian XYZ requires round-to-nearest binary64 arithmetic with gradual underflow");
        }
        if (limits.maximum_file_bytes == 0U || limits.maximum_cells == 0U) {
            return Failure("cartesian_xyz_limit_invalid",
                "strict Cartesian XYZ read limits must be positive");
        }
        if (cancellation.stop_requested()) {
            return TokenFailure(TokenResult::Cancelled);
        }
        if (!input) {
            return Failure("cartesian_xyz_stream_invalid",
                "XYZ input stream must be readable");
        }
        if (input_bytes > limits.maximum_file_bytes) {
            return Failure("cartesian_xyz_file_too_large",
                "XYZ input exceeds the configured byte limit");
        }
        if (expected.columns == 0U || expected.rows == 0U ||
            expected.columns > 9007199254740992ULL ||
            expected.rows > 9007199254740992ULL ||
            expected.rows > (std::numeric_limits<std::uint64_t>::max)() /
                expected.columns) {
            return Failure("cartesian_xyz_shape_invalid",
                "XYZ registration requires a positive nonoverflowing grid shape");
        }
        const auto cells = expected.rows * expected.columns;
        if (cells > limits.maximum_cells ||
            cells > (std::numeric_limits<std::size_t>::max)() / sizeof(double) ||
            cells > std::vector<double>{}.max_size()) {
            return Failure("cartesian_xyz_cell_limit_exceeded",
                "XYZ registered shape exceeds the configured or addressable cell limit");
        }
        if (!Finite(expected.first_cell_center_x) ||
            !Finite(expected.first_cell_center_y) ||
            !Finite(expected.cell_size_x) || expected.cell_size_x <= 0.0 ||
            !Finite(expected.cell_size_y) || expected.cell_size_y <= 0.0) {
            return Failure("cartesian_xyz_registration_invalid",
                "XYZ registration requires finite origins and positive finite metric spacing");
        }
        const double last_x = AffineCoordinate(expected.first_cell_center_x,
            expected.columns - 1U, expected.cell_size_x);
        const double last_y = AffineCoordinate(expected.first_cell_center_y,
            expected.rows - 1U, expected.cell_size_y);
        if (!Finite(last_x) || !Finite(last_y) ||
            (expected.columns > 1U && !(last_x > expected.first_cell_center_x)) ||
            (expected.rows > 1U && !(last_y > expected.first_cell_center_y))) {
            return Failure("cartesian_xyz_registration_overflow",
                "XYZ affine registration is nonfinite or collapses an axis");
        }

        StrictCartesianXyzGrid grid;
        grid.columns = expected.columns;
        grid.rows = expected.rows;
        grid.first_cell_center_x = expected.first_cell_center_x;
        grid.first_cell_center_y = expected.first_cell_center_y;
        grid.cell_size_x = expected.cell_size_x;
        grid.cell_size_y = expected.cell_size_y;
        grid.south_to_north_values.reserve(static_cast<std::size_t>(cells));
        Tokens tokens(input, input_bytes, cancellation);
        std::string token;
        token.reserve(128U);
        double previous_y = expected.first_cell_center_y;
        for (std::uint64_t row = 0U; row < expected.rows; ++row) {
            const double expected_y = AffineCoordinate(expected.first_cell_center_y,
                row, expected.cell_size_y);
            if (!Finite(expected_y) || (row != 0U && !(expected_y > previous_y))) {
                return Failure("cartesian_xyz_registration_overflow",
                    "XYZ affine northing coordinates are nonfinite or not strictly increasing");
            }
            previous_y = expected_y;
            double previous_x = expected.first_cell_center_x;
            for (std::uint64_t column = 0U; column < expected.columns; ++column) {
                const double expected_x = AffineCoordinate(expected.first_cell_center_x,
                    column, expected.cell_size_x);
                if (!Finite(expected_x) || (column != 0U && !(expected_x > previous_x))) {
                    return Failure("cartesian_xyz_registration_overflow",
                        "XYZ affine easting coordinates are nonfinite or not strictly increasing");
                }
                previous_x = expected_x;
                double values[3]{};
                for (unsigned component = 0U; component != 3U; ++component) {
                    const auto status = tokens.Next(token);
                    if (status != TokenResult::Value) return TokenFailure(status);
                    if (!ParseFiniteDouble(token, values[component])) {
                        return Failure("cartesian_xyz_cell_invalid",
                            "XYZ payload contains a malformed or nonfinite coordinate or value");
                    }
                }
                if (!SameCoordinate(values[0], expected_x) || !SameCoordinate(values[1], expected_y)) {
                    return Failure("cartesian_xyz_registration_mismatch",
                        "XYZ coordinates do not exactly match the registered metric cell centres and row order");
                }
                grid.south_to_north_values.push_back(values[2]);
            }
        }
        const auto end = tokens.Next(token);
        if (end == TokenResult::Value) {
            return Failure("cartesian_xyz_trailing_data",
                "XYZ payload contains data after the registered cell count");
        }
        if (end != TokenResult::End) return TokenFailure(end);
        return {std::move(grid), "cartesian_xyz_grid_read",
            "strict metric Cartesian XYZ parsed with exact registered south-first cell centres"};
    } catch (const std::exception&) {
        return Failure("cartesian_xyz_read_exception",
            "strict Cartesian XYZ read failed while parsing or allocating bounded input");
    } catch (...) {
        return Failure("cartesian_xyz_read_exception",
            "strict Cartesian XYZ read failed with an unknown exception");
    }
}

}  // namespace nsktsh::io
