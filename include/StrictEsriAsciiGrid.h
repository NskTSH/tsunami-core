#pragma once

#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <istream>
#include <limits>
#include <optional>
#include <stop_token>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace nsktsh::io {

inline constexpr std::uint32_t kStrictEsriAsciiGridReaderVersion = 1U;

struct StrictEsriAsciiGridReadLimits {
    std::uintmax_t maximum_file_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};
    std::uint64_t maximum_cells{16ULL * 1024ULL * 1024ULL};
};

struct StrictEsriAsciiGrid {
    std::uint32_t reader_version{kStrictEsriAsciiGridReaderVersion};
    std::uint64_t columns{0U};
    std::uint64_t rows{0U};
    double first_cell_center_x{0.0};
    double first_cell_center_y{0.0};
    double cell_size{0.0};
    double nodata_value{0.0};
    // ESRI ASCII stores the northern row first. The I/O contract preserves
    // that source order; solver adapters must explicitly choose their own row
    // orientation instead of relying on an implicit flip.
    std::vector<double> north_to_south_values;
};

struct StrictEsriAsciiGridReadResult {
    std::optional<StrictEsriAsciiGrid> grid;
    std::string code;
    std::string message;

    [[nodiscard]] bool Succeeded() const noexcept {
        return grid.has_value() && code == "esri_ascii_grid_read";
    }
};

namespace detail {

inline bool ParseUnsignedToken(
    const std::string& token,
    std::uint64_t& value) noexcept {
    if (token.empty() || token.front() == '-' || token.front() == '+') {
        return false;
    }
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), value, 10);
    return parsed.ec == std::errc{} &&
        parsed.ptr == token.data() + token.size();
}

inline bool ParseFiniteDoubleToken(
    const std::string& token,
    double& value) noexcept {
    if (token.empty()) return false;
    const auto parsed = std::from_chars(
        token.data(), token.data() + token.size(), value,
        std::chars_format::general);
    return parsed.ec == std::errc{} &&
        parsed.ptr == token.data() + token.size() && std::isfinite(value);
}

inline StrictEsriAsciiGridReadResult Failure(
    std::string code,
    std::string message) {
    return {std::nullopt, std::move(code), std::move(message)};
}

}  // namespace detail

[[nodiscard]] inline StrictEsriAsciiGridReadResult ReadStrictEsriAsciiGrid(
    std::istream& input,
    const std::uintmax_t input_bytes,
    const StrictEsriAsciiGridReadLimits limits = {},
    const std::stop_token cancellation = {}) noexcept {
    try {
        if (limits.maximum_file_bytes == 0U || limits.maximum_cells == 0U) {
            return detail::Failure(
                "esri_ascii_limit_invalid",
                "strict ESRI ASCII read limits must be positive");
        }
        if (cancellation.stop_requested()) {
            return detail::Failure(
                "esri_ascii_read_cancelled",
                "strict ESRI ASCII read was cancelled before parsing");
        }
        if (!input) {
            return detail::Failure(
                "esri_ascii_stream_invalid",
                "ESRI ASCII input stream must be readable");
        }
        if (input_bytes > limits.maximum_file_bytes) {
            return detail::Failure(
                "esri_ascii_file_too_large",
                "ESRI ASCII input exceeds the configured byte limit");
        }
        const auto read_pair = [&](std::string& key, std::string& value) {
            return static_cast<bool>(input >> key >> value);
        };

        StrictEsriAsciiGrid grid;
        std::string key;
        std::string value;
        if (!read_pair(key, value) || key != "ncols" ||
            !detail::ParseUnsignedToken(value, grid.columns) ||
            grid.columns == 0U) {
            return detail::Failure(
                "esri_ascii_ncols_invalid",
                "first header pair must be a positive ncols value");
        }
        if (!read_pair(key, value) || key != "nrows" ||
            !detail::ParseUnsignedToken(value, grid.rows) || grid.rows == 0U) {
            return detail::Failure(
                "esri_ascii_nrows_invalid",
                "second header pair must be a positive nrows value");
        }
        if (grid.rows > std::numeric_limits<std::uint64_t>::max() /
                grid.columns ||
            grid.rows * grid.columns >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()) ||
            grid.rows * grid.columns > limits.maximum_cells) {
            return detail::Failure(
                "esri_ascii_cell_limit_exceeded",
                "ESRI ASCII grid shape exceeds the configured cell limit");
        }

        bool corner_registration = false;
        double lower_left_x = 0.0;
        if (!read_pair(key, value) ||
            (key != "xllcorner" && key != "xllcenter") ||
            !detail::ParseFiniteDoubleToken(value, lower_left_x)) {
            return detail::Failure(
                "esri_ascii_x_registration_invalid",
                "third header pair must be finite xllcorner or xllcenter");
        }
        corner_registration = key == "xllcorner";

        double lower_left_y = 0.0;
        if (!read_pair(key, value) ||
            (key != "yllcorner" && key != "yllcenter") ||
            (corner_registration != (key == "yllcorner")) ||
            !detail::ParseFiniteDoubleToken(value, lower_left_y)) {
            return detail::Failure(
                "esri_ascii_y_registration_invalid",
                "fourth header pair must use the matching finite y registration");
        }
        if (!read_pair(key, value) || key != "cellsize" ||
            !detail::ParseFiniteDoubleToken(value, grid.cell_size) ||
            grid.cell_size <= 0.0) {
            return detail::Failure(
                "esri_ascii_cellsize_invalid",
                "fifth header pair must be a finite positive cellsize");
        }
        if (!read_pair(key, value) ||
            (key != "NODATA_value" && key != "nodata_value") ||
            !detail::ParseFiniteDoubleToken(value, grid.nodata_value)) {
            return detail::Failure(
                "esri_ascii_nodata_header_invalid",
                "sixth header pair must define a finite NODATA_value");
        }

        grid.first_cell_center_x = corner_registration
            ? lower_left_x + 0.5 * grid.cell_size
            : lower_left_x;
        grid.first_cell_center_y = corner_registration
            ? lower_left_y + 0.5 * grid.cell_size
            : lower_left_y;
        if (!std::isfinite(grid.first_cell_center_x) ||
            !std::isfinite(grid.first_cell_center_y)) {
            return detail::Failure(
                "esri_ascii_registration_overflow",
                "ESRI ASCII cell-center registration overflowed");
        }

        const auto cells = grid.rows * grid.columns;
        grid.north_to_south_values.reserve(
            static_cast<std::size_t>(cells));
        for (std::uint64_t index = 0U; index < cells; ++index) {
            if ((index & 4095U) == 0U && cancellation.stop_requested()) {
                return detail::Failure(
                    "esri_ascii_read_cancelled",
                    "strict ESRI ASCII read was cancelled while reading cells");
            }
            if (!(input >> value)) {
                return detail::Failure(
                    "esri_ascii_cell_missing",
                    "ESRI ASCII payload ended before the declared cell count");
            }
            double cell = 0.0;
            if (!detail::ParseFiniteDoubleToken(value, cell)) {
                return detail::Failure(
                    "esri_ascii_cell_invalid",
                    "ESRI ASCII payload contains a non-finite or malformed cell");
            }
            if (cell == grid.nodata_value) {
                return detail::Failure(
                    "esri_ascii_nodata_rejected",
                    "ESRI ASCII payload contains NODATA under the strict reader");
            }
            grid.north_to_south_values.push_back(cell);
        }
        if (input >> value) {
            return detail::Failure(
                "esri_ascii_trailing_data",
                "ESRI ASCII payload contains data after the declared grid");
        }
        if (!input.eof()) {
            return detail::Failure(
                "esri_ascii_read_failed",
                "ESRI ASCII input did not reach a clean end of file");
        }
        return {
            std::move(grid),
            "esri_ascii_grid_read",
            "strict ESRI ASCII grid parsed without implicit sign or row-order conversion"};
    } catch (const std::exception& error) {
        return detail::Failure(
            "esri_ascii_read_exception", error.what());
    } catch (...) {
        return detail::Failure(
            "esri_ascii_read_exception",
            "strict ESRI ASCII read failed with an unknown exception");
    }
}

// Filesystem compatibility entry point. Security-sensitive consumers should
// prefer the stream overload so verification and parsing can share one retained
// immutable byte source instead of reopening a mutable path.
[[nodiscard]] inline StrictEsriAsciiGridReadResult ReadStrictEsriAsciiGrid(
    const std::filesystem::path& path,
    const StrictEsriAsciiGridReadLimits limits = {},
    const std::stop_token cancellation = {}) noexcept {
    try {
        if (limits.maximum_file_bytes == 0U || limits.maximum_cells == 0U) {
            return detail::Failure(
                "esri_ascii_limit_invalid",
                "strict ESRI ASCII read limits must be positive");
        }
        if (cancellation.stop_requested()) {
            return detail::Failure(
                "esri_ascii_read_cancelled",
                "strict ESRI ASCII read was cancelled before opening");
        }

        std::error_code error;
        const auto status = std::filesystem::symlink_status(path, error);
        if (error || status.type() != std::filesystem::file_type::regular) {
            return detail::Failure(
                "esri_ascii_file_invalid",
                "ESRI ASCII input must be an existing regular non-symlink file");
        }
        const auto bytes = std::filesystem::file_size(path, error);
        if (error) {
            return detail::Failure(
                "esri_ascii_file_size_failed",
                "ESRI ASCII input size could not be inspected");
        }
        if (bytes > limits.maximum_file_bytes) {
            return detail::Failure(
                "esri_ascii_file_too_large",
                "ESRI ASCII input exceeds the configured byte limit");
        }

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            return detail::Failure(
                "esri_ascii_open_failed",
                "ESRI ASCII input could not be opened");
        }
        return ReadStrictEsriAsciiGrid(input, bytes, limits, cancellation);
    } catch (const std::exception& error) {
        return detail::Failure(
            "esri_ascii_read_exception", error.what());
    } catch (...) {
        return detail::Failure(
            "esri_ascii_read_exception",
            "strict ESRI ASCII read failed with an unknown exception");
    }
}

}  // namespace nsktsh::io
