#pragma once
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <stdexcept>

#include "EllipsoidalSource.h"
#include "IBaseParametersParser.h"

class DefaultParametersParser : public IBaseParametersParser
{

public:
	DefaultParametersParser() = default;

    // v.1.2.0 (backward compatible with 1.1.x and 1.0.0)
    void Read(const std::string& filename) override
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        auto trim = [](std::string s) {
            const char* ws = " \t\r\n";
            const auto b = s.find_first_not_of(ws);
            if (b == std::string::npos) return std::string();
            const auto e = s.find_last_not_of(ws);
            return s.substr(b, e - b + 1);
        };

        auto split_ws = [](const std::string& s) {
            std::vector<std::string> out;
            std::istringstream iss(s);
            for (std::string tok; iss >> tok;) out.push_back(tok);
            return out;
        };

        // 1) Read non-empty, non-comment lines
        std::vector<std::string> lines;
        for (std::string line; std::getline(file, line);) {
            line = trim(line);
            if (line.empty()) continue;
            if (!line.empty() && line[0] == '#') continue;
            lines.push_back(line);
        }
        if (lines.empty()) {
            throw std::runtime_error("Parameters file is empty: " + filename);
        }

        // 2) Detect format: with/without format_version line
        const auto firstTokens = split_ws(lines[0]);
        const bool hasFormatVersion = (firstTokens.size() == 1);

        size_t headerIndex = 0;
        if (hasFormatVersion) {
            this->format_version = firstTokens[0];
            headerIndex = 1;
        }
        else {
            // v1.0.0 (no format_version line)
            this->format_version = "1.0.0";
            headerIndex = 0;
        }

        if (lines.size() <= headerIndex + 1) {
            throw std::runtime_error("Parameters file has too few lines: " + filename);
        }

        // 3) Second line (header)
        {
            const auto t = split_ws(lines[headerIndex]);
            if (t.size() != 5 && t.size() != 6) {
                throw std::runtime_error("Invalid header line (expected 5 or 6 fields): " + lines[headerIndex]);
            }

            this->delta_t = std::stod(t[0]);
            this->timesteps = std::stoi(t[1]);
            this->num_threads = std::stoi(t[2]);

            if (t.size() == 5) {
                // legacy: num_subgrids == num_threads
                this->num_subgrids = this->num_threads;
                this->boundary_exchange_type = std::stoi(t[3]);
                this->hat_boundary_exchange_type = std::stoi(t[4]);
            }
            else {
                // new: separate fields
                this->num_subgrids = std::stoi(t[3]);
                this->boundary_exchange_type = std::stoi(t[4]);
                this->hat_boundary_exchange_type = std::stoi(t[5]);
            }
        }

        // 4) Third line (grid steps + generation + sources)
        {
            const auto t = split_ws(lines[headerIndex + 1]);
            if (t.size() != 6) {
                throw std::runtime_error("Invalid grid line (expected 6 fields): " + lines[headerIndex + 1]);
            }
            this->delta_x_type = t[0];
            this->delta_x_step = std::stod(t[1]);
            this->delta_y_type = t[2];
            this->delta_y_step = std::stod(t[3]);

            if (this->format_version == "1.0.0") {
                // v1.0.0: number_of_sources and generation_type are swapped
                this->number_of_sources = std::stoi(t[4]);
                this->generation_type = t[5];
            }
            else {
                // v1.1.0+ (incl. 1.2.0): generation_type then number_of_sources
                this->generation_type = t[4];
                this->number_of_sources = std::stoi(t[5]);
            }
        }

        // 5) Fourth line (bathymetry + min depth) for v1.1.0+
        size_t sourcesStartIndex = headerIndex + 2;
        if (hasFormatVersion) {
            if (lines.size() <= headerIndex + 2) {
                throw std::runtime_error("Missing bathymetry line for format " + this->format_version);
            }
            const auto t = split_ws(lines[headerIndex + 2]);
            if (t.size() != 3) {
                throw std::runtime_error("Invalid bathymetry line (expected 3 fields): " + lines[headerIndex + 2]);
            }
            this->input_bathymetry_generation_type = std::stoi(t[0]);
            this->input_bathymetry_type = std::stoi(t[1]);
            this->minimum_depth_to_calculate = std::stod(t[2]);
            sourcesStartIndex = headerIndex + 3;
        }
        else {
            // v1.0.0: no bathymetry section
            this->input_bathymetry_generation_type = 0;
            this->input_bathymetry_type = 0;
            this->minimum_depth_to_calculate = 0.0;
            sourcesStartIndex = headerIndex + 2;
        }

        // 6) Fifth line (coord_system) for v1.2.0+ (optional)
        // Default: Cartesian
        this->coord_system = CoordSystem::Cartesian;
        size_t coordSystemIndex = sourcesStartIndex; // сдвигаем для проверки
        if (hasFormatVersion && this->format_version >= "1.2.0") {
            // Проверяем, есть ли строка с coord_system перед источниками
            if (lines.size() > sourcesStartIndex && 
                split_ws(lines[sourcesStartIndex]).size() == 1) {
                const auto t = split_ws(lines[sourcesStartIndex]);
                this->coord_system = StringToCoordSystem(t[0]);
                sourcesStartIndex++;
            }
        }

        // 7) Sources
        this->sources_.clear();
        if (this->number_of_sources < 0) {
            throw std::runtime_error("Invalid number_of_sources: " + std::to_string(this->number_of_sources));
        }
        for (int i = 0; i < this->number_of_sources; ++i) {
            const size_t lineIndex = sourcesStartIndex + static_cast<size_t>(i);
            if (lineIndex >= lines.size()) {
                throw std::runtime_error("Not enough source lines (expected " + std::to_string(this->number_of_sources) + ")");
            }
            std::istringstream in(lines[lineIndex]);
            std::string name, type;
            in >> name >> type;
            if (!in) {
                throw std::runtime_error("Error reading source #" + std::to_string(i));
            }
            this->sources_.push_back(createSource(name, type, in));
        }

        // Safety: if num_subgrids not set or invalid, fallback to num_threads
        if (this->num_subgrids <= 0) {
            this->num_subgrids = this->num_threads;
        }
    }

    std::unique_ptr<ISource> createSource(
        const std::string& name,
        const std::string& type,
        std::istream& in) const
    {
        if (type == "0") {
            double x, y, r1, r2, angle;
            in >> x >> y >> r1 >> r2 >> angle;
            return std::make_unique<EllipsoidalSource>(name, x, y, r1, r2, angle);
        }
        else if (type == "1") {
            throw std::runtime_error("Not implemented type: " + type);
            //return std::make_unique<PointSource>(name, x_center, y_center, angle);
        }
        else if (type == "2") {
            throw std::runtime_error("Not implemented type: " + type);
            //return std::make_unique<CylindricalSource>(name, x_center, y_center, angle);
        }
        else if (type == "3") {
            throw std::runtime_error("Not implemented type: " + type);
            //return std::make_unique<RectangularSource>(name, x_center, y_center, angle);
        }
        else {
            throw std::runtime_error("Unknown source type: " + type);
        }
    }

    // v.1.0.0 reader is kept implicitly inside Read() for backward compatibility.
};
