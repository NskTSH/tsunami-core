#pragma once
#include <string>
#include <vector>
#include <memory>

#include "ISource.h"
#include "CoordSystem.h"


class IBaseParametersParser {
protected:
	// first line
	std::string format_version;
	// second line
    double delta_t;
    int timesteps;
    int num_threads;
	int num_subgrids;
    int boundary_exchange_type;
    int hat_boundary_exchange_type;
	// third line
    std::string delta_x_type;
    double delta_x_step;
    std::string delta_y_type;
    double delta_y_step;
	std::string generation_type;
    int number_of_sources;
	// fourth line
	int input_bathymetry_generation_type;
	int input_bathymetry_type;
	double minimum_depth_to_calculate;
	// fifth line (optional, default: Cartesian)
	CoordSystem coord_system = CoordSystem::Cartesian;

	std::vector<std::unique_ptr<ISource>> sources_;

public:
	virtual ~IBaseParametersParser() = default;
	virtual void Read(const std::string& filename) = 0;
	std::string GetFormatVersion() const { return format_version; }
	double GetDeltaT() const { return delta_t; }
	int GetTimesteps() const { return timesteps; }
	int GetNumThreads() const { return num_threads; }
	int GetNumSubgrids() const { return num_subgrids; }
	int GetBoundaryExchangeType() const { return boundary_exchange_type; }
	int GetHatBoundaryExchangeType() const { return hat_boundary_exchange_type; }
	std::string GetDeltaXType() const { return delta_x_type; }
	double GetDeltaXStep() const { return delta_x_step; }
	std::string GetDeltaYType() const { return delta_y_type; }
	double GetDeltaYStep() const { return delta_y_step; }
	int GetNumberOfSources() const { return number_of_sources; }
	std::string GetGenerationType() const { return generation_type; }
	const std::vector<std::unique_ptr<ISource>>& GetSources() const { return sources_; }
	double GetInputBathymetryGenerationType() const { return input_bathymetry_generation_type; }
	double GetInputBathymetryType() const { return input_bathymetry_type; }
	double GetMinDepthToCalc() const { return minimum_depth_to_calculate; }
	CoordSystem GetCoordSystem() const { return coord_system; }
};
