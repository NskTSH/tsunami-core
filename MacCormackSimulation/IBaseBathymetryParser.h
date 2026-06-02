#pragma once
#include <string>

class IBaseBathymetryParser {
public:
    virtual ~IBaseBathymetryParser() = default;
    virtual void Read(const std::string& filename) = 0;
    virtual int GetCols() const = 0;
    virtual int GetRows() const = 0;
    virtual double GetValue(int row, int col) const = 0;
};
