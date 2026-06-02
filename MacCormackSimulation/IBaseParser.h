#pragma once
#include <string>

class IBaseParser {
public:
    virtual ~IBaseParser() = default;
    virtual void loadFromFile(const std::string& filename) = 0;
    virtual int getCols() const = 0;
    virtual int getRows() const = 0;
    virtual double getXllCorner() const = 0;
    virtual double getYllCorner() const = 0;
    virtual double getCellSize() const = 0;
    virtual double getNoDataValue() const = 0;
    virtual double getValue(int row, int col) const = 0;
};
