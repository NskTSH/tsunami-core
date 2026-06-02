#pragma once
#include <iostream>
#include <stdexcept>
#include <vector>
#include <fstream>

#include "IBaseBathymetryParser.h"

class ASCParser : public IBaseBathymetryParser {
private:
    int ncols_;
    int nrows_;
    double xllcorner_;
    double yllcorner_;
    double cellsize_;
    double nodata_value_;
    std::vector<std::vector<double>> data_;

public:
    ASCParser() : ncols_(0), nrows_(0), xllcorner_(0), yllcorner_(0), cellsize_(0), nodata_value_(0) {}

    void Read(const std::string& filename) override {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Can't open file: " + filename);
        }

        std::string line;
        file >> line >> ncols_;
        file >> line >> nrows_;
        file >> line >> xllcorner_;
        file >> line >> yllcorner_;
        file >> line >> cellsize_;
        file >> line >> nodata_value_;

        data_.resize(nrows_, std::vector<double>(ncols_));
        for (int i = 0; i < nrows_; ++i) {
            for (int j = 0; j < ncols_; ++j) {
                file >> data_[i][j];
                data_[i][j] *= -1;
            }
        }

        file.close();
    }

    int GetCols() const override
    {
	    return ncols_;
    }

    int GetRows() const override
    {
	    return nrows_;
    }

    double GetXllCorner() const
    {
	    return xllcorner_;
    }

    double GetYllCorner() const
    {
	    return yllcorner_;
    }

    double GetCellSize() const
    {
	    return cellsize_;
    }

    double GetNoDataValue() const
    {
	    return nodata_value_;
    }

    double GetValue(int row, int col) const override {
        if (row < 0 || row >= nrows_ || col < 0 || col >= ncols_) {
            throw std::out_of_range("Index out of data range");
        }
        return data_[row][col];
    }

    // TODO: Is it necessary? Maybe need to add stream in args
    void PrintSummary() const {
        std::cout << "ASC File Summary:\n"
            << "Cols: " << ncols_ << ", Rows: " << nrows_ << "\n"
            << "XllCorner: " << xllcorner_ << ", YllCorner: " << yllcorner_ << "\n"
            << "CellSize: " << cellsize_ << ", NoDataValue: " << nodata_value_ << "\n"
            << "Sample Data (5x5):\n";

        for (int i = 0; i < std::min(5, nrows_); ++i) {
            for (int j = 0; j < std::min(5, ncols_); ++j) {
                std::cout << data_[i][j] << " ";
            }
            std::cout << "\n";
        }
    }
};
