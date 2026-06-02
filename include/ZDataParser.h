#pragma once
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <utility>

#include "IBaseBathymetryParser.h"


class ZDataParser:public IBaseBathymetryParser
{
private:
    std::vector<std::vector<double>> data_;
    int ncols_;
    int nrows_;

public:
    // Считывает весь файл построчно, разбивает строки по whitespace
    void Read(const std::string& filename) override
    {
        std::ifstream file{ filename };
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        data_.clear();
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;            // пропускаем пустые строки

            std::istringstream iss{ line };
            std::vector<double> row;
            double value;
            while (iss >> value) {  // чтение по пробелам
                row.push_back(value);
            }

            if (!row.empty())
                data_.push_back(std::move(row));
        }
        ncols_ = data_.empty() ? 0
            : static_cast<int>(data_[0].size());
        nrows_ = static_cast<int>(data_.size());
    }

    // Если нет строк, возвращаем 0; иначе — размер первого ряда
    int GetCols() const override
    {
        return ncols_;
    }

    // Количество строк в файле
    int GetRows() const override
    {
        return nrows_;
    }

    // (Опционально) доступ к самим данным
    const std::vector<std::vector<double>>& GetData() const
    {
        return data_;
    }

    double GetValue(int row, int col) const override {
        if (row < 0 || row >= nrows_ || col < 0 || col >= ncols_) {
            throw std::out_of_range("Index out of data range");
        }
        return data_[row][col];
    }
};
