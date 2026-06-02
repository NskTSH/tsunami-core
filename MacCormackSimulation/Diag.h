#pragma once
#include <fstream>
#include <string>

inline std::ofstream& _diag_stream() {
    static std::ofstream s("_diag_stages.txt", std::ios::app);
    return s;
}

#define DIAG_EXT(msg) do { _diag_stream() << msg << std::endl; _diag_stream().flush(); } while(0)
