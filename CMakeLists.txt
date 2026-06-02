cmake_minimum_required(VERSION 3.20)
project(tsunami_core LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Header-only I/O export target.
#
# This is the integration contract consumed by sibling repos (e.g. tsunami-ui),
# which add this repo as a git submodule and link `NskTSH::tsunami-io` instead
# of hardcoding include paths into ../MacCormackSimulation/...
#
# Public headers (parsers / formats / coordinate system):
#   ASCParser.h, ZDataParser.h, DefaultParametersParser.h,
#   EllipsoidalSource.h, CoordSystem.h, IBase*.h
#
# The full CPU+CUDA solver itself is built with the Visual Studio solution
# (MacCormackSimulation.sln); this CMakeLists only exposes the I/O headers.
# ---------------------------------------------------------------------------
add_library(tsunami-io INTERFACE)
add_library(NskTSH::tsunami-io ALIAS tsunami-io)

target_include_directories(tsunami-io INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/MacCormackSimulation>)

target_compile_features(tsunami-io INTERFACE cxx_std_20)
