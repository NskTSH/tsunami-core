cmake_minimum_required(VERSION 3.20)
project(tsunami_core LANGUAGES CXX)

# ---------------------------------------------------------------------------
# tsunami-core — header-only I/O foundation for the NskTSH tsunami stack.
#
# Exposes the target `NskTSH::tsunami-io`, the shared source-level contract
# consumed by BOTH downstream repos:
#   - tsunami-simulator (the SWE solver / runner)
#   - tsunami-ui        (the Qt results viewer / runner)
#
# Public headers (parsers / formats / parameters / coordinate system / sources):
#   ASCParser.h, ZDataParser.h, DefaultParametersParser.h, EllipsoidalSource.h,
#   CoordSystem.h, ISource.h, IBaseBathymetryParser.h, IBaseParametersParser.h,
#   IBaseParser.h   (all header-only, self-contained — no solver dependency)
#
# This is a header-only SOURCE dependency, not a binary (DLL) contract.
# ---------------------------------------------------------------------------
add_library(tsunami-io INTERFACE)
add_library(NskTSH::tsunami-io ALIAS tsunami-io)

target_include_directories(tsunami-io INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(tsunami-io INTERFACE cxx_std_20)
