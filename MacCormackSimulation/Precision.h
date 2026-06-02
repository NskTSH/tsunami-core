// Precision.h - Numeric type for simulation state.
//
// Real is `float` when USE_FLOAT_PRECISION is defined at build time,
// otherwise `double`. All simulation storage and kernel arithmetic that should
// follow the build-time precision use `Real`.
//
// Scalar physics parameters (g, tau, delta_y) and grid-geometry callbacks
// (std::function<double(int,int)> delta_x) stay as `double` regardless —
// they're inputs from outside the simulator, not state.

#ifndef PRECISION_H
#define PRECISION_H

#ifdef USE_FLOAT_PRECISION
using Real = float;
#else
using Real = double;
#endif

#endif // PRECISION_H
