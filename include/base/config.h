/**
 * @file config.h
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

#include <deal.II/base/config.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>


/*!
 * @namespace LA
 *
 * @brief Namespace for TrilinosWrappers.
 */
namespace LA
{
#if defined(DEAL_II_WITH_TRILINOS)
#  define USE_TRILINOS_LA
  using namespace dealii::TrilinosWrappers;
#else
#  error DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#define DYCOREPLANET_OPEN_NAMESPACE \
  namespace DyCorePlanet            \
  {
#define DYCOREPLANET_CLOSE_NAMESPACE }


/*!
 * @namespace DyCorePlanet
 *
 * Namespace containing functions and classes to simulate a 3D aqua-planet. We
 * provide the dealii namespace inside DyCorePlanet since it provides the base
 * functionality for simulations.
 */
namespace DyCorePlanet
{
  // Provide dealii
  using namespace dealii;
} // namespace DyCorePlanet



/*!
 * Numeric epsilon for types::REAL. Interface to C++ STL.
 */
static const double double_eps = std::numeric_limits<double>::epsilon();

/*!
 * Numeric minimum for types::REAL. Interface to C++ STL.
 */
static const double double_min = std::numeric_limits<double>::min();

/*!
 * Numeric maximum for types::REAL. Interface to C++ STL.
 */
static const double double_max = std::numeric_limits<double>::max();

/*!
 * Function to compare two non-integer values. Return a bool. Interface to C++
 * STL.
 */
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type
is_approx(T x, T y, int ulp = 2)
{
  /* Machine epsilon has to be scaled to the magnitude
   * of the values used and multiplied by the desired precision
   * in ULPs (units in the last place) */
  return std::abs(x - y) <=
           std::numeric_limits<T>::epsilon() * std::abs(x + y) * ulp
         /* unless the result is subnormal. */
         || std::abs(x - y) < std::numeric_limits<T>::min();
}
