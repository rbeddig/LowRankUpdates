/**
 * @file utilities.h
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <base/config.h>
#include <sys/stat.h>

#include <stdexcept>
#include <string>

DYCOREPLANET_OPEN_NAMESPACE

/*!
 * @namespace Tools
 *
 * @brief Namespace containing all tools that do not fit other more specific namespaces.
 */
namespace Tools
{
  /*!
   * @brief Creates a directory with a name from a given string
   *
   * @param dir_name
   */
  void
  create_data_directory(std::string dir_name);

} // namespace Tools

DYCOREPLANET_CLOSE_NAMESPACE
