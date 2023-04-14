/**
 * @file planet_geometry.h
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/numerics/data_out.h>


// C++ STL
#include <fstream>
#include <iostream>
#include <string>
#include <vector>


// DyCorePlanet headers
#include <base/config.h>
#include <model_data/boussinesq_model_data.h>


DYCOREPLANET_OPEN_NAMESPACE


/*!
 * Base class to handle mesh for aqua planet. The mesh
 * is a 3D spherical shell. Derived classes will implement different model
 * details.
 */
template <int dim>
class PlanetGeometry
{
public:
  /*!
   * Constructor of mesh handler for spherical shell.
   */
  PlanetGeometry(double inner_radius,
                 double outer_radius,
                 bool   cuboid_geometry = false);
  ~PlanetGeometry();

protected:
  void
  write_mesh_vtu();

  void
  refine_global(unsigned int n_refine);

  MPI_Comm mpi_communicator;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  parallel::distributed::Triangulation<dim> triangulation;

  Point<dim> center;
  double     inner_radius, outer_radius;

  double global_Omega_diameter;

  /*!
   * Sets the domain geometry to cuboid. All directions are periodic apart from
   * the z-direction. This is useful for debugging and later to restrict global
   * simulations to a full 3D column.
   */
  bool cuboid_geometry;
};


// Extern template instantiations
extern template class PlanetGeometry<2>;
extern template class PlanetGeometry<3>;

DYCOREPLANET_CLOSE_NAMESPACE
