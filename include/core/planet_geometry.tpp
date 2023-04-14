/**
 * @file planet_geometry.tpp
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

#include <core/planet_geometry.h>

DYCOREPLANET_OPEN_NAMESPACE

template <int dim>
PlanetGeometry<dim>::PlanetGeometry(double inner_radius,
                                    double outer_radius,
                                    bool   cuboid_geometry)
  : mpi_communicator(MPI_COMM_WORLD)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening),
                  parallel::distributed::Triangulation<
                    dim>::Settings::construct_multigrid_hierarchy)
  , center(Point<dim>()) // origin
  , inner_radius(inner_radius)
  , outer_radius(outer_radius)
  , cuboid_geometry(false)
{
  TimerOutput::Scope timing_section(
    computing_timer, "PlanetGeometry - constructor with grid generation");

  if (cuboid_geometry)
    {
      Point<dim> p0, p1;
      p1(0) = 1;
      p1(1) = 1;
      if (dim == 3)
        p1(2) = 1;
      center = 0.5 * (p0 + p1);
      GridGenerator::hyper_rectangle(triangulation,
                                     p0,
                                     p1,
                                     /* colorize */ true);

      std::vector<
        GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodicity_vector;

      /*
       * All dimensions up to the last are periodic (z-direction is always
       * bounded from below and from above)
       */
      for (unsigned int d = 0; d < dim - 1; ++d)
        {
          GridTools::collect_periodic_faces(triangulation,
                                            /*b_id1*/ 2 * (d + 1) - 2,
                                            /*b_id2*/ 2 * (d + 1) - 1,
                                            /*direction*/ d,
                                            periodicity_vector);
        }
    }
  else
    {
      if (true)
        {
          unsigned int n_cells  = (dim == 3) ? 6 : 12;
          bool         colorize = true;

          // take more cells per 'ring'(shell layer) for a thin shell to avoid
          // having long and flat cells.
          // if (outer_radius < inner_radius * 1.15)
          //   {
          //     n_cells = 2 * 96;
          //     // colorize = false; // not yet implemented for n_cells = 48
          //   }
          GridGenerator::hyper_shell(triangulation,
                                     center,
                                     inner_radius,
                                     outer_radius,
                                     ///* n_cells */ (dim == 3) ? 6 : 12,
                                     /* n_cells */ n_cells,
                                     ///* colorize */ true);
                                     /* colorize */ colorize);
          //          GridGenerator::hyper_ball(
          //            triangulation,
          //            center,
          //            outer_radius,
          //            /* attach_spherical_manifold_on_boundary_cells */ true);
        }
      else
        {
          Point<dim> p0, p1;
          p0(0) = -3;
          p0(1) = -3;
          p1(0) = 3;
          p1(1) = 3;
          if (dim == 3)
            {
              p0(2) = -3;
              p1(2) = 3;
            }
          center = 0.5 * (p0 + p1);
          GridGenerator::hyper_rectangle(triangulation,
                                         p0,
                                         p1,
                                         /* colorize */ true);
        }
    }

  global_Omega_diameter = GridTools::diameter(triangulation);

  pcout << "   Number of active cells:       " << triangulation.n_active_cells()
        << std::endl;
}



template <int dim>
PlanetGeometry<dim>::~PlanetGeometry()
{}



template <int dim>
void
PlanetGeometry<dim>::refine_global(unsigned int n_refine)
{
  TimerOutput::Scope timing_section(computing_timer,
                                    "PlanetGeometry - global refinement");

  triangulation.refine_global(n_refine);

  pcout << "   Number of active cells after global refinement:       "
        << triangulation.n_active_cells() << std::endl;
}



template <int dim>
void
PlanetGeometry<dim>::write_mesh_vtu()
{
  TimerOutput::Scope timing_section(computing_timer,
                                    "PlanetGeometry - write mesh to disk");

  DataOut<dim> data_out;
  data_out.attach_triangulation(triangulation);

  // Add data to indicate subdomain
  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    {
      subdomain(i) = triangulation.locally_owned_subdomain();
    }
  data_out.add_data_vector(subdomain, "subdomain");

  // Now build all data patches
  data_out.build_patches();

  const std::string filename_local =
    "boussinesq_palnet_mesh." +
    Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4) +
    ".vtu";

  std::ofstream output(filename_local.c_str());
  data_out.write_vtu(output);

  // Write a pvtu record on master process
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::vector<std::string> filenames;
      for (unsigned int i = 0;
           i < Utilities::MPI::n_mpi_processes(mpi_communicator);
           ++i)
        filenames.emplace_back("boussinesq_palnet_mesh." +
                               Utilities::int_to_string(i, 4) + ".vtu");

      std::string   master_file("boussinesq_palnet_mesh.pvtu");
      std::ofstream master_output(master_file.c_str());
      data_out.write_pvtu_record(master_output, filenames);
    }
}


DYCOREPLANET_CLOSE_NAMESPACE
