/**
 * @file nullspace.cc
 * @author Rebekka Beddig, adapted from deal.II Code
 * @version 0.1
 */
#include <core/boussinesq_model_commutator.h>


DYCOREPLANET_OPEN_NAMESPACE

// Implementation of the functions that handle the pressure nullspace constraint

namespace Standard
{
  // function for fixing one pressure dof (to remove it from the nullspace)
  // adapted from ASPECT
  template <int dim>
  void
  BoussinesqModel<dim>::setup_nullspace_constraints(
    AffineConstraints<double> &constraints,
    types::global_dof_index &  p_idx)
  {
    if (!parameters.remove_nullspace)
      return;

    IndexSet pressure_boundary_dofs(nse_dof_handler.n_dofs());
    IndexSet free_pressure_boundary_dofs(nse_dof_handler.n_dofs());
    if (parameters.cuboid_geometry)
      { // dofs on periodic boundary
        FEValuesExtractors::Scalar   pressure_components(dim);
        std::set<types::boundary_id> periodic_boundaries;
        periodic_boundaries.insert(0);
        periodic_boundaries.insert(1);
        if (dim == 3)
          {
            periodic_boundaries.insert(2);
            periodic_boundaries.insert(3);
          }
        // std::set<types::boundary_id> periodic_boundaries(boundaries);
        DoFTools::extract_boundary_dofs(nse_dof_handler,
                                        nse_fe.component_mask(
                                          pressure_components),
                                        pressure_boundary_dofs,
                                        periodic_boundaries);
        std::set<types::boundary_id> nonperiodic_boundaries;
        if (dim == 2)
          {
            nonperiodic_boundaries.insert(2);
            nonperiodic_boundaries.insert(3);
          }
        else if (dim == 3)
          {
            nonperiodic_boundaries.insert(4);
            nonperiodic_boundaries.insert(5);
          }

        // std::set<types::boundary_id> periodic_boundaries(boundaries);
        DoFTools::extract_boundary_dofs(nse_dof_handler,
                                        nse_fe.component_mask(
                                          pressure_components),
                                        free_pressure_boundary_dofs,
                                        nonperiodic_boundaries);
      }

    // Note: We want to add a single Dirichlet zero constraint for the pressure.
    // This is complicated by the fact that we need to find a DoF that is not
    // already constrained. In parallel the constraint needs to be added on all
    // processors where it is locally_relevant and all processors need to agree
    // on the index.

    // First find a candidate for DoF indices to constrain for the pressure
    // component.
    // types::global_dof_index p_idx;
    {
      p_idx = numbers::invalid_dof_index;

      unsigned int n_left_to_find = 1;

      std::vector<types::global_dof_index> local_dof_indices(
        nse_fe.dofs_per_cell);
      typename DoFHandler<dim>::active_cell_iterator cell;
      for (const auto &cell : nse_dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int i = 0; i < nse_fe.dofs_per_cell; ++i)
              {
                const unsigned int component =
                  nse_fe.system_to_component_index(i).first;

                // this->pcout
                //   << "   system to component index (i) = " << component
                //   << std::endl;
                // this->pcout << "   (i) = " << i << std::endl;

                if (component != dim)
                  continue; // only look at the pressure

                // const unsigned int velocity_component =
                //  component - introspection.component_indices.velocities[0];

                if (p_idx != numbers::invalid_dof_index)
                  continue; // already found one

                const types::global_dof_index idx = local_dof_indices[i];

                // constraint p_idx for all processors that have access to is
                // and if it is not yet constrained
                if (parameters.cuboid_geometry)
                  {
                    if (constraints.can_store_line(idx) &&
                        !constraints.is_constrained(idx) //)
                        && !pressure_boundary_dofs.is_element(idx) &&
                        free_pressure_boundary_dofs.is_element(idx))
                      {
                        p_idx = idx;
                        --n_left_to_find;
                      }
                  }
                else
                  {
                    if (constraints.can_store_line(idx) &&
                        !constraints.is_constrained(idx))
                      {
                        p_idx = idx;
                        --n_left_to_find;
                      }
                  }

                // are we done searching?
                if (n_left_to_find == 0)
                  break; // exit inner loop
              }

            if (n_left_to_find == 0)
              break; // exit outer loop
          }
    }

    // if (parameters.nullspace_removal)
    {
      // Make a reduction to find the smallest index (processors that
      // found a larger candidate just happened to not be able to store
      // that index with the minimum value). Note that it is possible that
      // some processors might not be able to find a potential DoF, for
      // example because they don't own any DoFs. On those processors we
      // will use dof_handler.n_dofs() when building the minimum (larger
      // than any valid DoF index).
      const types::global_dof_index global_idx =
        dealii::Utilities::MPI::min((p_idx != numbers::invalid_dof_index) ?
                                      p_idx :
                                      nse_dof_handler.n_dofs(),
                                    this->mpi_communicator);

      Assert(global_idx < nse_dof_handler.n_dofs(),
             ExcMessage("Error, couldn't find a pressure DoF to constrain."));

      // Finally set this DoF to zero (if the current MPI process
      // cares about it):
      if (constraints.can_store_line(global_idx))
        {
          Assert(!constraints.is_constrained((global_idx)), ExcInternalError());
          constraints.add_line(global_idx);
          // constraints.add_entry(global_idx, global_idx + 1, 1.0); // TODO
          // test
          this->pcout << "   constrained index = " << global_idx << std::endl;
        }
      // std::vector<types::global_dof_index> indices;
      // indices.push_back(global_idx);
      // constraints.resolve_indices(indices);
      // IndexSet constrained_associated_indices(nse_dof_handler.n_dofs());
      // constrained_associated_indices.add_indices(indices.begin(),
      //                                            indices.end());
      // constrained_associated_indices.print(this->pcout);
    }
  }



  //////////////////////////
  // function for fixing one pressure dof (to remove it from the nullspace)
  // adapted from ASPECT
  template <int dim>
  void
  BoussinesqModel<dim>::setup_nullspace_constraints_cell(
    AffineConstraints<double> &constraints,
    types::global_dof_index &  p_idx)
  {
    std::vector<types::global_dof_index>
      mean_value_dofs; // for the pressure dofs used in the local mean value
                       // constraint
    {
      p_idx = numbers::invalid_dof_index;

      unsigned int n_left_to_find = 1;

      std::vector<types::global_dof_index> local_dof_indices(
        nse_fe.dofs_per_cell);
      typename DoFHandler<dim>::active_cell_iterator cell;


      // choose the first active cell
      cell = nse_dof_handler.begin_active();
      if (cell->is_locally_owned())
        {
          // get all dof indices of this cell
          cell->get_dof_indices(local_dof_indices);

          // for (unsigned int i = 0; i < nse_fe.dofs_per_cell; ++i)
          //   std::cout << "cell dofs " << local_dof_indices[i] << std::endl;

          // filter pressure dofs
          for (unsigned int i = 0; i < nse_fe.dofs_per_cell; ++i)
            {
              const unsigned int component =
                nse_fe.system_to_component_index(i).first;

              // this->pcout << "   system to component index (i) = " <<
              // component
              //             << std::endl;
              // this->pcout << "   (i) = " << i << std::endl;

              if (component != dim)
                continue; // only look at the pressure

              const types::global_dof_index idx = local_dof_indices[i];

              // if (p_idx != numbers::invalid_dof_index)
              //   continue; // already found one


              if (constraints.can_store_line(idx) &&
                  !constraints.is_constrained(idx) && n_left_to_find != 0)
                {
                  p_idx = idx;
                  --n_left_to_find;
                }
              else
                {
                  mean_value_dofs.push_back(
                    idx); // save the pressure dofs of the cell
                }
            }

          // TODO add mean value constraint
        }
      else
        {
          ++cell; // TODO while until owned cell found
        }
    }

    // if (parameters.nullspace_removal)
    {
      // Make a reduction to find the smallest index (processors that
      // found a larger candidate just happened to not be able to store
      // that index with the minimum value). Note that it is possible that
      // some processors might not be able to find a potential DoF, for
      // example because they don't own any DoFs. On those processors we
      // will use dof_handler.n_dofs() when building the minimum (larger
      // than any valid DoF index).
      const types::global_dof_index global_idx =
        dealii::Utilities::MPI::min((p_idx != numbers::invalid_dof_index) ?
                                      p_idx :
                                      nse_dof_handler.n_dofs(),
                                    this->mpi_communicator);

      Assert(global_idx < nse_dof_handler.n_dofs(),
             ExcMessage("Error, couldn't find a pressure DoF to constrain."));

      // Finally set this DoF to zero (if the current MPI process
      // cares about it):
      if (constraints.can_store_line(global_idx))
        {
          Assert(!constraints.is_constrained((global_idx)), ExcInternalError());
          constraints.add_line(global_idx);
          this->pcout << "   constrained index = " << global_idx << std::endl;
        }

      // TODO fill entries
      std::vector<std::pair<types::global_dof_index, double>> entries;
      for (auto &dof : mean_value_dofs)
        {
          std::pair<types::global_dof_index, double> entry(dof, -1.0);
          // entries.push_back(std::pair(dof, -1.0));
          entries.push_back(entry);
        }
      constraints.add_entries(global_idx, entries);
      // test
      // constraints.add_entry(global_idx, global_idx + 1, 1.0);
    }
  }
  ///////////////////////////



  /*!
   * Make the pressure rhs compatible with the pressure null space, i.e.
   * subtract its mean value.
   * Adapted from ASPECTs corresponding function.
   */
  template <int dim>
  void
  BoussinesqModel<dim>::correct_pressure_rhs(LA::MPI::BlockVector &vector)
  {
    Assert(false, ExcNotImplemented());
    std::ignore = vector;
  }


  /*!
   * Normalize the pressure (because it's only defined up to a constant).
   * Adapted from ASPECTs corresponding function.
   */
  template <int dim>
  void
  BoussinesqModel<dim>::normalize_pressure(LA::MPI::BlockVector &vector)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Correct pressure to zero mean.");
    bool               easy = true;
    if (easy)
      {
        const double mean =
          VectorTools::compute_mean_value(nse_dof_handler,
                                          QGauss<dim>(
                                            parameters.nse_velocity_degree),
                                          vector,
                                          1);
        LA::MPI::BlockVector distributed_vector(nse_partitioning,
                                                this->mpi_communicator);
        distributed_vector = vector;
        distributed_vector.block(1).add(mean);
        this->pcout << "   Corrected pressure by " << mean << std::endl;
        vector = distributed_vector;
      }
    else
      {
        const FEValuesExtractors::Scalar &extractor_pressure(dim);

        double my_pressure = 0.0;
        double my_area     = 0.0;
        // volume normalization
        const QGauss<dim> quadrature(parameters.nse_velocity_degree + 1);

        const unsigned int n_q_points = quadrature.size();
        FEValues<dim>      fe_values(nse_fe,
                                quadrature,
                                update_JxW_values | update_values);

        std::vector<double> pressure_values(n_q_points);

        for (const auto &cell : nse_dof_handler.active_cell_iterators())
          if (cell->is_locally_owned())
            {
              fe_values.reinit(cell);
              fe_values[extractor_pressure].get_function_values(
                vector, pressure_values);

              for (unsigned int q = 0; q < n_q_points; ++q)
                {
                  my_pressure += pressure_values[q] * fe_values.JxW(q);
                  my_area += fe_values.JxW(q);
                }
            }


        // sum up the integrals from each processor and compute the result we
        // care about
        double pressure_adjustment = numbers::signaling_nan<double>();
        {
          const double my_temp[2] = {my_pressure, my_area};
          double       temp[2];
          Utilities::MPI::sum(my_temp, this->mpi_communicator, temp);
          const double pressure = temp[0];
          const double area     = temp[1];

          Assert(area > 0,
                 ExcMessage(
                   "While computing the average pressure, the area/volume "
                   "to integrate over was found to be zero or negative. This "
                   "indicates that no appropriate surface faces were found, "
                   "which is typically the case if the geometry model is not "
                   "set up correctly."));


          pressure_adjustment = -pressure / area;
        }

        // A complication is that we can't modify individual
        // elements of the solution vector since that one has ghost element.
        // rather, we first need to localize it and then distribute back
        LA::MPI::BlockVector distributed_vector(nse_partitioning,
                                                this->mpi_communicator);
        distributed_vector = vector;

        distributed_vector.block(1).add(pressure_adjustment);
        this->pcout << "   Corrected pressure by " << pressure_adjustment
                    << std::endl;

        // now get back to the original vector and return the adjustment used
        // in the computations above
        vector = distributed_vector;
      }
  }

} // namespace Standard

DYCOREPLANET_CLOSE_NAMESPACE


// explicit instantiations of the functions that are implemented in this file
DYCOREPLANET_OPEN_NAMESPACE
namespace Standard
{
  template void
  BoussinesqModel<2>::setup_nullspace_constraints(AffineConstraints<double> &,
                                                  types::global_dof_index &);
  template void
  BoussinesqModel<3>::setup_nullspace_constraints(AffineConstraints<double> &,
                                                  types::global_dof_index &);

  template void
  BoussinesqModel<2>::setup_nullspace_constraints_cell(
    AffineConstraints<double> &,
    types::global_dof_index &);
  template void
  BoussinesqModel<3>::setup_nullspace_constraints_cell(
    AffineConstraints<double> &,
    types::global_dof_index &);

  template void
  BoussinesqModel<2>::correct_pressure_rhs(LA::MPI::BlockVector &);
  template void
  BoussinesqModel<3>::correct_pressure_rhs(LA::MPI::BlockVector &);

  template void
  BoussinesqModel<2>::normalize_pressure(LA::MPI::BlockVector &);
  template void
  BoussinesqModel<3>::normalize_pressure(LA::MPI::BlockVector &);
} // namespace Standard
DYCOREPLANET_CLOSE_NAMESPACE