/**
 * @file boussinesq_model_commutator.tpp
 * @author Konrad Simon, Rebekka Beddig
 * @version 0.1
 */
#pragma once

#include <deal.II/base/function.h>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/vector_tools.h>

#include <core/boussinesq_model_commutator.h>
#include <core/planet_geometry.h>


// STL
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// preconditioners
#include <preconditioners/preconditioner_selector.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace Standard
{
  //////////////////////////////////////////////////////
  /// Standard Boussinesq model in H1-L2
  //////////////////////////////////////////////////////

  template <int dim>
  BoussinesqModel<dim>::BoussinesqModel(CoreModelData::Parameters &parameters_)
    : PlanetGeometry<dim>(parameters_.physical_constants.R0,
                          parameters_.physical_constants.R1,
                          parameters_.cuboid_geometry)
    , parameters(parameters_)
    , mapping(3)
    , nse_fe(FE_Q<dim>(parameters.nse_velocity_degree),
             dim,
             (parameters.use_locally_conservative_discretization ?
                static_cast<const FiniteElement<dim> &>(
                  FE_DGP<dim>(parameters.nse_velocity_degree - 1)) :
                static_cast<const FiniteElement<dim> &>(
                  FE_Q<dim>(parameters.nse_velocity_degree - 1))),
             1)
    , nse_dof_handler(this->triangulation)
    , temperature_fe(parameters.temperature_degree)
    , temperature_dof_handler(this->triangulation)
    , timestep_number(0)
    , cumulative_fgmres_iterations(0)
    , cumulative_fgmres_cpu_times(0)
    , cumulative_fgmres_wall_times(0)
    , rebuild_nse_matrix(true)
    , rebuild_nse_preconditioner(true)
    , rebuild_A_preconditioner(true)
    , rebuild_schur_preconditioner(true)
    , reassemble_nse_preconditioner(true)
    , rebuild_temperature_matrix(true)
    , rebuild_temperature_preconditioner(true)
  {
    TimerOutput::Scope timing_section(
      this->computing_timer,
      "BoussinesqModel - constructor and grid rescaling");

    /*
     * Rescale the original this->triangulation to the one scaled by the
     * reference length.
     */
    GridTools::scale(1 / parameters.reference_quantities.length,
                     this->triangulation);
    {
      /*
       * We must also rescale the domain parameters since this enters the data
       * of other objects (initial conditions etc)
       */
      if (parameters.cuboid_geometry)
        {
          /*
           * Note that this assumes that the lower left corner is the origin.
           */
          this->center /= parameters.reference_quantities.length;
        }

      this->inner_radius /= parameters.reference_quantities.length;
      this->outer_radius /= parameters.reference_quantities.length;
      this->global_Omega_diameter /= parameters.reference_quantities.length;
      parameters.physical_constants.R0 /=
        parameters.reference_quantities.length;
      parameters.physical_constants.R1 /=
        parameters.reference_quantities.length;
    }
  }



  template <int dim>
  BoussinesqModel<dim>::~BoussinesqModel()
  {}



  /////////////////////////////////////////////////////////////
  // System and dof setup
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::setup_nse_matrices(
    const std::vector<IndexSet> &nse_partitioning,
    const std::vector<IndexSet> &nse_relevant_partitioning)
  {
    nse_matrix.clear();
    // nse_constant_matrix.clear();
    // nse_diffusion_matrix.clear();
    nse_advection_matrix.clear();

    LA::BlockSparsityPattern     sp(nse_partitioning,
                                nse_partitioning,
                                nse_relevant_partitioning,
                                this->mpi_communicator);
    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        if (!((c == dim) && (d == dim)))
          coupling[c][d] = DoFTools::always;
        else
          coupling[c][d] = DoFTools::none;

    DoFTools::make_sparsity_pattern(nse_dof_handler,
                                    coupling,
                                    sp,
                                    nse_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      this->mpi_communicator));


    sp.compress();

    nse_matrix.reinit(sp);
    nse_advection_matrix.reinit(sp);
  }



  /**********************************************
   * local assembly velocity mass
   ***********************************************/

  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_velocity_mass(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::VelocityMass<dim> &                scratch,
    Assembly::CopyData::VelocityMass<dim> &               data)
  {
    const unsigned int dofs_per_cell =
      scratch.nse_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.nse_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);

    scratch.nse_fe_values.reinit(cell);

    data.local_matrix = 0;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_u[k] = scratch.nse_fe_values[velocities].value(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            data.local_matrix(i, j) +=
              (scratch.phi_u[i] * scratch.phi_u[j] // mass term
               ) *
              scratch.nse_fe_values.JxW(q);
      }

    cell->get_dof_indices(data.local_dof_indices);
  }

  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_velocity_mass(
    const Assembly::CopyData::VelocityMass<dim> &data)
  {
    nse_constraints.distribute_local_to_global(data.local_matrix,
                                               data.local_dof_indices,
                                               nse_mass_matrix);
  }


  template <int dim>
  void
  BoussinesqModel<dim>::assemble_velocity_mass_matrix(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assembly velocity mass");

    {
      nse_mass_matrix = 0;
      const QGauss<dim> quadrature_formula(parameters.nse_velocity_degree + 1);
      using CellFilter =
        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

      WorkStream::run(
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   nse_dof_handler.begin_active()),
        CellFilter(IteratorFilters::LocallyOwnedCell(), nse_dof_handler.end()),
        std::bind(&BoussinesqModel<dim>::local_assemble_velocity_mass,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3),
        std::bind(&BoussinesqModel<dim>::copy_local_to_global_velocity_mass,
                  this,
                  std::placeholders::_1),
        Assembly::Scratch::VelocityMass<dim>(parameters.time_step,
                                             time_index,
                                             nse_fe,
                                             quadrature_formula,
                                             mapping,
                                             update_JxW_values | update_values),
        Assembly::CopyData::VelocityMass<dim>(nse_fe));

      nse_mass_matrix.compress(VectorOperation::add);
    }
  }


  template <int dim>
  void
  BoussinesqModel<dim>::setup_nse_preconditioner(
    const std::vector<IndexSet> &nse_partitioning,
    const std::vector<IndexSet> &nse_relevant_partitioning)
  {
    nse_preconditioner_matrix.clear();
    nse_mass_matrix.clear();
    LA::BlockSparsityPattern sp(nse_partitioning,
                                nse_partitioning,
                                nse_relevant_partitioning,
                                this->mpi_communicator);

    Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
    bool                         need_block = false;
    for (unsigned int c = 0; c < dim + 1; ++c)
      for (unsigned int d = 0; d < dim + 1; ++d)
        {
          if (parameters.use_picard)
            need_block = (c != dim && d != dim);
          else
            need_block = (c == d && c != dim);
          if (need_block) // we only have diagonal blocks but not for the
                          // pressure
            coupling[c][d] = DoFTools::always;
          else
            coupling[c][d] = DoFTools::none;
        }

    DoFTools::make_sparsity_pattern(nse_dof_handler,
                                    coupling,
                                    sp,
                                    nse_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      this->mpi_communicator));

    sp.compress();

    nse_preconditioner_matrix.reinit(sp);

    nse_mass_matrix.reinit(sp);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::setup_temperature_matrices(
    const IndexSet &temperature_partitioner,
    const IndexSet &temperature_relevant_partitioner)
  {
    T_preconditioner.reset();
    temperature_mass_matrix.clear();
    temperature_stiffness_matrix.clear();
    temperature_matrix.clear();

    LA::SparsityPattern sp(temperature_partitioner,
                           temperature_partitioner,
                           temperature_relevant_partitioner,
                           this->mpi_communicator);
    DoFTools::make_sparsity_pattern(temperature_dof_handler,
                                    sp,
                                    temperature_constraints,
                                    false,
                                    Utilities::MPI::this_mpi_process(
                                      this->mpi_communicator));
    sp.compress();

    temperature_matrix.reinit(sp);
    temperature_mass_matrix.reinit(sp);
    temperature_advection_matrix.reinit(sp);
    temperature_stiffness_matrix.reinit(sp);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::setup_dofs()
  {
    TimerOutput::Scope timing_section(
      this->computing_timer, "BoussinesqModel - setup dofs of systems");

    /*
     * Setup dof handlers for nse and temperature
     */
    std::vector<unsigned int> nse_sub_blocks(dim + 1, 0);
    nse_sub_blocks[dim] = 1;

    nse_dof_handler.distribute_dofs(nse_fe);

    // if (parameters.use_schur_complement_solver)
    {
      DoFRenumbering::Cuthill_McKee(nse_dof_handler);
      //  DoFRenumbering::boost::king_ordering(nse_dof_handler);
    }

    DoFRenumbering::component_wise(nse_dof_handler, nse_sub_blocks);

    temperature_dof_handler.distribute_dofs(temperature_fe);

    /*
     * Count dofs
     */
    std::vector<types::global_dof_index> nse_dofs_per_block =
      DoFTools::count_dofs_per_fe_block(nse_dof_handler, nse_sub_blocks);
    const unsigned int n_u = nse_dofs_per_block[0], n_p = nse_dofs_per_block[1],
                       n_T = temperature_dof_handler.n_dofs();

    /*
     * Comma separated large numbers
     */
    std::locale s = this->pcout.get_stream().getloc();
    this->pcout.get_stream().imbue(std::locale(""));

    /*
     * Print some mesh and dof info
     */
    this->pcout << "Number of active cells: "
                << this->triangulation.n_global_active_cells() << " (on "
                << this->triangulation.n_levels() << " levels)" << std::endl
                << "Number of degrees of freedom: " << n_u + n_p + n_T << " ("
                << n_u << " + " << n_p << " + " << n_T << ")" << std::endl
                << std::endl;
    this->pcout.get_stream().imbue(s);

    /*
     * Print number of dofs to table of numerical results.
     */
    this->numerical_results.add_value("$n_u$", n_u);
    this->numerical_results.add_value("$n_p$", n_p);

    /*
     * Setup partitioners to store what dofs and matrix entries are stored on
     * the local processor
     */
    IndexSet temperature_partitioning(n_T),
      temperature_relevant_partitioning(n_T);

    // clear nse_partitioning and nse_relevant_partitioning (needed if we
    // restart with a finer mesh)
    nse_partitioning.clear();
    nse_relevant_partitioning.clear();
    {
      nse_index_set = nse_dof_handler.locally_owned_dofs();
      nse_partitioning.push_back(nse_index_set.get_view(0, n_u));
      nse_partitioning.push_back(nse_index_set.get_view(n_u, n_u + n_p));
      DoFTools::extract_locally_relevant_dofs(nse_dof_handler,
                                              nse_relevant_set);
      nse_relevant_partitioning.push_back(nse_relevant_set.get_view(0, n_u));
      nse_relevant_partitioning.push_back(
        nse_relevant_set.get_view(n_u, n_u + n_p));
      temperature_partitioning = temperature_dof_handler.locally_owned_dofs();
      DoFTools::extract_locally_relevant_dofs(
        temperature_dof_handler, temperature_relevant_partitioning);
    }


    /*
     * Setup constraints and boundary values for NSE. Make sure this is
     * consistent with the initial data.
     */
    {
      nse_constraints.clear();
      nse_constraints.reinit(nse_relevant_set);
      DoFTools::make_hanging_node_constraints(nse_dof_handler, nse_constraints);


      if (parameters.cuboid_geometry)
        {
          // bool only_dirichlet = true;
          if (parameters.only_dirichlet)
            {
              FEValuesExtractors::Vector velocity_components(0);

              for (unsigned int boundary_id = 0;
                   boundary_id < (dim == 2 ? 4 : 12);
                   ++boundary_id)
                VectorTools::interpolate_boundary_values(
                  nse_dof_handler,
                  boundary_id,
                  Functions::ZeroFunction<dim>(dim + 1),
                  nse_constraints,
                  nse_fe.component_mask(velocity_components));
            }
          else
            {
              std::vector<GridTools::PeriodicFacePair<
                typename DoFHandler<dim>::cell_iterator>>
                periodicity_vector;

              /*
               * All dimensions up to the last are periodic (z-direction is
               always
               * bounded from below and form above)
               */
              for (unsigned int d = 0; d < dim - 1; ++d)
                {
                  GridTools::collect_periodic_faces(nse_dof_handler,
                                                    /*b_id1*/ 2 * (d + 1) - 2,
                                                    /*b_id2*/ 2 * (d + 1) - 1,
                                                    /*direction*/ d,
                                                    periodicity_vector);
                }

#if (DEAL_II_VERSION_MAJOR == 9 && DEAL_II_VERSION_MINOR > 2)
              DoFTools::make_periodicity_constraints<dim, dim, double>(
                periodicity_vector, nse_constraints);
#else
              DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
                periodicity_vector, nse_constraints);
#endif

              FEValuesExtractors::Vector velocity_components(0);

              // No-slip on boundary id 2/4 (lower in 2d/3d)
              VectorTools::interpolate_boundary_values(
                nse_dof_handler,
                (dim == 2 ? 2 : 4),
                Functions::ZeroFunction<dim>(dim + 1),
                nse_constraints,
                nse_fe.component_mask(velocity_components));

              // No-flux on boundary id 3/5 (upper in 2d/3d)
              std::set<types::boundary_id> no_normal_flux_boundaries;
              no_normal_flux_boundaries.insert((dim == 2 ? 3 : 5));

              VectorTools::compute_no_normal_flux_constraints(
                nse_dof_handler,
                0,
                no_normal_flux_boundaries,
                nse_constraints,
                mapping);
            }
        }
      else // shell geometry
        {
          FEValuesExtractors::Vector velocity_components(0);

          // No-slip on boundary 0 (lower)
          VectorTools::interpolate_boundary_values(
            nse_dof_handler,
            0,
            Functions::ZeroFunction<dim>(dim + 1),
            nse_constraints,
            nse_fe.component_mask(velocity_components));

          // No-flux on upper boundary
          std::set<types::boundary_id> no_normal_flux_boundaries;
          no_normal_flux_boundaries.insert(1);

          VectorTools::compute_no_normal_flux_constraints(
            nse_dof_handler,
            /* first_vector_component */ 0,
            no_normal_flux_boundaries,
            nse_constraints,
            mapping);
        }

      // fix one pressure dof
      setup_nullspace_constraints(nse_constraints, p_fixed_idx);

      nse_constraints.close();
    }

    /*
     * Setup temperature constraints and boundary values
     */
    {
      temperature_constraints.clear();
      temperature_constraints.reinit(temperature_relevant_partitioning);
      DoFTools::make_hanging_node_constraints(temperature_dof_handler,
                                              temperature_constraints);

      if (parameters.cuboid_geometry)
        {
          std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>>
            periodicity_vector;

          /*
           * All dimensions up to the last are periodic (z-direction is
           always
           * bounded from below and form above)
           */
          for (unsigned int d = 0; d < dim - 1; ++d)
            {
              GridTools::collect_periodic_faces(temperature_dof_handler,
                                                /*b_id1*/ 2 * (d + 1) - 2,
                                                /*b_id2*/ 2 * (d + 1) - 1,
                                                /*direction*/ d,
                                                periodicity_vector);
            }

#if (DEAL_II_VERSION_MAJOR == 9 && DEAL_II_VERSION_MINOR > 2)
          DoFTools::make_periodicity_constraints<dim, dim, double>(
            periodicity_vector, temperature_constraints);
#else
          DoFTools::make_periodicity_constraints<DoFHandler<dim>>(
            periodicity_vector, temperature_constraints);
#endif


          // Dirchlet on boundary id 2/4 (lower in 2d/3d)
          VectorTools::interpolate_boundary_values(
            temperature_dof_handler,
            (dim == 2 ? 2 : 4),
            CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<dim>(
              this->center, this->global_Omega_diameter),
            temperature_constraints);
        }
      else
        {
          // Lower boundary is Dirichlet, upper is no-flux and natural
          VectorTools::interpolate_boundary_values(
            temperature_dof_handler,
            0,
            CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
              parameters.physical_constants.R0,
              parameters.physical_constants.R1),
            temperature_constraints);
        }

      temperature_constraints.close();
    }

    /*
     * Setup the matrix and vector objects.
     */
    setup_nse_matrices(nse_partitioning, nse_relevant_partitioning);
    setup_nse_preconditioner(nse_partitioning, nse_relevant_partitioning);

    setup_temperature_matrices(temperature_partitioning,
                               temperature_relevant_partitioning);

    nse_rhs.reinit(nse_partitioning,
                   nse_relevant_partitioning,
                   this->mpi_communicator,
                   true);
    nse_solution.reinit(nse_relevant_partitioning, this->mpi_communicator);
    old_nse_solution.reinit(nse_solution);
    nse_picard_iterate.reinit(nse_solution);

    temperature_rhs.reinit(temperature_partitioning,
                           temperature_relevant_partitioning,
                           this->mpi_communicator,
                           true);
    temperature_solution.reinit(temperature_relevant_partitioning,
                                this->mpi_communicator);
    old_temperature_solution.reinit(temperature_solution);

    // Rebuild system matrices and preconditioners in the next step
    rebuild_nse_matrix = true;
    if ((parameters.block00_preconditioner_name == "amg" &&
         !parameters.use_Mu) ||
        (parameters.schur_preconditioner_name == "bfbt" &&
         parameters.scale_with_velocity_mass && this->timestep_number == 0))
      {
        reassemble_nse_preconditioner = true;
      }
    else
      {
        // we use the system matrix for the preconditioner
        reassemble_nse_preconditioner = false;
      }
    rebuild_temperature_matrix         = true;
    rebuild_temperature_preconditioner = true;
  }



  /////////////////////////////////////////////////////////////
  // Assembly NSE preconditioner
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_nse_preconditioner(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::NSEPreconditioner<dim> &           scratch,
    Assembly::CopyData::NSEPreconditioner<dim> &          data)
  {
    const unsigned int dofs_per_cell = nse_fe.dofs_per_cell;
    const unsigned int n_q_points = scratch.nse_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    const double one_over_reynolds_number =
      (1. / CoreModelData::get_reynolds_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.kinematic_viscosity));

    scratch.nse_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);
    data.local_matrix = 0;

    double use_advection;
    if (parameters.use_picard)
      use_advection = 1.0;
    else
      use_advection = 0.0;

    if (parameters.use_picard)
      scratch.nse_fe_values[velocities].get_function_values(
        nse_picard_iterate, scratch.nse_picard_iterate);

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const Tensor<1, dim> old_picard_iterate = scratch.nse_picard_iterate[q];

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_u[k] = scratch.nse_fe_values[velocities].value(k, q);
            scratch.grad_phi_u[k] =
              scratch.nse_fe_values[velocities].gradient(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              // M_u + dt/Re L ( + A)
              // assemble the nonsymmetric diffusion matrix
              data.local_matrix(i, j) +=
                (scratch.phi_u[i] * scratch.phi_u[j] +
                 parameters.time_step * one_over_reynolds_number *
                   scalar_product(scratch.grad_phi_u[i],
                                  scratch.grad_phi_u[j]) //+
                 + use_advection * parameters.time_step *
                     (scratch.phi_u[i] *
                      (old_picard_iterate * scratch.grad_phi_u[j])) //
                 //      advection
                 ) *
                scratch.nse_fe_values.JxW(q);
            }
      }
  }



  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_nse_preconditioner(
    const Assembly::CopyData::NSEPreconditioner<dim> &data)
  {
    nse_constraints.distribute_local_to_global(data.local_matrix,
                                               data.local_dof_indices,
                                               nse_preconditioner_matrix);
  }


  template <int dim>
  void
  BoussinesqModel<dim>::assemble_nse_preconditioner(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assembly NSE preconditioner");

    {
      nse_preconditioner_matrix = 0;
      const QGauss<dim> quadrature_formula(parameters.nse_velocity_degree + 1);
      using CellFilter =
        FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

      WorkStream::run(
        CellFilter(IteratorFilters::LocallyOwnedCell(),
                   nse_dof_handler.begin_active()),
        CellFilter(IteratorFilters::LocallyOwnedCell(), nse_dof_handler.end()),
        std::bind(&BoussinesqModel<dim>::local_assemble_nse_preconditioner,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2,
                  std::placeholders::_3),
        std::bind(
          &BoussinesqModel<dim>::copy_local_to_global_nse_preconditioner,
          this,
          std::placeholders::_1),
        Assembly::Scratch::NSEPreconditioner<dim>(parameters.time_step,
                                                  time_index,
                                                  nse_fe,
                                                  quadrature_formula,
                                                  mapping,
                                                  update_JxW_values |
                                                    update_values |
                                                    update_gradients),
        Assembly::CopyData::NSEPreconditioner<dim>(nse_fe));

      nse_preconditioner_matrix.compress(VectorOperation::add);
      nse_schur_preconditioner_matrix.compress(VectorOperation::add);
    }
  }



  template <int dim>
  void
  BoussinesqModel<dim>::build_nse_preconditioner(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Build NSE preconditioner");

    this->pcout
      << "   Assembling and building the Navier-Stokes block preconditioner..."
      << std::endl;

    if (!rebuild_nse_preconditioner)
      return;

    if (reassemble_nse_preconditioner)
      {
        // Assemble div (grad) for the preconditioner of the upper left block
        assemble_nse_preconditioner(time_index);
      }



    ///////////////////////////////////////
    // Setup and initialize the preconditioners for the upper left block and
    // the Schur complement in the first time step.
    //////////////////////////////////////
    if (time_index == 0 && picarditeration_number == 0)
      {
        Timer timer;
        timer.start();
        if (rebuild_A_preconditioner)
          block00_preconditioner =
            std::make_shared<dealii::Preconditioners::PreconditionerSelector>(
              parameters.block00_preconditioner_name);
        timer.stop();
        std::cout << "   construct block 00 preconditioner: "
                  << timer.cpu_time() << std::endl;
        timer.restart();
        if (rebuild_schur_preconditioner)
          schur_preconditioner = std::make_shared<
            dealii::Preconditioners::SchurPreconditionerSelector>(
            parameters.schur_preconditioner_name, nse_matrix, nse_partitioning);
        timer.stop();
        std::cout << "   construct schur preconditioner: " << timer.cpu_time()
                  << std::endl;
        timer.restart();
        inner_A_preconditioner = std::make_shared<LA::PreconditionILU>();
        timer.stop();
        std::cout << "   default construct block 00 ILU: " << timer.cpu_time()
                  << std::endl;
        timer.restart();

        setup_preconditioner_data();
        timer.stop();
        std::cout << "   setup data: " << timer.cpu_time() << std::endl;
        timer.restart();
        initialize_preconditioner();
        timer.stop();
        std::cout << "   initialize preconditioner: " << timer.cpu_time()
                  << std::endl;
      }
    else
      {
        Timer timer;
        timer.start();
        setup_preconditioner_data();
        timer.stop();
        std::cout << "   setup data: " << timer.cpu_time() << std::endl;

        timer.restart();
        initialize_preconditioner();
        timer.stop();
        std::cout << "   initialize preconditioner: " << timer.cpu_time()
                  << std::endl;
      }


    // only rebuild preconditioner in the next time step if the time step
    // length changes or if we use the Picard iteration we have to rebuild the
    // preconditioner if
    //    - we use the Picard iteration
    //    - we use adaptive time steps
    //    - we don't have only Dirichlet BCs (i.e. the system matrix changes
    //    slightly)
    if (!parameters.adapt_time_step &&
        !((time_index == 0) &&
          parameters.use_picard)) // &&
                                  // (parameters.only_dirichlet &&
                                  // parameters.cuboid_geometry))
      rebuild_nse_preconditioner = false;
    else
      {
        rebuild_nse_preconditioner = true;
        rebuild_A_preconditioner   = true;
      }

    if ((parameters.only_dirichlet && parameters.cuboid_geometry) &&
        !parameters.use_picard)
      reassemble_nse_preconditioner = false;
    else
      reassemble_nse_preconditioner = true;

    this->pcout << std::endl;
  }


  /////////////////////////////////////////////////////////////
  // Assembly NSE system
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_nse_system(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::NSESystem<dim> &                   scratch,
    Assembly::CopyData::NSESystem<dim> &                  data)
  {
    const unsigned int dofs_per_cell =
      scratch.nse_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points = scratch.nse_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);

    const double one_over_reynolds_number =
      (1. / CoreModelData::get_reynolds_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.kinematic_viscosity));

    scratch.nse_fe_values.reinit(cell);

    typename DoFHandler<dim>::active_cell_iterator temperature_cell(
      &this->triangulation,
      cell->level(),
      cell->index(),
      &temperature_dof_handler);

    scratch.temperature_fe_values.reinit(temperature_cell);

    if (rebuild_nse_matrix)
      {
        data.local_matrix = 0;
      }

    if ((parameters.use_picard && picarditeration_number == 0) ||
        !parameters.use_picard)
      data.local_rhs = 0;

    scratch.temperature_fe_values.get_function_values(
      old_temperature_solution, scratch.old_temperature_values);

    // use the Picard iterate for the advection instead the solution of the
    // previous time step
    // nse_picard_iterate = 1.0;
    if (parameters.use_picard)
      scratch.nse_fe_values[velocities].get_function_values(
        nse_picard_iterate, scratch.nse_picard_iterate);

    scratch.nse_fe_values[velocities].get_function_values(
      old_nse_solution, scratch.old_velocity_values);

    if (!parameters.use_picard)
      scratch.nse_fe_values[velocities].get_function_gradients(
        old_nse_solution, scratch.old_velocity_grads);

    double use_advection;
    if (parameters.use_picard)
      use_advection = 1.0;
    else
      use_advection = 0.0;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        const double old_temperature = scratch.old_temperature_values[q];
        const double density_scaling = CoreModelData::density_scaling(
          parameters.physical_constants.expansion_coefficient,
          old_temperature,
          parameters.reference_quantities.temperature_ref);
        const Tensor<1, dim> old_velocity = scratch.old_velocity_values[q];
        const Tensor<1, dim> old_picard_iterate = scratch.nse_picard_iterate[q];
        const Tensor<2, dim> old_velocity_grads =
          transpose(scratch.old_velocity_grads[q]);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_u[k] = scratch.nse_fe_values[velocities].value(k, q);

            scratch.grads_phi_u[k] = transpose(
              scratch.nse_fe_values[velocities].symmetric_gradient(k, q));

            scratch.grad_phi_u[k] =
              scratch.nse_fe_values[velocities].gradient(k, q); // TODO check

            scratch.div_phi_u[k] =
              scratch.nse_fe_values[velocities].divergence(k, q);

            scratch.phi_p[k] = scratch.nse_fe_values[pressure].value(k, q);
          }

        Tensor<1, dim> coriolis;
        if (parameters.cuboid_geometry)
          coriolis = parameters.reference_quantities.length *
                     CoreModelData::coriolis_vector(
                       scratch.nse_fe_values.quadrature_point(q),
                       parameters.physical_constants.omega) /
                     parameters.reference_quantities.velocity;

        /*
         * Move everything to the LHS here.
         */
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              if (rebuild_nse_matrix)
                {
                  data.local_matrix(i, j) +=
                    (scratch.phi_u[i] * scratch.phi_u[j] // mass term
                     +
                     parameters.time_step *
                       (one_over_reynolds_number * 2 * scratch.grads_phi_u[i] *
                        scratch.grads_phi_u[j]) // eps(v):sigma(eps(u))
                     - (scratch.div_phi_u[i] *
                        scratch.phi_p[j]) // div(v)*p (solve for scaled
                                          // pressure dt*p)
                     - (scratch.phi_p[i] * scratch.div_phi_u[j]) // q*div(u)
                     +
                     use_advection * parameters.time_step *
                       (scratch.phi_u[i] * (old_picard_iterate *
                                            scratch.grad_phi_u[j])) // advection
                     ) *
                    scratch.nse_fe_values.JxW(q);
                }
            }


        const Tensor<1, dim> gravity =
          (parameters.reference_quantities.length /
           (parameters.reference_quantities.velocity *
            parameters.reference_quantities.velocity)) *
          (parameters.cuboid_geometry ?
             CoreModelData::vertical_gravity_vector(
               scratch.nse_fe_values.quadrature_point(q),
               parameters.physical_constants.gravity_constant) :
             CoreModelData::gravity_vector(
               scratch.nse_fe_values.quadrature_point(q),
               parameters.physical_constants.gravity_constant));

        /*
         * This is only the RHS
         */
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            // keep the same RHS during the Picard iteration
            if (parameters.use_picard || picarditeration_number == 0)
              {
                data.local_rhs(i) +=
                  (scratch.phi_u[i] *
                     old_velocity // part of the time derivative
                   + parameters.time_step * density_scaling * gravity *
                       scratch.phi_u[i] // gravity
                   -
                   parameters.time_step *
                     (dim == 2 ?
                        -2 * scratch.phi_u[i] * cross_product_2d(old_velocity) :
                        2 * scratch.phi_u[i] *
                          cross_product_3d(coriolis,
                                           old_velocity)) // coriolis force
                   ) *
                  scratch.nse_fe_values.JxW(q);
              }
            else
              {
                data.local_rhs(i) +=
                  (scratch.phi_u[i] *
                     old_velocity // part of the time derivative
                   + parameters.time_step * density_scaling * gravity *
                       scratch.phi_u[i] // gravity
                   - parameters.time_step * scratch.phi_u[i] *
                       (old_velocity *
                        old_velocity_grads) // advection at previous time
                   -
                   parameters.time_step *
                     (dim == 2 ?
                        -2 * scratch.phi_u[i] * cross_product_2d(old_velocity) :
                        2 * scratch.phi_u[i] *
                          cross_product_3d(coriolis,
                                           old_velocity)) // coriolis force
                   ) *
                  scratch.nse_fe_values.JxW(q);
              }
          }
      }

    cell->get_dof_indices(data.local_dof_indices);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_nse_system(
    const Assembly::CopyData::NSESystem<dim> &data)
  {
    if (rebuild_nse_matrix)
      {
        if ((parameters.use_picard && picarditeration_number == 0) ||
            !parameters.use_picard)
          {
            nse_constraints.distribute_local_to_global(data.local_matrix,
                                                       data.local_rhs,
                                                       data.local_dof_indices,
                                                       nse_matrix,
                                                       nse_rhs);
          }
        else
          {
            nse_constraints.distribute_local_to_global(data.local_matrix,
                                                       data.local_dof_indices,
                                                       nse_matrix);
          }
      }
    else
      {
        // We reuse the system matrices, so only we only have distribute the
        // constraints to the rhs
        nse_constraints.distribute_local_to_global(data.local_rhs,
                                                   data.local_dof_indices,
                                                   nse_rhs);
      }
  }



  template <int dim>
  void
  BoussinesqModel<dim>::assemble_nse_system(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble NSE system");
    this->pcout << "   Assembling Navier-Stokes system..." << std::flush;

    if (rebuild_nse_matrix)
      {
        nse_matrix = 0;
      }

    if ((parameters.use_picard && picarditeration_number == 0) ||
        !parameters.use_picard)
      nse_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.nse_velocity_degree + 1);
    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 nse_dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(), nse_dof_handler.end()),
      std::bind(&BoussinesqModel<dim>::local_assemble_nse_system,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&BoussinesqModel<dim>::copy_local_to_global_nse_system,
                this,
                std::placeholders::_1),
      Assembly::Scratch::NSESystem<dim>(parameters.time_step,
                                        time_index,
                                        nse_fe,
                                        mapping,
                                        quadrature_formula,
                                        (update_values |
                                         update_quadrature_points |
                                         update_JxW_values | update_gradients),
                                        temperature_fe,
                                        update_values),
      Assembly::CopyData::NSESystem<dim>(nse_fe));

    if (rebuild_nse_matrix)
      {
        nse_matrix.compress(VectorOperation::add);

        // we have rebuilt the nse system, so we don't need to repeat that in
        // the next call but we now want to rebuild the preconditioner for the
        // new system
        if ((parameters.only_dirichlet && parameters.cuboid_geometry) &&
            !parameters.use_picard)
          rebuild_nse_matrix = false;
        else
          rebuild_nse_matrix = true;
      }
    nse_rhs.compress(VectorOperation::add);


    this->pcout << std::endl;
  }


  /////////////////////////////////////////////////////////////
  // Assembly temperature
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_temperature_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::TemperatureMatrix<dim> &           scratch,
    Assembly::CopyData::TemperatureMatrix<dim> &          data)
  {
    const unsigned int dofs_per_cell =
      scratch.temperature_fe_values.get_fe().dofs_per_cell;
    const unsigned int n_q_points =
      scratch.temperature_fe_values.n_quadrature_points;

    const double one_over_peclet_number =
      (1. / CoreModelData::get_peclet_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.thermal_diffusivity));

    scratch.temperature_fe_values.reinit(cell);

    cell->get_dof_indices(data.local_dof_indices);

    data.local_mass_matrix      = 0;
    data.local_advection_matrix = 0;
    data.local_stiffness_matrix = 0;

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.grad_phi_T[k] =
              scratch.temperature_fe_values.shape_grad(k, q);
            scratch.phi_T[k] = scratch.temperature_fe_values.shape_value(k, q);
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              data.local_mass_matrix(i, j) +=
                (scratch.phi_T[i] * scratch.phi_T[j] *
                 scratch.temperature_fe_values.JxW(q));

              /*
               * TODO!!!
               */
              data.local_advection_matrix(i, j) += 0;

              data.local_stiffness_matrix(i, j) +=
                (one_over_peclet_number * scratch.grad_phi_T[i] *
                 scratch.grad_phi_T[j] * scratch.temperature_fe_values.JxW(q));
            }
      }
  }



  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_temperature_matrix(
    const Assembly::CopyData::TemperatureMatrix<dim> &data)
  {
    temperature_constraints.distribute_local_to_global(data.local_mass_matrix,
                                                       data.local_dof_indices,
                                                       temperature_mass_matrix);

    temperature_constraints.distribute_local_to_global(
      data.local_stiffness_matrix,
      data.local_dof_indices,
      temperature_stiffness_matrix);
  }



  template <int dim>
  void
  BoussinesqModel<dim>::assemble_temperature_matrix(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble temperature matrices");

    if (!rebuild_temperature_matrix)
      return;

    this->pcout << "   Assembling temperature matrix..." << std::flush;

    temperature_mass_matrix      = 0;
    temperature_advection_matrix = 0;
    temperature_stiffness_matrix = 0;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);

    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.end()),
      std::bind(&BoussinesqModel<dim>::local_assemble_temperature_matrix,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&BoussinesqModel<dim>::copy_local_to_global_temperature_matrix,
                this,
                std::placeholders::_1),
      Assembly::Scratch::TemperatureMatrix<dim>(parameters.time_step,
                                                time_index,
                                                temperature_fe,
                                                mapping,
                                                quadrature_formula),
      Assembly::CopyData::TemperatureMatrix<dim>(temperature_fe));

    temperature_mass_matrix.compress(VectorOperation::add);
    temperature_advection_matrix.compress(VectorOperation::add);
    temperature_stiffness_matrix.compress(VectorOperation::add);

    rebuild_temperature_matrix         = false;
    rebuild_temperature_preconditioner = true;

    this->pcout << std::endl;
  }



  /////////////////////////////////////////////////////////////
  // Assembly temperature RHS
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::local_assemble_temperature_rhs(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    Assembly::Scratch::TemperatureRHS<dim> &              scratch,
    Assembly::CopyData::TemperatureRHS<dim> &             data)
  {
    const unsigned int dofs_per_cell =
      scratch.temperature_fe_values.get_fe().dofs_per_cell;

    const unsigned int n_q_points =
      scratch.temperature_fe_values.n_quadrature_points;

    const FEValuesExtractors::Vector velocities(0);

    data.local_rhs     = 0;
    data.matrix_for_bc = 0;

    cell->get_dof_indices(data.local_dof_indices);

    scratch.temperature_fe_values.reinit(cell);

    typename DoFHandler<dim>::active_cell_iterator nse_cell(
      &this->triangulation, cell->level(), cell->index(), &nse_dof_handler);
    scratch.nse_fe_values.reinit(nse_cell);

    scratch.temperature_fe_values.get_function_values(
      old_temperature_solution, scratch.old_temperature_values);
    scratch.temperature_fe_values.get_function_gradients(
      old_temperature_solution, scratch.old_temperature_grads);

    scratch.nse_fe_values[velocities].get_function_values(
      nse_solution, scratch.old_velocity_values);

    const double one_over_peclet_number =
      (1. / CoreModelData::get_peclet_number(
              parameters.reference_quantities.velocity,
              parameters.reference_quantities.length,
              parameters.physical_constants.thermal_diffusivity));

    for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            scratch.phi_T[k] = scratch.temperature_fe_values.shape_value(k, q);
            scratch.grad_phi_T[k] =
              scratch.temperature_fe_values.shape_grad(k, q);
          }

        const double gamma =
          (parameters.reference_quantities.length /
           (parameters.reference_quantities.velocity *
            parameters.reference_quantities.temperature_ref)) *
          0; // CoreModelData::Boussinesq::TemperatureRHS value at quad point

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            data.local_rhs(i) +=
              (scratch.phi_T[i] * scratch.old_temperature_values[q] -
               parameters.time_step / (parameters.NSE_solver_interval) *
                 scratch.phi_T[i] * scratch.old_velocity_values[q] *
                 scratch.old_temperature_grads[q] -
               parameters.time_step / (parameters.NSE_solver_interval) * gamma *
                 scratch.phi_T[i]) *
              scratch.temperature_fe_values.JxW(q);

            if (temperature_constraints.is_inhomogeneously_constrained(
                  data.local_dof_indices[i]))
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  data.matrix_for_bc(j, i) +=
                    (scratch.phi_T[i] * scratch.phi_T[j] +
                     parameters.time_step / (parameters.NSE_solver_interval) *
                       one_over_peclet_number * scratch.grad_phi_T[i] *
                       scratch.grad_phi_T[j]) *
                    scratch.temperature_fe_values.JxW(q);
              }
          }
      }
  }


  template <int dim>
  void
  BoussinesqModel<dim>::copy_local_to_global_temperature_rhs(
    const Assembly::CopyData::TemperatureRHS<dim> &data)
  {
    temperature_constraints.distribute_local_to_global(data.local_rhs,
                                                       data.local_dof_indices,
                                                       temperature_rhs,
                                                       data.matrix_for_bc);
  }

  template <int dim>
  void
  BoussinesqModel<dim>::assemble_temperature_rhs(const double time_index)
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Assemble temperature RHS");

    this->pcout << "   Assembling temperature right-hand side..." << std::flush;

    temperature_matrix.copy_from(temperature_mass_matrix);
    temperature_matrix.add(parameters.time_step /
                             (parameters.NSE_solver_interval),
                           temperature_stiffness_matrix);

    if (rebuild_temperature_preconditioner == true)
      {
        T_preconditioner = std::make_shared<LA::PreconditionJacobi>();
        T_preconditioner->initialize(temperature_matrix);

        rebuild_temperature_preconditioner = false;
      }

    temperature_rhs = 0;

    const QGauss<dim> quadrature_formula(parameters.temperature_degree + 2);


    using CellFilter =
      FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

    WorkStream::run(
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.begin_active()),
      CellFilter(IteratorFilters::LocallyOwnedCell(),
                 temperature_dof_handler.end()),
      std::bind(&BoussinesqModel<dim>::local_assemble_temperature_rhs,
                this,
                std::placeholders::_1,
                std::placeholders::_2,
                std::placeholders::_3),
      std::bind(&BoussinesqModel<dim>::copy_local_to_global_temperature_rhs,
                this,
                std::placeholders::_1),
      Assembly::Scratch::TemperatureRHS<dim>(parameters.time_step,
                                             time_index,
                                             temperature_fe,
                                             nse_fe,
                                             mapping,
                                             quadrature_formula),
      Assembly::CopyData::TemperatureRHS<dim>(temperature_fe));

    temperature_rhs.compress(VectorOperation::add);

    this->pcout << std::endl;
  }



  template <int dim>
  double
  BoussinesqModel<dim>::get_maximal_velocity() const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.nse_velocity_degree);
    const unsigned int   n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping, nse_fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);
    double                           max_local_velocity = 0;

    for (const auto &cell : nse_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(nse_solution,
                                                    velocity_values);
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              max_local_velocity =
                std::max(max_local_velocity, velocity_values[q].norm());
            }
        }

    double max_global_velocity =
      Utilities::MPI::max(max_local_velocity, this->mpi_communicator);

    this->pcout << "   Max velocity (dimensionsless): " << max_global_velocity
                << std::endl;
    this->pcout << "   Max velocity (with dimensions): "
                << max_global_velocity *
                     parameters.reference_quantities.velocity
                << " m/s" << std::endl;

    return max_global_velocity;
  }


  template <int dim>
  double
  BoussinesqModel<dim>::get_cfl_number() const
  {
    const QIterated<dim> quadrature_formula(QTrapez<1>(),
                                            parameters.nse_velocity_degree);
    const unsigned int   n_q_points = quadrature_formula.size();

    FEValues<dim> fe_values(mapping, nse_fe, quadrature_formula, update_values);
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);

    const FEValuesExtractors::Vector velocities(0);
    double                           max_local_cfl = 0;

    for (const auto &cell : nse_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          fe_values[velocities].get_function_values(nse_solution,
                                                    velocity_values);
          double max_local_velocity = 1e-10;
          for (unsigned int q = 0; q < n_q_points; ++q)
            {
              max_local_velocity =
                std::max(max_local_velocity, velocity_values[q].norm());
            }
          max_local_cfl =
            std::max(max_local_cfl, max_local_velocity / cell->diameter());
        }

    double max_global_cfl =
      Utilities::MPI::max(max_local_cfl, this->mpi_communicator);

    this->pcout << "   Max of local CFL numbers: " << max_local_cfl
                << std::endl;

    return max_global_cfl;
  }


  template <int dim>
  void
  BoussinesqModel<dim>::recompute_time_step()
  {
    /*
     * Since we have the same geometry as in Deal.ii's mantle convection code
     * (step-32) we can determine the new step similarly.
     */
    const double scaling = (dim == 3 ? 0.25 : 1.0);
    parameters.time_step = (scaling / (2.1 * dim * std::sqrt(1. * dim)) /
                            (std::max(parameters.temperature_degree,
                                      parameters.nse_velocity_degree) *
                             get_cfl_number()));

    // const double maximal_velocity = get_maximal_velocity();

    this->pcout << "   New Time step (dimensionsless): " << parameters.time_step
                << std::endl;
    this->pcout << "   New Time step (with dimensions): "
                << parameters.time_step * parameters.reference_quantities.time
                << " s" << std::endl;
  }


  /////////////////////////////////////////////////////////////
  // refine mesh and interpolate coarse solution to fine mesh
  ////////////////////////////////////////////////////////////
  template <int dim>
  void
  BoussinesqModel<dim>::refine_mesh_and_interpolate_solution(
    LA::MPI::BlockVector &interpolated_nse_solution,
    LA::MPI::Vector &     interpolated_temperature_solution)
  {
    // initialize SolutionTransfer objects for the NSE and the temperature
    parallel::distributed::
      SolutionTransfer<dim, LA::MPI::BlockVector, DoFHandler<dim>>
        nse_trans(nse_dof_handler);
    parallel::distributed::
      SolutionTransfer<dim, LA::MPI::Vector, DoFHandler<dim>>
        temperature_trans(temperature_dof_handler);


    // prepare for global refinement
    // this->triangulation.set_all_refine_flags();

    // prepare SolutionTransfer objects for refinement
    nse_trans.prepare_for_coarsening_and_refinement(nse_solution);
    temperature_trans.prepare_for_coarsening_and_refinement(
      temperature_solution);

    // execute refinement
    // this->triangulation.execute_coarsening_and_refinement();

    // prepare and execute refinement
    this->triangulation.refine_global(1);

    setup_dofs();
    // // redistribute dofs
    // nse_dof_handler.distribute_dofs(nse_fe);

    // // interpolate the solution
    // interpolated_solution.reinit(nse_dof_handler.locally_owned_dofs(),
    //                              this->mpi_communicator);
    interpolated_nse_solution.reinit(nse_solution);
    interpolated_temperature_solution.reinit(temperature_solution);
    nse_trans.interpolate(interpolated_nse_solution);
    temperature_trans.interpolate(interpolated_temperature_solution);

    // distribute constraints
    nse_constraints.distribute(interpolated_nse_solution);
    temperature_constraints.distribute(interpolated_temperature_solution);
  }


  /////////////////////////////////////////////////////////////
  // solve
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::solve_NSE_block_preconditioned()
  {
    if ((timestep_number == 0) ||
        ((timestep_number > 0) &&
         (timestep_number % parameters.NSE_solver_interval == 0)))
      {
        TimerOutput::Scope timer_section(this->computing_timer,
                                         "   Solve (Navier-) Stokes system");
        this->pcout
          << "   Solving Navier-Stokes system for one time step with (block preconditioned solver)... "
          << std::endl;

        LA::MPI::BlockVector distributed_nse_solution(nse_rhs);
        distributed_nse_solution = nse_solution;
        /*
         * We solved only for a scaled pressure to
         * keep the system symmetric. So transform now and rescale later.
         */
        distributed_nse_solution.block(1) *= parameters.time_step;

        const unsigned int
          start = (distributed_nse_solution.block(0).size() +
                   distributed_nse_solution.block(1).local_range().first),
          end   = (distributed_nse_solution.block(0).size() +
                 distributed_nse_solution.block(1).local_range().second);

        for (unsigned int i = start; i < end; ++i)
          if (nse_constraints.is_constrained(i))
            distributed_nse_solution(i) = 0;

        if (parameters.correct_pressure_to_zero_mean)
          {
            normalize_pressure(nse_solution);
          }

        PrimitiveVectorMemory<LA::MPI::BlockVector> mem;
        unsigned int                                n_iterations = 0;
        const double  solver_tolerance = 1e-8 * nse_rhs.l2_norm();
        SolverControl solver_control(/* n_max_iter */ 40,
                                     solver_tolerance,
                                     /* log_history */ true,
                                     /* log_result */ true);
        SolverControl solver_control_refined(1000, // nse_matrix.m(),
                                             solver_tolerance,
                                             /* log_history */ false,
                                             /* log_result */ true);

        /*
         * We have only the actual pressure but need
         * to solve for a scaled pressure to keep the
         * system symmetric. Hence for the initial guess
         * we need to transform to the rescaled version.
         */
        distributed_nse_solution.block(1) *= parameters.time_step;

        //////////////////////////////////////////////////////
        // Solve the NSE with right preconditioned FGMRES
        /////////////////////////////////////////////////////
        Timer timer;
        Timer timer2;
        timer2.start();
        try
          {
            {
              using BlockPreconditionerType =
                LinearAlgebra::BlockSchurapproxPreconditioner<
                  Preconditioners::PreconditionerSelector,
                  Preconditioners::SchurPreconditionerSelector>;
              const BlockPreconditionerType preconditioner(
                nse_matrix,
                *block00_preconditioner,
                *schur_preconditioner,
                this->computing_timer);

              SolverFGMRES<LA::MPI::BlockVector> solver(
                solver_control,
                mem,
                SolverFGMRES<LA::MPI::BlockVector>::AdditionalData(30));
              timer.start();
              solver.solve(nse_matrix,
                           distributed_nse_solution,
                           nse_rhs,
                           preconditioner);
              timer.stop();
              n_iterations = solver_control.last_step();

              this->pcout << "   First residual "
                          << solver_control.initial_value() << std::endl;
              this->pcout << "   Last residual " << solver_control.last_value()
                          << std::endl;
            }
          }
        catch (SolverControl::NoConvergence &)
          {
            {
              const LinearAlgebra::BlockSchurapproxPreconditioner<
                Preconditioners::PreconditionerSelector,
                Preconditioners::SchurPreconditionerSelector>
                                                 preconditioner(nse_matrix,
                               *block00_preconditioner,
                               *schur_preconditioner,
                               this->computing_timer);
              SolverFGMRES<LA::MPI::BlockVector> solver(
                solver_control_refined,
                mem,
                SolverFGMRES<LA::MPI::BlockVector>::AdditionalData(50));

              timer.start();
              solver.solve(nse_matrix,
                           distributed_nse_solution,
                           nse_rhs,
                           preconditioner);
              timer.stop();

              this->pcout << "   First residual "
                          << solver_control.initial_value() << std::endl;
              this->pcout << "   Last residual "
                          << solver_control_refined.last_value() << std::endl;


              n_iterations = (solver_control.last_step() +
                              solver_control_refined.last_step());
            }
          }
        timer2.stop();


        nse_constraints.distribute(distributed_nse_solution);

        /*
         * We solved only for a scaled pressure to
         * keep the system symmetric. So retransform.
         */
        distributed_nse_solution.block(1) /= parameters.time_step;

        nse_solution = distributed_nse_solution;

        this->pcout << "   The FGMRes solver needed ";
        this->pcout << n_iterations << " iterations." << std::endl;

        numerical_results.add_value("iterations", n_iterations);
        numerical_results.add_value("t_solve", timer.cpu_time());
        numerical_results.start_new_row();

        cumulative_fgmres_iterations += n_iterations;
        cumulative_fgmres_cpu_times += timer2.cpu_time();
        cumulative_fgmres_wall_times += timer2.wall_time();
      } // solver time intervall constraint
  }



  template <int dim>
  void
  BoussinesqModel<dim>::solve_NSE_Schur_complement()
  {
    if ((timestep_number == 0) ||
        ((timestep_number > 0) &&
         (timestep_number % parameters.NSE_solver_interval == 0)))
      {
        TimerOutput::Scope timer_section(this->computing_timer,
                                         "   Solve NSE system");
        this->pcout
          << "   Solving Navier-Stokes system for one time step with (preconditioned Schur complement solver)... "
          << std::endl;

        /*
         * Initialize the inner preconditioner.
         */
        inner_schur_preconditioner =
          std::make_shared<InnerPreconditionerType>();

        // Fill preconditioner with life
        inner_schur_preconditioner->initialize(nse_matrix.block(0, 0), data);

        bool use_cg;
        if (parameters.use_picard)
          use_cg = false;
        else
          use_cg = true;
        using BlockInverseType = LinearAlgebra::InverseMatrix<
          LA::SparseMatrix,
          //  dealii::Preconditioners::PreconditionerSelector>;
          InnerPreconditionerType>;
        const BlockInverseType block_inverse(nse_matrix.block(0, 0),
                                             //*block00_preconditioner,
                                             *inner_schur_preconditioner,
                                             use_cg /*CG*/,
                                             false); // true /*print iters*/);

        LA::MPI::BlockVector distributed_nse_solution(nse_rhs);
        distributed_nse_solution = nse_solution;

        /*
         * We solved only for a scaled pressure to
         * keep the system symmetric. So transform now and rescale later.
         */
        distributed_nse_solution.block(1) *= parameters.time_step;

        const unsigned int
          start = (distributed_nse_solution.block(0).size() +
                   distributed_nse_solution.block(1).local_range().first),
          end   = (distributed_nse_solution.block(0).size() +
                 distributed_nse_solution.block(1).local_range().second);

        for (unsigned int i = start; i < end; ++i)
          if (nse_constraints.is_constrained(i))
            distributed_nse_solution(i) = 0;

        // tmp of size block(0)
        LA::MPI::Vector tmp(nse_partitioning[0], this->mpi_communicator);

        // Set up Schur complement
        LinearAlgebra::SchurComplement<LA::BlockSparseMatrix,
                                       // LA::MPI::Vector,
                                       BlockInverseType>
          schur_complement(nse_matrix,
                           block_inverse,
                           nse_partitioning,
                           this->mpi_communicator);

        // Compute schur_rhs = -g + C*A^{-1}*f
        LA::MPI::Vector schur_rhs(nse_partitioning[1], this->mpi_communicator);

        this->pcout
          << std::endl
          << "      Apply inverse of block (0,0) for Schur complement solver RHS..."
          << std::endl;

        block_inverse.vmult(tmp, nse_rhs.block(0));
        nse_matrix.block(1, 0).vmult(schur_rhs, tmp);
        schur_rhs -= nse_rhs.block(1);

        this->pcout << "      Schur complement solver RHS computation done..."
                    << std::endl
                    << std::endl;

        {
          TimerOutput::Scope t(
            this->computing_timer,
            "      Solve NSE system - Schur complement solver with " +
              parameters.schur_preconditioner_name + " (for pressure)");

          this->pcout << "      Apply Schur complement solver with "
                      << parameters.schur_preconditioner_name << "..."
                      << std::endl;

          // Set Solver parameters for solving for p
          const int max_iters = std::min((int)nse_matrix.block(1, 0).m(), 500);
          SolverControl solver_control(max_iters, 1e-8 * schur_rhs.l2_norm());
          solver_control.enable_history_data();
          Timer time_solver;
          if (!parameters.use_picard) // for comparison of the update
            {
              SolverMinRes<LA::MPI::Vector> schur_solver(solver_control);
              time_solver.restart();
              schur_solver.solve(schur_complement,
                                 distributed_nse_solution.block(1),
                                 schur_rhs,
                                 *schur_preconditioner);
              time_solver.stop();
            }
          else
            {
              SolverGMRES<LA::MPI::Vector>::AdditionalData gmres_data;
              gmres_data.right_preconditioning = true;
              // our problem is not symmetric, so we need to use GMRes
              SolverGMRES<LA::MPI::Vector> schur_solver(solver_control,
                                                        gmres_data);

              time_solver.restart();
              schur_solver.solve(schur_complement,
                                 distributed_nse_solution.block(1),
                                 schur_rhs,
                                 *schur_preconditioner);
              time_solver.stop();
            }

          this->pcout << "      Iterative Schur complement solver "
                      << parameters.schur_preconditioner_name
                      << " converged in " << solver_control.last_step()
                      << " iterations." << std::endl
                      << std::endl;
          this->pcout << "        Pressure solve needed "
                      << time_solver.wall_time() << " s. (wall time)"
                      << std::endl;
          this->pcout << "        Pressure solve needed "
                      << time_solver.cpu_time() << " s. (cpu time)"
                      << std::endl;

          std::vector<double> residuals(solver_control.last_step() + 1);
          residuals = solver_control.get_history_data();

          this->pcout << " First residual " << solver_control.initial_value()
                      << std::endl;
          this->pcout << " Last residual " << solver_control.last_value()
                      << std::endl;
          for (unsigned int i = 0; i < solver_control.last_step() + 1; ++i)
            {
              this->pcout << " Residual " << i << ": " << residuals[i]
                          << std::endl;
            }

          nse_constraints.distribute(distributed_nse_solution);
        } // solve for pressure

        {
          TimerOutput::Scope t(
            this->computing_timer,
            "      Solve NSE system - outer CG solver (for u)");

          this->pcout << "      Apply outer solver..." << std::endl;

          // use computed u to solve for sigma
          nse_matrix.block(0, 1).vmult(tmp, distributed_nse_solution.block(1));
          tmp *= -1;
          tmp += nse_rhs.block(0);

          // Solve for velocity
          // block_inverse.vmult(distributed_nse_solution.block(0), tmp);

          SolverControl solver_control_u(
            std::min(static_cast<std::size_t>(tmp.size()),
                     static_cast<std::size_t>(1000)),
            1e-8 * tmp.l2_norm());

          if (parameters.use_picard)
            {
              SolverGMRES<LA::MPI::Vector> solver_u(solver_control_u);
              solver_u.solve(nse_matrix.block(0, 0),
                             distributed_nse_solution.block(0),
                             tmp,
                             // *inner_A_preconditioner);
                             *block00_preconditioner);

              this->pcout << "         velocity solver needed "
                          << solver_control_u.last_step() << " iterations."
                          << std::endl;
            }
          else
            {
              SolverCG<LA::MPI::Vector> solver_u(solver_control_u);
              solver_u.solve(nse_matrix.block(0, 0),
                             distributed_nse_solution.block(0),
                             tmp,
                             *block00_preconditioner);

              this->pcout << "         velocity solver needed "
                          << solver_control_u.last_step() << " iterations."
                          << std::endl;
            }

          this->pcout << "      Outer solver completed." << std::endl
                      << std::endl;

          nse_constraints.distribute(distributed_nse_solution);

          /*
           * We solved only for a scaled pressure to
           * keep the system symmetric. So retransform.
           */
          distributed_nse_solution.block(1) /= parameters.time_step;

          nse_solution = distributed_nse_solution;

        } // solve for velocity
      }   // solver time intervall constraint
  }



  template <int dim>
  void
  BoussinesqModel<dim>::solve_temperature()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Solve temperature system");

    this->pcout << "      Apply temperature solver..." << std::endl;

    SolverControl solver_control(temperature_matrix.m(),
                                 1e-12 * temperature_rhs.l2_norm(),
                                 /* log_history */ false,
                                 /* log_result */ false);

    SolverCG<LA::MPI::Vector> cg(solver_control);

    LA::MPI::Vector distributed_temperature_solution(temperature_rhs);

    distributed_temperature_solution = temperature_solution;

    cg.solve(temperature_matrix,
             distributed_temperature_solution,
             temperature_rhs,
             *T_preconditioner);

    temperature_constraints.distribute(distributed_temperature_solution);

    temperature_solution = distributed_temperature_solution;

    this->pcout << "      " << solver_control.last_step()
                << " CG iterations for temperature" << std::endl;

    /*
     * Compute global max and min temerature. Needs MPI communication.
     */
    double temperature[2] = {std::numeric_limits<double>::max(),
                             -std::numeric_limits<double>::max()};
    double global_temperature[2];

    for (unsigned int i = distributed_temperature_solution.local_range().first;
         i < distributed_temperature_solution.local_range().second;
         ++i)
      {
        temperature[0] =
          std::min<double>(temperature[0], distributed_temperature_solution(i));
        temperature[1] =
          std::max<double>(temperature[1], distributed_temperature_solution(i));
      }
    temperature[0] *= -1.0;

    Utilities::MPI::max(temperature,
                        this->mpi_communicator,
                        global_temperature);

    global_temperature[0] *= -1.0;

    this->pcout << "      Temperature range: " << global_temperature[0] << ' '
                << global_temperature[1] << std::endl
                << std::endl;
  }

  /////////////////////////////////////////////////////////////
  // Postprocessor
  /////////////////////////////////////////////////////////////


  template <int dim>
  BoussinesqModel<dim>::Postprocessor::Postprocessor(
    const unsigned int               partition,
    const CoreModelData::Parameters &parameters)
    : partition(partition)
    , parameters(parameters)
  {}



  template <int dim>
  std::vector<std::string>
  BoussinesqModel<dim>::Postprocessor::get_names() const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("p");
    solution_names.emplace_back("T");
    solution_names.emplace_back("partition");
    return solution_names;
  }



  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  BoussinesqModel<dim>::Postprocessor::get_data_component_interpretation() const
  {
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    return interpretation;
  }



  template <int dim>
  UpdateFlags
  BoussinesqModel<dim>::Postprocessor::get_needed_update_flags() const
  {
    return update_values | update_gradients | update_quadrature_points;
  }



  template <int dim>
  void
  BoussinesqModel<dim>::Postprocessor::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &inputs,
    std::vector<Vector<double>> &               computed_quantities) const
  {
    const unsigned int n_quadrature_points = inputs.solution_values.size();

    Assert(inputs.solution_gradients.size() == n_quadrature_points,
           ExcInternalError());
    Assert(computed_quantities.size() == n_quadrature_points,
           ExcInternalError());
    Assert(inputs.solution_values[0].size() == dim + 2, ExcInternalError());

    // Rescale to physical quantities
    // Compute pressure scale (p_ref = rho_ref * v_ref^2)
    double reference_pressure = parameters.physical_constants.density *
                                parameters.reference_quantities.velocity *
                                parameters.reference_quantities.velocity;
    for (unsigned int q = 0; q < n_quadrature_points; ++q)
      {
        for (unsigned int d = 0; d < dim; ++d)
          computed_quantities[q](d) = inputs.solution_values[q](d) *
                                      parameters.reference_quantities.velocity;

        const double pressure       = (inputs.solution_values[q](dim));
        computed_quantities[q](dim) = pressure * reference_pressure;

        const double temperature = inputs.solution_values[q](dim + 1);
        computed_quantities[q](dim + 1) =
          temperature * parameters.reference_quantities.temperature_ref;

        computed_quantities[q](dim + 2) = partition;
      }
  }



  /////////////////////////////////////////////////////////////
  // Output results
  /////////////////////////////////////////////////////////////


  template <int dim>
  void
  BoussinesqModel<dim>::output_results()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "Postprocessing and output");

    this->pcout << "   Writing Boussinesq solution for one timestep... "
                << std::flush;

    const FESystem<dim> joint_fe(nse_fe, 1, temperature_fe, 1);

    DoFHandler<dim> joint_dof_handler(this->triangulation);
    joint_dof_handler.distribute_dofs(joint_fe);

    Assert(joint_dof_handler.n_dofs() ==
             nse_dof_handler.n_dofs() + temperature_dof_handler.n_dofs(),
           ExcInternalError());

    LA::MPI::Vector joint_solution;

    joint_solution.reinit(joint_dof_handler.locally_owned_dofs(),
                          this->mpi_communicator);

    {
      std::vector<types::global_dof_index> local_joint_dof_indices(
        joint_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_nse_dof_indices(
        nse_fe.dofs_per_cell);
      std::vector<types::global_dof_index> local_temperature_dof_indices(
        temperature_fe.dofs_per_cell);

      typename DoFHandler<dim>::active_cell_iterator
        joint_cell       = joint_dof_handler.begin_active(),
        joint_endc       = joint_dof_handler.end(),
        nse_cell         = nse_dof_handler.begin_active(),
        temperature_cell = temperature_dof_handler.begin_active();
      for (; joint_cell != joint_endc;
           ++joint_cell, ++nse_cell, ++temperature_cell)
        if (joint_cell->is_locally_owned())
          {
            joint_cell->get_dof_indices(local_joint_dof_indices);
            nse_cell->get_dof_indices(local_nse_dof_indices);
            temperature_cell->get_dof_indices(local_temperature_dof_indices);

            for (unsigned int i = 0; i < joint_fe.dofs_per_cell; ++i)
              if (joint_fe.system_to_base_index(i).first.first == 0)
                {
                  Assert(joint_fe.system_to_base_index(i).second <
                           local_nse_dof_indices.size(),
                         ExcInternalError());

                  joint_solution(local_joint_dof_indices[i]) = nse_solution(
                    local_nse_dof_indices[joint_fe.system_to_base_index(i)
                                            .second]);
                }
              else
                {
                  Assert(joint_fe.system_to_base_index(i).first.first == 1,
                         ExcInternalError());
                  Assert(joint_fe.system_to_base_index(i).second <
                           local_temperature_dof_indices.size(),
                         ExcInternalError());

                  joint_solution(local_joint_dof_indices[i]) =
                    temperature_solution(
                      local_temperature_dof_indices
                        [joint_fe.system_to_base_index(i).second]);
                }
          } // end if is_locally_owned()
    }       // end for ++joint_cell

    joint_solution.compress(VectorOperation::insert);

    IndexSet locally_relevant_joint_dofs(joint_dof_handler.n_dofs());
    DoFTools::extract_locally_relevant_dofs(joint_dof_handler,
                                            locally_relevant_joint_dofs);

    LA::MPI::Vector locally_relevant_joint_solution;
    locally_relevant_joint_solution.reinit(locally_relevant_joint_dofs,
                                           this->mpi_communicator);
    locally_relevant_joint_solution = joint_solution;

    Postprocessor postprocessor(
      Utilities::MPI::this_mpi_process(this->mpi_communicator), parameters);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(joint_dof_handler);

    data_out.add_data_vector(locally_relevant_joint_solution, postprocessor);

    data_out.build_patches(parameters.nse_velocity_degree);

    static int        out_index = 0;
    const std::string filename =
      (parameters.filename_output + "-" +
       Utilities::int_to_string(out_index, 5) + "." +
       Utilities::int_to_string(this->triangulation.locally_owned_subdomain(),
                                4) +
       ".vtu");
    std::ofstream output(parameters.dirname_output + "/" + filename);
    data_out.write_vtu(output);

    /*
     * Write pvtu record
     */
    if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             ++i)
          filenames.push_back(std::string(parameters.filename_output + "-") +
                              Utilities::int_to_string(out_index, 5) + "." +
                              Utilities::int_to_string(i, 4) + ".vtu");

        const std::string pvtu_master_filename =
          (parameters.filename_output + "-" +
           Utilities::int_to_string(out_index, 5) + ".pvtu");
        std::ofstream pvtu_master(parameters.dirname_output + "/" +
                                  pvtu_master_filename);
        data_out.write_pvtu_record(pvtu_master, filenames);
      }
    out_index++;

    this->pcout << std::endl;
  }


  /////////////////////////////////////////////////////////////
  // Print parameters
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::print_paramter_info() const
  {
    this->pcout << "-------------------- Paramter info --------------------"
                << std::endl
                << "Earth radius                         :   "
                << parameters.physical_constants.R0 << std::endl
                << "Atmosphere height                    :   "
                << parameters.physical_constants.atm_height << std::endl
                << std::endl
                << "Reference pressure                   :   "
                << parameters.physical_constants.pressure << std::endl
                << "Reference length                     :   "
                << parameters.reference_quantities.length << std::endl
                << "Reference velocity                   :   "
                << parameters.reference_quantities.velocity << std::endl
                << "Reference time                       :   "
                << parameters.reference_quantities.time << std::endl
                << "Reference atmosphere temperature     :   "
                << parameters.reference_quantities.temperature_ref << std::endl
                << "Atmosphere temperature change        :   "
                << parameters.reference_quantities.temperature_change
                << std::endl
                << std::endl
                << "Reynolds number                      :   "
                << CoreModelData::get_reynolds_number(
                     parameters.reference_quantities.velocity,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.kinematic_viscosity)
                << std::endl
                << "Peclet number                        :   "
                << CoreModelData::get_peclet_number(
                     parameters.reference_quantities.velocity,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.thermal_diffusivity)
                << std::endl
                << "Rossby number                        :   "
                << CoreModelData::get_rossby_number(
                     parameters.reference_quantities.length,
                     parameters.physical_constants.omega,
                     parameters.reference_quantities.velocity)
                << std::endl
                << "Reference accelertion                :   "
                << CoreModelData::get_reference_accelleration(
                     parameters.reference_quantities.length,
                     parameters.reference_quantities.velocity)
                << std::endl
                << "Grashoff number                      :   "
                << CoreModelData::get_grashoff_number(
                     dim,
                     parameters.physical_constants.gravity_constant,
                     parameters.physical_constants.expansion_coefficient,
                     parameters.reference_quantities.temperature_change,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.kinematic_viscosity)
                << std::endl
                << "Prandtl number                       :   "
                << CoreModelData::get_prandtl_number(
                     parameters.physical_constants.kinematic_viscosity,
                     parameters.physical_constants.thermal_diffusivity)
                << std::endl
                << "Rayleigh number                      :   "
                << CoreModelData::get_rayleigh_number(
                     dim,
                     parameters.physical_constants.gravity_constant,
                     parameters.physical_constants.expansion_coefficient,
                     parameters.reference_quantities.temperature_change,
                     parameters.reference_quantities.length,
                     parameters.physical_constants.kinematic_viscosity,
                     parameters.physical_constants.thermal_diffusivity)
                << std::endl
                << "-------------------------------------------------------"
                << std::endl
                << std::endl;
  }



  template <int dim>
  void
  BoussinesqModel<dim>::print_preconditioner_parameter_info() const
  {
    this->pcout
      << "------------- Preconditioner parameter info -----------" << std::endl
      << "BCs                               :   " << parameters.only_dirichlet
      << std::endl
      << "Picard linearization              :   " << parameters.use_picard
      << std::endl
      << "Velocity block preconditioner     :   "
      << parameters.block00_preconditioner_name << std::endl
      << "AMG aggregation threshold         :   "
      << parameters.AMG_aggregation_threshold << std::endl
      << "Prec. velocity block with M_u     :   " << parameters.use_Mu
      << std::endl
      << "Schur complement preconditioner   :   "
      << parameters.schur_preconditioner_name << std::endl
      << "AMG aggregation threshold in BFBt :   "
      << parameters.bfbt_aggregation_threshold << std::endl
      << "Use only AMG to solve (BB^T)^-1   :   " << parameters.only_amg
      << std::endl
      << "Use IC(0) to solve (BB^T)^-1      :   " << parameters.use_ic
      << std::endl
      << "CG solver tol. for (BB^T)^-1      :   " << parameters.tolerance
      << std::endl
      << "Precondition BB^T                 :   "
      << parameters.precondition_poisson << std::endl
      << "Use scaled BB^T (B M B^T)         :   " << parameters.bfbt_scale
      << std::endl
      << "Scale BB^T with M_u               :   "
      << parameters.scale_with_velocity_mass << std::endl
      << "Damp boundary dofs in BB^T        :   "
      << parameters.damping_coefficient << std::endl
      << "Weight in error matrix            :   " << parameters.weight
      << std::endl
      << "SVD solver type                   :   " << parameters.svd_solver_type
      << std::endl
      << "Update rank                       :   " << parameters.rank
      << std::endl
      << "Number of updates                 :   " << parameters.number
      << std::endl
      << std::endl
      << "-------------------------------------------------------" << std::endl
      << std::endl;
  }


  template <int dim>
  void
  BoussinesqModel<dim>::initialize_numerical_results()
  {
    this->numerical_results.declare_column("system"); // Oseen/Stokes
    this->numerical_results.declare_column("$n_u$");
    this->numerical_results.declare_column("$n_p$");
    this->numerical_results.declare_column("preconditioner");
    this->numerical_results.declare_column("rank");
    this->numerical_results.declare_column("number");
    this->numerical_results.declare_column("iterations");
    this->numerical_results.declare_column("t_setup");
    this->numerical_results.declare_column("t_solve");

    // add system parameters
    this->numerical_results.add_value("system",
                                      this->parameters.use_picard ? "Oseen" :
                                                                    "Stokes");
  }

  template <int dim>
  void
  BoussinesqModel<dim>::print_numerical_results(
    const std::string &file_name) const
  {
    std::ofstream output_file(file_name,
                              std::fstream::in | std::fstream::out |
                                std::fstream::app);
    numerical_results.write_tex(output_file, false);
  }


  /////////////////////////////////////////////////////////////
  // Run function
  /////////////////////////////////////////////////////////////

  template <int dim>
  void
  BoussinesqModel<dim>::run()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "BoussinesqModel - global run function");

    // initialize table for numerical results
    initialize_numerical_results();

    // call refinement routine in base class
    this->refine_global(parameters.initial_global_refinement);

    setup_dofs();

    // Compute initial time step: scale with the time scale (=l_ref/v_ref) and
    // the minimal cell diameter
    double hK = 1;
    for (const auto &cell : nse_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hK = std::min(hK, cell->diameter());
        }

    print_preconditioner_parameter_info();

    print_paramter_info();

    /*
     * Initial values.
     */
    nse_solution = 0;

    LA::MPI::Vector solution_tmp(temperature_dof_handler.locally_owned_dofs());

    if (parameters.cuboid_geometry)
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<dim>(
            this->center, this->global_Omega_diameter),
          solution_tmp);
      }
    else
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
            parameters.physical_constants.R0, parameters.physical_constants.R1),
          solution_tmp);
      }

    old_nse_solution = nse_solution;
    if (parameters.use_picard)
      nse_picard_iterate = nse_solution;
    temperature_solution     = solution_tmp;
    old_temperature_solution = solution_tmp;

    try
      {
        Tools::create_data_directory(parameters.dirname_output);
      }
    catch (std::runtime_error &e)
      {
        // No exception handling here.
      }

    output_results();

    // Time iteration
    double time_index = 0;
    do
      {
        if ((timestep_number > 0) &&
            (timestep_number % parameters.NSE_solver_interval == 0) &&
            parameters.adapt_time_step)
          {
            recompute_time_step();
          }
        else
          {
            /*
             * This is just informative output
             */
            get_cfl_number();
            get_maximal_velocity();
          }

        this->pcout << "----------------------------------------" << std::endl
                    << "Time step " << timestep_number << ":  t=" << time_index
                    << " -> t=" << time_index + parameters.time_step
                    << "  (dt=" << parameters.time_step
                    << " | final time=" << parameters.final_time << ")"
                    << std::endl;

        if (timestep_number == 0)
          {
            assemble_nse_system(time_index);

            build_nse_preconditioner(time_index);
          }
        else if ((timestep_number > 0) &&
                 (timestep_number % parameters.NSE_solver_interval == 0))
          {
            assemble_nse_system(time_index);

            build_nse_preconditioner(time_index);
          }

        assemble_temperature_matrix(time_index);
        assemble_temperature_rhs(time_index);

        // Solve the Navier-Stokes equations
        solve_nse(time_index, parameters.time_step);

        // solve temperature advection diffusion equation
        solve_temperature();

        output_results();

        /*
         * Print summary after a NSE system has been solved.
         */
        if ( //(timestep_number > 0) &&
          (timestep_number % parameters.NSE_solver_interval == 0))
          {
            this->computing_timer.print_summary();
          }

        time_index += parameters.time_step / parameters.NSE_solver_interval;
        ++timestep_number;

        old_nse_solution = nse_solution;
        if (parameters.use_picard)
          nse_picard_iterate = nse_solution;
        old_temperature_solution = temperature_solution;

        this->pcout << "----------------------------------------" << std::endl;
      }
    while (time_index <= parameters.final_time);

    // for testing
    get_maximal_velocity();

    print_numerical_results("numerical_results.tex");

    this->pcout << "   Cumulative FGMRes iterations: "
                << cumulative_fgmres_iterations << std::endl;
    this->pcout << "   Cumulative FGMRes times (cpu times): "
                << cumulative_fgmres_cpu_times << std::endl;
    this->pcout << "   Cumulative FGMRes times (wall times): "
                << cumulative_fgmres_wall_times << std::endl;
  }


  template <int dim>
  void
  BoussinesqModel<dim>::run_test_rank()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "BoussinesqModel - global run function");

    // initialize table for numerical results
    initialize_numerical_results();

    // call refinement routine in base class
    this->refine_global(parameters.initial_global_refinement);

    setup_dofs();

    // Compute initial time step: scale with the time scale (=l_ref/v_ref) and
    // the minimal cell diameter
    double hK = 1;
    for (const auto &cell : nse_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hK = std::min(hK, cell->diameter());
        }

    print_preconditioner_parameter_info();

    print_paramter_info();

    /*
     * Initial values.
     */
    nse_solution = 0;

    LA::MPI::Vector solution_tmp(temperature_dof_handler.locally_owned_dofs());

    if (parameters.cuboid_geometry)
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<dim>(
            this->center, this->global_Omega_diameter),
          solution_tmp);
      }
    else
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
            parameters.physical_constants.R0, parameters.physical_constants.R1),
          solution_tmp);
      }

    old_nse_solution = nse_solution;
    if (parameters.use_picard)
      nse_picard_iterate = nse_solution;
    temperature_solution     = solution_tmp;
    old_temperature_solution = solution_tmp;

    try
      {
        Tools::create_data_directory(parameters.dirname_output);
      }
    catch (std::runtime_error &e)
      {
        // No exception handling here.
      }

    output_results();

    // Time iteration
    double time_index = 0;
    do
      {
        if ((timestep_number > 0) &&
            (timestep_number % parameters.NSE_solver_interval == 0) &&
            parameters.adapt_time_step)
          {
            recompute_time_step();
          }
        else
          {
            /*
             * This is just informative output
             */
            get_cfl_number();
            get_maximal_velocity();
          }

        this->pcout << "----------------------------------------" << std::endl
                    << "Time step " << timestep_number << ":  t=" << time_index
                    << " -> t=" << time_index + parameters.time_step
                    << "  (dt=" << parameters.time_step
                    << " | final time=" << parameters.final_time << ")"
                    << std::endl;

        if (timestep_number == 0)
          {
            assemble_nse_system(time_index);

            build_nse_preconditioner(time_index);
          }
        else if ((timestep_number > 0) &&
                 (timestep_number % parameters.NSE_solver_interval == 0))
          {
            assemble_nse_system(time_index);

            build_nse_preconditioner(time_index);
          }

        assemble_temperature_matrix(time_index);
        assemble_temperature_rhs(time_index);

        // Solve the Navier-Stokes equations
        solve_nse_test_updaterank(time_index, parameters.time_step);

        // solve temperature advection diffusion equation
        solve_temperature();

        output_results();

        /*
         * Print summary after a NSE system has been solved.
         */
        if ( //(timestep_number > 0) &&
          (timestep_number % parameters.NSE_solver_interval == 0))
          {
            this->computing_timer.print_summary();
          }

        time_index += parameters.time_step / parameters.NSE_solver_interval;
        ++timestep_number;

        old_nse_solution = nse_solution;
        if (parameters.use_picard)
          nse_picard_iterate = nse_solution;
        old_temperature_solution = temperature_solution;

        this->pcout << "----------------------------------------" << std::endl;
      }
    while (time_index <= parameters.final_time);

    // for testing
    get_maximal_velocity();

    print_numerical_results("numerical_results.tex");

    this->pcout << "   Cumulative FGMRes iterations: "
                << cumulative_fgmres_iterations << std::endl;
    this->pcout << "   Cumulative FGMRes times (cpu times): "
                << cumulative_fgmres_cpu_times << std::endl;
    this->pcout << "   Cumulative FGMRes times (wall times): "
                << cumulative_fgmres_wall_times << std::endl;
  }


  template <int dim>
  void
  BoussinesqModel<dim>::run_test_repeated_update()
  {
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "BoussinesqModel - global run function");

    // initialize table for numerical results
    initialize_numerical_results();

    // call refinement routine in base class
    this->refine_global(parameters.initial_global_refinement);

    setup_dofs();

    // Compute initial time step: scale with the time scale (=l_ref/v_ref) and
    // the minimal cell diameter
    double hK = 1;
    for (const auto &cell : nse_dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          hK = std::min(hK, cell->diameter());
        }

    print_preconditioner_parameter_info();

    print_paramter_info();

    /*
     * Initial values.
     */
    nse_solution = 0;

    LA::MPI::Vector solution_tmp(temperature_dof_handler.locally_owned_dofs());

    if (parameters.cuboid_geometry)
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<dim>(
            this->center, this->global_Omega_diameter),
          solution_tmp);
      }
    else
      {
        VectorTools::project(
          temperature_dof_handler,
          temperature_constraints,
          QGauss<dim>(parameters.temperature_degree + 2),
          CoreModelData::Boussinesq::TemperatureInitialValues<dim>(
            parameters.physical_constants.R0, parameters.physical_constants.R1),
          solution_tmp);
      }

    old_nse_solution = nse_solution;
    if (parameters.use_picard)
      nse_picard_iterate = nse_solution;
    temperature_solution     = solution_tmp;
    old_temperature_solution = solution_tmp;

    try
      {
        Tools::create_data_directory(parameters.dirname_output);
      }
    catch (std::runtime_error &e)
      {
        // No exception handling here.
      }

    output_results();

    // Time iteration
    double time_index = 0;
    do
      {
        if ((timestep_number > 0) &&
            (timestep_number % parameters.NSE_solver_interval == 0) &&
            parameters.adapt_time_step)
          {
            recompute_time_step();
          }
        else
          {
            /*
             * This is just informative output
             */
            get_cfl_number();
            get_maximal_velocity();
          }

        this->pcout << "----------------------------------------" << std::endl
                    << "Time step " << timestep_number << ":  t=" << time_index
                    << " -> t=" << time_index + parameters.time_step
                    << "  (dt=" << parameters.time_step
                    << " | final time=" << parameters.final_time << ")"
                    << std::endl;

        if (timestep_number == 0)
          {
            assemble_nse_system(time_index);

            build_nse_preconditioner(time_index);
          }
        else if ((timestep_number > 0) &&
                 (timestep_number % parameters.NSE_solver_interval == 0))
          {
            assemble_nse_system(time_index);

            build_nse_preconditioner(time_index);
          }

        assemble_temperature_matrix(time_index);
        assemble_temperature_rhs(time_index);

        // Solve the Navier-Stokes equations
        solve_nse_test_repeat_update(time_index, parameters.time_step);

        // solve temperature advection diffusion equation
        solve_temperature();

        output_results();

        /*
         * Print summary after a NSE system has been solved.
         */
        if ( //(timestep_number > 0) &&
          (timestep_number % parameters.NSE_solver_interval == 0))
          {
            this->computing_timer.print_summary();
          }

        time_index += parameters.time_step / parameters.NSE_solver_interval;
        ++timestep_number;

        old_nse_solution = nse_solution;
        if (parameters.use_picard)
          nse_picard_iterate = nse_solution;
        old_temperature_solution = temperature_solution;

        this->pcout << "----------------------------------------" << std::endl;
      }
    while (time_index <= parameters.final_time);

    // for testing
    get_maximal_velocity();

    print_numerical_results("numerical_results.tex");

    this->pcout << "   Cumulative FGMRes iterations: "
                << cumulative_fgmres_iterations << std::endl;
    this->pcout << "   Cumulative FGMRes times (cpu times): "
                << cumulative_fgmres_cpu_times << std::endl;
    this->pcout << "   Cumulative FGMRes times (wall times): "
                << cumulative_fgmres_wall_times << std::endl;
  }

} // namespace Standard

DYCOREPLANET_CLOSE_NAMESPACE
