/**
 * @file nse_preconditioners.cc
 * @author Rebekka Beddig
 * @version 0.1
 */
// deal.II
#include <deal.II/base/index_set.h>

// AquaPlanet
#include <core/boussinesq_model_commutator.h>
#include <linear_algebra/inverse_matrix.hpp>

// preconditioners
#include <preconditioners/updates2.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace Standard
{
  /**
   * @brief: Fill the AdditionalData structs of the preconditioners for the
   * upper left block and the Schur complement.
   */
  template <int dim>
  void
  BoussinesqModel<dim>::setup_preconditioner_data()
  {
    ////////////////////////////////////////////////////////
    // Setup data for the preconditioner of the upper left block
    ////////////////////////////////////////////////////////
    if (parameters.block00_preconditioner_name == "amg")
      {
        typename LA::PreconditionAMG::AdditionalData AMG_preconditioner_data;
        // data for AMG
        // determine constant modes for the AMG
        std::vector<std::vector<bool>> constant_modes;
        FEValuesExtractors::Vector     velocity_components(0);
        DoFTools::extract_constant_modes(nse_dof_handler,
                                         nse_fe.component_mask(
                                           velocity_components),
                                         constant_modes);
        AMG_preconditioner_data.constant_modes = constant_modes;
        if (parameters.use_picard)
          {
            AMG_preconditioner_data.elliptic = false;
            // AMG_preconditioner_data.w_cycle         = true;
            AMG_preconditioner_data.smoother_type   = "Gauss-Seidel";
            AMG_preconditioner_data.smoother_sweeps = 2; // 2;
            AMG_preconditioner_data.n_cycles        = 1;
          }
        else
          {
            AMG_preconditioner_data.elliptic        = true;
            AMG_preconditioner_data.smoother_sweeps = 2;
          }
        AMG_preconditioner_data.higher_order_elements = true;
        AMG_preconditioner_data.aggregation_threshold =
          parameters.AMG_aggregation_threshold; // default: 0.02
        if (parameters.solver_diagnostics_print_level == 2)
          AMG_preconditioner_data.output_details = true;
        block00_preconditioner->set_data(AMG_preconditioner_data);
      }
    else if (parameters.block00_preconditioner_name == "ilu")
      {
        LA::PreconditionILU::AdditionalData ilu_preconditioner_data;
        block00_preconditioner->set_data(ilu_preconditioner_data);
      }
    else if (parameters.block00_preconditioner_name == "jacobi")
      { // no data needed here
        // block00_preconditioner.set_data(jacobi_data);
      }

    std::map<std::string, dealii::Preconditioners::SVDSolverType>
      svd_solver_map;
    svd_solver_map["random"]  = dealii::Preconditioners::SVDSolverType::random;
    svd_solver_map["arnoldi"] = dealii::Preconditioners::SVDSolverType::arnoldi;


    ////////////////////////////////////////////////////////
    // set up the data for the Schur complement preconditioner
    ////////////////////////////////////////////////////////
    if (parameters.schur_preconditioner_name == "bfbt")
      {
        dealii::Preconditioners::TrilinosBFBt::AdditionalData bfbt_options;
        bfbt_options.use_scaling = parameters.bfbt_scale;
        bfbt_options.scale_with_inv_row_sum =
          false; // parameters.scale_with_velocity_mass;
        bfbt_options.AMG_aggregation_threshold =
          parameters.bfbt_aggregation_threshold;
        bfbt_options.tolerance = parameters.tolerance;

        if (parameters.only_amg)
          bfbt_options.solver_type = Preconditioners::PoissonTypeSolver::AMG;
        else if (parameters.use_ic)
          bfbt_options.solver_type = Preconditioners::PoissonTypeSolver::IC;
        else if (parameters.precondition_poisson)
          bfbt_options.solver_type =
            Preconditioners::PoissonTypeSolver::PCG_AMG;
        else
          bfbt_options.solver_type = Preconditioners::PoissonTypeSolver::CG;

        if ((parameters.only_dirichlet || !parameters.cuboid_geometry) &&
            parameters.remove_nullspace)
          {
            const unsigned int p_fixed =
              p_fixed_idx - nse_matrix.block(0, 0).n();
            std::vector<unsigned int> constrained_p_dofs(1, p_fixed);
            bfbt_options.constrained_dofs = constrained_p_dofs;
          }
        else
          {
            // get all constrained indices
            IndexSet constrained_index_set;
            constrained_index_set = nse_constraints.get_local_lines();

            // extract only constrained pressure dofs
            IndexSet pressure_filter(nse_matrix.n());
            pressure_filter.add_range(nse_matrix.block(0, 0).n(),
                                      nse_matrix.n());

            const unsigned int n_p =
              nse_matrix.n() - nse_matrix.block(0, 0).n();
            AffineConstraints<double> pressure_constraints(
              complete_index_set(n_p));
            pressure_constraints.add_selected_constraints(nse_constraints,
                                                          pressure_filter);
            pressure_constraints.close();
            constrained_index_set.clear();
            AffineConstraints<double>::LineRange range =
              pressure_constraints.get_lines();
            for (auto &r : range)
              {
                constrained_index_set.add_index(r.index);
              }

            // fill std::vector with constrained pressure dofs
            std::vector<unsigned int> constrained_indices;
            constrained_index_set.fill_index_vector(constrained_indices);

            bfbt_options.constrained_dofs = constrained_indices;
          }


        schur_preconditioner->set_data(bfbt_options);


        // Data for the low-rank update
        using UpdateType = dealii::Preconditioners::LowRankUpdates<
          LinearAlgebra::ApproxSchurComplement<
            TrilinosWrappers::BlockSparseMatrix,
            TrilinosWrappers::PreconditionILU>,
          // dealii::Preconditioners::PreconditionerSelector>,
          dealii::Preconditioners::TrilinosBFBt>;
        UpdateType::AdditionalData update_data;
        if (picarditeration_number < 1 && parameters.use_picard)
          update_data.rank = 0;
        else
          update_data.rank = parameters.rank;
        update_data.solver_type = svd_solver_map[parameters.svd_solver_type];
        update_data.number      = parameters.number;
        update_data.symmetric   = false;
        update_data.weight      = parameters.weight;
        schur_preconditioner->set_data(update_data);
      }
    else if (parameters.schur_preconditioner_name ==
             "orig") // CG( B Jacobi(A) B^^T )
      {              // no parameters
        if ((parameters.only_dirichlet || !parameters.cuboid_geometry) &&
            parameters.remove_nullspace)
          {
            const unsigned int p_fixed =
              p_fixed_idx - nse_matrix.block(0, 0).n();
            std::vector<unsigned int> constrained_p_dofs(1, p_fixed);
            dealii::Preconditioners::OrigSchurPreconditioner::AdditionalData
              orig_data;
            orig_data.constrained_dofs    = constrained_p_dofs;
            orig_data.use_ic              = parameters.use_ic;
            orig_data.tolerance           = parameters.tolerance;
            orig_data.solve_only_with_amg = parameters.only_amg;
            orig_data.precondition_cg     = parameters.precondition_poisson;
            schur_preconditioner->set_data(orig_data);
          }
        else
          {
            // get all constrained indices
            IndexSet constrained_index_set;
            constrained_index_set = nse_constraints.get_local_lines();

            // extract only constrained pressure dofs
            IndexSet pressure_filter(nse_matrix.n());
            pressure_filter.add_range(nse_matrix.block(0, 0).n(),
                                      nse_matrix.n());

            const unsigned int n_p =
              nse_matrix.n() - nse_matrix.block(0, 0).n();
            AffineConstraints<double> pressure_constraints(
              complete_index_set(n_p));
            pressure_constraints.add_selected_constraints(nse_constraints,
                                                          pressure_filter);
            pressure_constraints.close();
            constrained_index_set.clear();
            AffineConstraints<double>::LineRange range =
              pressure_constraints.get_lines();
            for (auto &r : range)
              {
                constrained_index_set.add_index(r.index);
              }

            // fill std::vector with constrained pressure dofs
            std::vector<unsigned int> constrained_indices;
            constrained_index_set.fill_index_vector(constrained_indices);

            dealii::Preconditioners::OrigSchurPreconditioner::AdditionalData
              orig_data;
            orig_data.constrained_dofs    = constrained_indices;
            orig_data.use_ic              = parameters.use_ic;
            orig_data.tolerance           = parameters.tolerance;
            orig_data.solve_only_with_amg = parameters.only_amg;
            orig_data.precondition_cg     = parameters.precondition_poisson;
            schur_preconditioner->set_data(orig_data);
          }

        using UpdateType = dealii::Preconditioners::LowRankUpdates<
          LinearAlgebra::ApproxSchurComplement<
            TrilinosWrappers::BlockSparseMatrix,
            TrilinosWrappers::PreconditionILU>,
          dealii::Preconditioners::OrigSchurPreconditioner>;
        UpdateType::AdditionalData update_data;
        if (picarditeration_number < 1 && parameters.use_picard)
          update_data.rank = 0;
        else
          update_data.rank = parameters.rank;
        update_data.solver_type = svd_solver_map[parameters.svd_solver_type];
        update_data.number      = parameters.number;
        update_data.symmetric   = false;
        update_data.weight      = parameters.weight;
        schur_preconditioner->set_data(update_data);
      }

    /*
     *  Print preconditioner parameters to table.
     */
    numerical_results.add_value("preconditioner",
                                parameters.schur_preconditioner_name == "orig" ?
                                  "SIMPLE" :
                                  "LSC");
    numerical_results.add_value("number", parameters.number);
  }



  /**
   * @brief: Initialize the preconditioners for the upper left block and the
   * Schur complement.
   */
  template <int dim>
  void
  BoussinesqModel<dim>::initialize_preconditioner()
  {
    //////////////////////////////////////////
    // initialize the block00 preconditioner
    //////////////////////////////////////////
    // AMG:    it is initialized either with the velocity mass matrix or with
    //         the upper left block of the NSE preconditioner matrix
    // Jacobi/ILU: it is initialized with the upper left block of the NSE system
    //         matrix
    Timer              timer;
    TimerOutput::Scope timer_section(this->computing_timer,
                                     "   Initialize NSE preconditioner");
    if (rebuild_A_preconditioner)
      {
        TimerOutput::Scope timer_section_A(
          this->computing_timer, "   Initialize velocity block preconditioner");
        timer.start();
        if (parameters.block00_preconditioner_name == "amg")
          if (parameters.use_Mu)
            {
              Assert(false, ExcNotImplemented());
            }
          else
            {
              block00_preconditioner->initialize(
                nse_preconditioner_matrix.block(0, 0));
            }
        else if (parameters.block00_preconditioner_name == "ilu")
          {
            // inner_A_preconditioner->initialize(nse_matrix.block(0, 0));
            // block00_preconditioner->select(parameters.block00_preconditioner_name);
            block00_preconditioner->initialize(nse_matrix.block(0, 0));
          }
        else if (parameters.block00_preconditioner_name == "jacobi")
          block00_preconditioner->initialize(nse_matrix.block(0, 0));

        timer.stop();
      }

    this->pcout << "      initialize block(0,0)-preconditioner: "
                << timer.cpu_time() << std::endl;

    rebuild_A_preconditioner = false;

    //////////////////////////////////////////////////
    // initialize the Schur complement preconditioner
    //////////////////////////////////////////////////
    // Options are:
    // BFBt: either with or without scaling, the scaling can be done with
    // the
    //       velocity mass matrix or with the upper left block of the NSE
    //       system matrix
    // orig: SIMPLE preconditioner: the Schur complement is approximated
    //       with B diag(A)^-1 B^T and then
    //       inverted with a Krylov subspace solver
    if (rebuild_schur_preconditioner)
      {
        TimerOutput::Scope timer_section(
          this->computing_timer,
          "   Initialize Schur complement preconditioner");
        timer.restart();
        if (parameters.schur_preconditioner_name == "bfbt")
          if (parameters.bfbt_scale)
            if (parameters.scale_with_velocity_mass)
              {
                // Assert(false, ExcNotImplemented());
                assemble_velocity_mass_matrix(0.0);
                schur_preconditioner->initialize(nse_mass_matrix.block(0, 0));
              }
            else
              schur_preconditioner->initialize(nse_matrix.block(0, 0));
          else
            {
              schur_preconditioner->initialize();
            }
        else if (parameters.schur_preconditioner_name == "orig")
          schur_preconditioner->initialize();
        timer.stop();
        this->pcout << "      initialize schur complement preconditioner: "
                    << timer.cpu_time() << std::endl;

        rebuild_schur_preconditioner = false;
      }
    else
      {
        // update pointer to A
        schur_preconditioner->reinit_A(nse_matrix.block(0, 0));
      }



    /////////////////////////////////////////////////////
    // Initialize Schur complement preconditioner update
    /////////////////////////////////////////////////////
    {
      TimerOutput::Scope timer_section(this->computing_timer,
                                       "   Initialize update");
      timer.restart();
      if (parameters.rank != 0 &&
          ((parameters.use_picard && picarditeration_number == 5) ||
           !parameters.use_picard))
        {
          TrilinosWrappers::PreconditionILU::AdditionalData ilu_data(0,
                                                                     0.,
                                                                     1.,
                                                                     0);
          inner_A_preconditioner->initialize(nse_matrix.block(0, 0), ilu_data);

          if (parameters.schur_preconditioner_name == "bfbt")
            {
              using InverseType = LA::PreconditionBase;

              using SchurComplementType =
                LinearAlgebra::ApproxSchurComplement<LA::BlockSparseMatrix,
                                                     InverseType>;
              approx_schur_complement = std::make_shared<SchurComplementType>(
                nse_matrix,
                *inner_A_preconditioner,
                // block00_preconditioner->dealii_preconditioner(),
                nse_partitioning,
                this->mpi_communicator);

              schur_preconditioner->initialize_update(*approx_schur_complement,
                                                      nse_constraints);
            }
          else if (parameters.schur_preconditioner_name == "orig")
            {
              using InverseType = LA::PreconditionBase;

              using SchurComplementType =
                LinearAlgebra::ApproxSchurComplement<LA::BlockSparseMatrix,
                                                     InverseType>;
              approx_schur_complement = std::make_shared<SchurComplementType>(
                nse_matrix,
                // block00_preconditioner->dealii_preconditioner(),
                *inner_A_preconditioner,
                nse_partitioning,
                this->mpi_communicator);

              schur_preconditioner->initialize_update(*approx_schur_complement,
                                                      nse_constraints);
            }
        }
      timer.stop();
      this->pcout << "      initialize update: " << timer.cpu_time()
                  << std::endl;

      numerical_results.add_value("t_setup", timer.cpu_time());
    }
  }
} // namespace Standard

DYCOREPLANET_CLOSE_NAMESPACE

// explicit instantiations of the functions that are implemented in this file
DYCOREPLANET_OPEN_NAMESPACE
namespace Standard
{
  template void
  BoussinesqModel<2>::setup_preconditioner_data();
  template void
  BoussinesqModel<3>::setup_preconditioner_data();

  template void
  BoussinesqModel<2>::initialize_preconditioner();
  template void
  BoussinesqModel<3>::initialize_preconditioner();
} // namespace Standard
DYCOREPLANET_CLOSE_NAMESPACE
