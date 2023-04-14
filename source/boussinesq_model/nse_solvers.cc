/**
 * @file nse_solvers.cc
 * @author Rebekka Beddig
 * @version 0.1
 */
#include <core/boussinesq_model_commutator.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace Standard
{
  /**
   * Solver for the Navier-Stokes equations. Calls the solver that was chosen in
   * the parameter file.
   */
  template <int dim>
  void
  BoussinesqModel<dim>::solve_nse(const double time_index,
                                  const double time_step)
  {
    std::ignore = time_step;
    if (parameters.use_picard)
      {
        const unsigned int max_picard_iters = 10;
        bool               converged        = false;

        for (picarditeration_number = 0;
             picarditeration_number < max_picard_iters;)
          {
            // Solve
            if (parameters.use_schur_complement_solver)
              solve_NSE_Schur_complement();
            else
              solve_NSE_block_preconditioned();

            // correct pressure to zero mean to obtain a unique solution
            if (parameters.correct_pressure_to_zero_mean)
              normalize_pressure(nse_solution);

            // Compute the difference between the previous and the current
            // solution to check if the Picard iteration converges
            LA::MPI::BlockVector difference(nse_solution);
            difference = nse_solution;
            difference.add(-1., nse_picard_iterate);
            this->pcout << "      ||u_k - u_(k-1)|| = " << difference.l2_norm()
                        << std::endl;

            // Check for convergence
            double relative_improvement =
              difference.l2_norm() / nse_solution.l2_norm();
            if (relative_improvement < 1e-8)
              {
                converged = true;
                this->pcout << "   The Picard iteration converged after "
                            << picarditeration_number << " iterations."
                            << std::endl;
              }

            // Leave the iteration if we have converged
            if (converged)
              break;

            // else: Assemble new advection and compute the new system matrix
            nse_picard_iterate = nse_solution;

            // Set Picard iterate to current solution and use it for the new
            // assembly
            assemble_nse_system(time_index);

            picarditeration_number += 1;


            // build the preconditioners for A and S
            build_nse_preconditioner(time_index);
          }
      }
    else
      {
        if (parameters.use_direct_solver)
          {
            TimerOutput::Scope t(this->computing_timer,
                                 " direct solver (MUMPS)");

            throw std::runtime_error(
              "Solver not implemented: MUMPS does not work on "
              "TrilinosWrappers::MPI::BlockSparseMatrix classes.");
          }

        // Schur complement solver
        if (parameters.use_schur_complement_solver)
          {
            solve_NSE_Schur_complement();
          }
        else
          {
            // Block solver
            solve_NSE_block_preconditioned();
          }
      }

    // set the picard-iterate number to 0 for the next time step
    picarditeration_number = 0;

    // correct pressure to zero mean to obtain a unique solution
    if (parameters.correct_pressure_to_zero_mean)
      normalize_pressure(nse_solution);

    // print info on solution
    this->pcout << "      ||u||_2 = " << nse_solution.block(0).l2_norm()
                << std::endl;
    this->pcout << "      ||p||_2 = " << nse_solution.block(1).l2_norm()
                << std::endl;
  }


  /**
   * Solver for the Navier-Stokes equations. Calls the solver that was chosen in
   * the parameter file.
   */
  template <int dim>
  void
  BoussinesqModel<dim>::solve_nse_test_updaterank(const double time_index,
                                                  const double time_step)
  {
    std::ignore = time_step;
    if (parameters.use_picard)
      {
        const unsigned int max_picard_iters = 5;
        bool               converged        = false;
        std::ignore                         = converged;

        for (picarditeration_number = 0;
             picarditeration_number < max_picard_iters;)
          {
            // Solve
            if (parameters.use_schur_complement_solver)
              solve_NSE_Schur_complement();
            else
              solve_NSE_block_preconditioned();

            // correct pressure to zero mean to obtain a unique solution
            if (parameters.correct_pressure_to_zero_mean)
              normalize_pressure(nse_solution);

            // Compute the difference between the previous and the current
            // solution to check if the Picard iteration converges
            LA::MPI::BlockVector difference(nse_solution);
            difference = nse_solution;
            difference.add(-1., nse_picard_iterate);
            this->pcout << "      ||u_k - u_(k-1)|| = " << difference.l2_norm()
                        << std::endl;

            // Check for convergence
            // double relative_improvement =
            //   difference.l2_norm() / nse_solution.l2_norm();

            // if (relative_improvement < 1e-8)
            //   {
            //     converged = true;
            //     this->pcout << "   The Picard iteration converged after "
            //                 << picarditeration_number << " iterations."
            //                 << std::endl;
            //   }

            // Leave the iteration if we have converged
            // if (converged)
            //   break;

            // else: Assemble new advection and compute the new system matrix
            nse_picard_iterate = nse_solution;

            // Set Picard iterate to current solution and use it for the new
            // assembly
            assemble_nse_system(time_index);

            picarditeration_number += 1;


            // build the preconditioners for A and S
            build_nse_preconditioner(time_index);
          }
      }
    else
      {
        if (parameters.use_direct_solver)
          {
            TimerOutput::Scope t(this->computing_timer,
                                 " direct solver (MUMPS)");

            throw std::runtime_error(
              "Solver not implemented: MUMPS does not work on "
              "TrilinosWrappers::MPI::BlockSparseMatrix classes.");
          }

        // Schur complement solver
        if (parameters.use_schur_complement_solver)
          {
            solve_NSE_Schur_complement();
          }
        else
          {
            // Block solver
            solve_NSE_block_preconditioned();
          }
      }

    // Solve without update
    // parameters.rank = 0;
    build_nse_preconditioner(time_index);
    numerical_results.add_value("rank", parameters.rank);
    // nse_solution = nse_picard_iterate;
    if (parameters.use_schur_complement_solver)
      solve_NSE_Schur_complement();
    else
      solve_NSE_block_preconditioned();
    // Solve the same system again,
    //  but now with the updated preconditioner

    for (unsigned int rank = 0; rank <= 60; rank += 15)
      {
        parameters.rank              = rank;
        rebuild_nse_preconditioner   = true;
        rebuild_A_preconditioner     = false;
        rebuild_schur_preconditioner = false;
        build_nse_preconditioner(time_index);
        numerical_results.add_value("rank", parameters.rank);
        try
          {
            if (parameters.use_schur_complement_solver)
              solve_NSE_Schur_complement();
            else
              solve_NSE_block_preconditioned();
          }
        catch (SolverControl::NoConvergence &)
          {}
      }

    // set the picard-iterate number to 0 for the next time step
    picarditeration_number = 0;

    // correct pressure to zero mean to obtain a unique solution
    if (parameters.correct_pressure_to_zero_mean)
      normalize_pressure(nse_solution);

    // print info on solution
    this->pcout << "      ||u||_2 = " << nse_solution.block(0).l2_norm()
                << std::endl;
    this->pcout << "      ||p||_2 = " << nse_solution.block(1).l2_norm()
                << std::endl;
  }


  /**
   * Solver for the Navier-Stokes equations. Calls the solver that was chosen in
   * the parameter file.
   */
  template <int dim>
  void
  BoussinesqModel<dim>::solve_nse_test_repeat_update(const double time_index,
                                                     const double time_step)
  {
    std::ignore = time_step;
    if (parameters.use_picard)
      {
        const unsigned int max_picard_iters = 5;
        bool               converged        = false;
        std::ignore                         = converged;

        for (picarditeration_number = 0;
             picarditeration_number < max_picard_iters;)
          {
            // Solve
            if (parameters.use_schur_complement_solver)
              solve_NSE_Schur_complement();
            else
              solve_NSE_block_preconditioned();

            // correct pressure to zero mean to obtain a unique solution
            if (parameters.correct_pressure_to_zero_mean)
              normalize_pressure(nse_solution);

            // Compute the difference between the previous and the current
            // solution to check if the Picard iteration converges
            LA::MPI::BlockVector difference(nse_solution);
            difference = nse_solution;
            difference.add(-1., nse_picard_iterate);
            this->pcout << "      ||u_k - u_(k-1)|| = " << difference.l2_norm()
                        << std::endl;

            // Check for convergence
            // double relative_improvement =
            //   difference.l2_norm() / nse_solution.l2_norm();

            // if (relative_improvement < 1e-8)
            //   {
            //     converged = true;
            //     this->pcout << "   The Picard iteration converged after "
            //                 << picarditeration_number << " iterations."
            //                 << std::endl;
            //   }

            // Leave the iteration if we have converged
            // if (converged)
            //   break;

            // else: Assemble new advection and compute the new system matrix
            nse_picard_iterate = nse_solution;

            // Set Picard iterate to current solution and use it for the new
            // assembly
            assemble_nse_system(time_index);

            picarditeration_number += 1;


            // build the preconditioners for A and S
            build_nse_preconditioner(time_index);
          }
      }
    else
      {
        if (parameters.use_direct_solver)
          {
            TimerOutput::Scope t(this->computing_timer,
                                 " direct solver (MUMPS)");

            throw std::runtime_error(
              "Solver not implemented: MUMPS does not work on "
              "TrilinosWrappers::MPI::BlockSparseMatrix classes.");
          }

        // Schur complement solver
        if (parameters.use_schur_complement_solver)
          {
            solve_NSE_Schur_complement();
          }
        else
          {
            // Block solver
            solve_NSE_block_preconditioned();
          }
      }

    // Solve without update
    // parameters.rank = 0;
    build_nse_preconditioner(time_index);
    numerical_results.add_value("rank", parameters.rank);
    // nse_solution = nse_picard_iterate;
    if (parameters.use_schur_complement_solver)
      solve_NSE_Schur_complement();
    else
      solve_NSE_block_preconditioned();
    // Solve the same system again,
    //  but now with the updated preconditioner

    for (unsigned int rank = 0; rank <= 60; rank += 30)
      {
        parameters.rank              = rank;
        parameters.number            = 1;
        rebuild_nse_preconditioner   = true;
        rebuild_A_preconditioner     = false;
        rebuild_schur_preconditioner = false;
        build_nse_preconditioner(time_index);
        numerical_results.add_value("rank", parameters.rank);
        try
          {
            if (parameters.use_schur_complement_solver)
              solve_NSE_Schur_complement();
            else
              solve_NSE_block_preconditioned();
          }
        catch (SolverControl::NoConvergence &)
          {}
      }

    {
      parameters.rank              = 30;
      parameters.number            = 2;
      rebuild_nse_preconditioner   = true;
      rebuild_A_preconditioner     = false;
      rebuild_schur_preconditioner = false;
      build_nse_preconditioner(time_index);
      numerical_results.add_value("rank", parameters.rank);
      try
        {
          if (parameters.use_schur_complement_solver)
            solve_NSE_Schur_complement();
          else
            solve_NSE_block_preconditioned();
        }
      catch (SolverControl::NoConvergence &)
        {}
    }

    // set the picard-iterate number to 0 for the next time step
    picarditeration_number = 0;

    // correct pressure to zero mean to obtain a unique solution
    if (parameters.correct_pressure_to_zero_mean)
      normalize_pressure(nse_solution);

    // print info on solution
    this->pcout << "      ||u||_2 = " << nse_solution.block(0).l2_norm()
                << std::endl;
    this->pcout << "      ||p||_2 = " << nse_solution.block(1).l2_norm()
                << std::endl;
  }


} // namespace Standard

DYCOREPLANET_CLOSE_NAMESPACE

// explicit instantiations of the functions that are implemented in this file
DYCOREPLANET_OPEN_NAMESPACE
namespace Standard
{
  template void
  BoussinesqModel<2>::solve_nse(const double, const double);
  template void
  BoussinesqModel<3>::solve_nse(const double, const double);

  template void
  BoussinesqModel<2>::solve_nse_test_updaterank(const double, const double);
  template void
  BoussinesqModel<3>::solve_nse_test_updaterank(const double, const double);

  template void
  BoussinesqModel<2>::solve_nse_test_repeat_update(const double, const double);
  template void
  BoussinesqModel<3>::solve_nse_test_repeat_update(const double, const double);
} // namespace Standard
DYCOREPLANET_CLOSE_NAMESPACE
