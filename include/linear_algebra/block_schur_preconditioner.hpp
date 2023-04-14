/**
 * @file block_schur_preconditioner.hpp
 * @author Konrad Simon, Rebekka Beddig
 * @version 0.1
 */
#pragma once

// deal.II
#include <deal.II/base/timer.h>

// AquaPlanet
#include <base/config.h>
#include <linear_algebra/schur_complement.hpp>

// preconditioners
#include <preconditioners/wrappers.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /**
   * @class BlockSchurapproxPrecondtioner using a Schur complement approximation
   * such as the BFBt method.
   *
   * @brief Right preconditioner for FGMRES.
   *
   * @tparam PreconditiointerTypeA: preconditioner for upper left block of the system
   * @tparam PreconditionerTypeS: preconditioner for the Schur complement
   *
   */
  template <class PreconditionerTypeA, class PreconditionerTypeS>
  class BlockSchurapproxPreconditioner : public Subscriptor
  {
  public:
    BlockSchurapproxPreconditioner(const LA::BlockSparseMatrix &nse_matrix,
                                   const PreconditionerTypeA &A_preconditioner,
                                   const PreconditionerTypeS &S_preconditioner,
                                   TimerOutput &              timer_output)
      : nse_matrix(nse_matrix)
      , A_preconditioner(&A_preconditioner)
      , S_preconditioner(&S_preconditioner)
      , timer_output(timer_output)
    {}

    ~BlockSchurapproxPreconditioner()
    {}

    void
    vmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      LA::MPI::Vector utmp(src.block(0));

      // Approximate - S^-1 v_1 (src = [v_0 \\ v_1 ])
      {
        TimerOutput::Scope timer_section(
          timer_output, "   Apply Schur complement preconditioner");
        // S_preconditioner is an approximation of (B A^-1 B^T)^-1
        S_preconditioner->vmult(dst.block(1), src.block(1));
      }
      dst.block(1) *= -1.0;

      // Compute - B^T S^-1 v_1
      nse_matrix.block(0, 1).vmult(utmp, dst.block(1));
      utmp *= -1.0;

      // Compute v_0 + B^T S^-1 v_1
      utmp += src.block(0);

      // Approximate A^-1 (v_0 - B^T S^-1 v_1)
      {
        TimerOutput::Scope timer_section(timer_output,
                                         "   Apply A preconditioner");
        A_preconditioner->vmult(dst.block(0), utmp);
      }
    }


    void
    Tvmult(LA::MPI::BlockVector &dst, const LA::MPI::BlockVector &src) const
    {
      LA::MPI::Vector ptmp(src.block(1));

      // Approximate S^-1 v_1 (src = [v_0 \\ v_1 ])
      {
        TimerOutput::Scope timer_section(timer_output,
                                         "   Apply velocity preconditioner");
        // S_preconditioner is an approximation of (B A^-1 B^T)^-1
        A_preconditioner->Tvmult(dst.block(0), src.block(0));
      }

      // Compute B A^-1 v_0
      nse_matrix.block(1, 0).vmult(ptmp, dst.block(0));

      // Compute v_1 - B A^-1 v_0
      ptmp -= src.block(1);

      // Approximate A^-1 (v_0 - B^T S^-1 v_1)
      {
        TimerOutput::Scope timer_section(timer_output,
                                         "   Apply S preconditioner");
        S_preconditioner->Tvmult(dst.block(1), ptmp);
      }
    }


  private:
    Preconditioners::BlockWrapperMatrix           nse_matrix;
    const SmartPointer<const PreconditionerTypeA> A_preconditioner;
    const SmartPointer<const PreconditionerTypeS> S_preconditioner;

    mutable TimerOutput timer_output;
  };

} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
