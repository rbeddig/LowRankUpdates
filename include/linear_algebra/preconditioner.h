/**
 * @file preconditioner.h
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

// Deal.ii
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

// My headers
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class InnerSchurPreconditioner
   *
   * @brief Inner preconditioner type for Schur complement solver
   *
   * Encapsulation of preconditioner type used for the inner
   * matrix in a Schur complement. Works with MPI.
   */
  class InnerSchurPreconditioner
  {
  public:
    // using type = LA::PreconditionAMG;
    // using type = LA::PreconditionILU;
    using type = LA::PreconditionJacobi;
    //	  using type = LA::PreconditionIdentity;
  };

} // namespace LinearAlgebra


DYCOREPLANET_CLOSE_NAMESPACE
