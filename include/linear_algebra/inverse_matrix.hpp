/**
 * @file inverse_matrix.hpp
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

// Deal.ii
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/solver.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

// STL
#include <memory>

// AquaPlanet
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class InverseMatrix
   *
   * @brief Implements an iterative inverse
   *
   * Implement the inverse matrix of a given matrix through
   * its action by a preconditioned CG solver. This class also
   * works with MPI.
   *
   * @note The inverse is not constructed explicitly.
   */
  template <typename MatrixType, typename PreconditionerType>
  class InverseMatrix : public Subscriptor
  {
  public:
    /*!
     * Constructor.
     *
     * @param m
     * @param preconditioner
     */
    // InverseMatrix(const MatrixType &        m,
    //               const PreconditionerType &preconditioner,
    InverseMatrix(MatrixType &        m,
                  PreconditionerType &preconditioner,
                  bool                use_simple_cg           = true,
                  bool                print_needed_iterations = false);

    /*!
     * Matrix-vector multiplication.
     *
     * @param[out] dst
     * @param[in] src
     */
    template <typename VectorType>
    void
    vmult(VectorType &dst, const VectorType &src) const;

    template <typename VectorType>
    void
    Tvmult(VectorType &dst, const VectorType &src) const;

    void
    transpose() const; // const bool use_transpose);

  private:
    /*!
     * Samrt pointer to system matrix.
     */
    // const SmartPointer<const MatrixType> matrix;
    const SmartPointer<MatrixType> matrix;

    /*!
     * Preconditioner.
     */
    // const PreconditionerType &preconditioner;
    PreconditionerType &preconditioner;

    const bool use_simple_cg;

    const bool print_needed_iterations;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename MatrixType, typename PreconditionerType>
  InverseMatrix<MatrixType, PreconditionerType>::InverseMatrix(
    // const MatrixType &        m,
    // const PreconditionerType &preconditioner,
    MatrixType &        m,
    PreconditionerType &preconditioner,
    bool                use_simple_cg,
    bool                print_needed_iterations)
    : matrix(&m)
    , preconditioner(preconditioner)
    , use_simple_cg(use_simple_cg)
    , print_needed_iterations(print_needed_iterations)
  {}

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  void
  InverseMatrix<MatrixType, PreconditionerType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    SolverControl solver_control(std::max(static_cast<std::size_t>(src.size()),
                                          static_cast<std::size_t>(1000)),
                                 1e-6 * src.l2_norm());

    dst = 0;

    try
      {
        if (use_simple_cg)
          {
            SolverCG<VectorType> local_solver(solver_control);
            local_solver.solve(*matrix, dst, src, preconditioner);
          }
        else
          {
            SolverGMRES<VectorType> local_solver(solver_control);
            local_solver.solve(*matrix, dst, src, preconditioner);
          }
      }
    catch (std::exception &e)
      {
        Assert(false, ExcMessage(e.what()));
      }

    if (print_needed_iterations)
      std::cout << " Needed " << solver_control.last_step() << " iterations."
                << std::endl;
  }

  template <typename MatrixType, typename PreconditionerType>
  template <typename VectorType>
  void
  InverseMatrix<MatrixType, PreconditionerType>::Tvmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    SolverControl solver_control(std::max(static_cast<std::size_t>(src.size()),
                                          static_cast<std::size_t>(1000)),
                                 1e-6 * src.l2_norm());

    dst = 0;

    transpose();
    try
      {
        if (use_simple_cg)
          {
            SolverCG<VectorType> local_solver(solver_control);
            local_solver.solve(*matrix, dst, src, preconditioner);
          }
        else
          {
            SolverGMRES<VectorType> local_solver(solver_control);
            local_solver.solve(*matrix, dst, src, preconditioner);
          }
      }
    catch (std::exception &e)
      {
        Assert(false, ExcMessage(e.what()));
      }
    transpose();

    if (print_needed_iterations)
      std::cout << " Needed " << solver_control.last_step() << " iterations."
                << std::endl;
  }



  template <typename MatrixType, typename PreconditionerType>
  void
  InverseMatrix<MatrixType, PreconditionerType>::transpose() const
  // const bool use_transpose)
  {
    matrix->transpose(); // trilinos_matrix()->SetUseTranspose(use_transpose);
    preconditioner.transpose(); // SetUseTranspose(use_transpose);
  }

} // end namespace LinearAlgebra

DYCOREPLANET_CLOSE_NAMESPACE
