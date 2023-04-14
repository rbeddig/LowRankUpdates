/**
 * @file utilities.h
 * @author Rebekka Beddig, adapted from deal.II code
 * @version 0.1
 */
#ifndef OWN_UTILITIES
#define OWN_UTILITIES

// deal.II
#include <deal.II/base/exceptions.h>

#include <deal.II/lac/arpack_solver.h>

// Trilinos
#include <EpetraExt_MatrixMatrix.h>
#include <Teuchos_RCP.hpp>


// preconditioners
#include <preconditioners/constrained_identity_matrix.h>
#include <preconditioners/error_matrix.h>
#include <preconditioners/multivector2.h>
#include <preconditioners/random_svd.h>
#include <preconditioners/wrappers.h>

// STL
#include <random>
namespace dealii
{
  /**
   * @brief modified matrix-matrix multiplication
   *
   * modified version of
   * dealii::TrilinosWrappers::internals::perform_mmult
   *
   * computes result = inputleft(^T) * diag(V) * inputright
   * afterwards the diagonal entries of result corresponding to indices are
   * set to 1 (to "remove" zero rows/columns of result)
   */
  inline void
  perform_modified_mmult(
    const TrilinosWrappers::SparseMatrix &inputleft,
    const TrilinosWrappers::SparseMatrix &inputright,
    TrilinosWrappers::SparseMatrix &      result,
    const TrilinosWrappers::MPI::Vector & V, /* scaling vector */
    const bool                            transpose_left,
    const std::vector<int>
      indices /* corresponding diagonal values are set to 1 */)
  {
    const bool use_vector = (V.size() == inputright.m() ? true : false);
    if (transpose_left == false)
      {
        Assert(inputleft.n() == inputright.m(),
               ExcDimensionMismatch(inputleft.n(), inputright.m()));
        Assert(inputleft.trilinos_matrix().DomainMap().SameAs(
                 inputright.trilinos_matrix().RangeMap()),
               ExcMessage("Parallel partitioning of A and B does not fit."));
      }
    else
      {
        Assert(inputleft.m() == inputright.m(),
               ExcDimensionMismatch(inputleft.m(), inputright.m()));
        Assert(inputleft.trilinos_matrix().RangeMap().SameAs(
                 inputright.trilinos_matrix().RangeMap()),
               ExcMessage("Parallel partitioning of A and B does not fit."));
      }

    result.clear();

    // create a suitable operator B: in case
    // we do not use a vector, all we need to
    // do is to set the pointer. Otherwise,
    // we insert the data from B, but
    // multiply each row with the respective
    // vector element.
    Teuchos::RCP<Epetra_CrsMatrix> mod_B;
    if (use_vector == false)
      {
        mod_B = Teuchos::rcp(
          const_cast<Epetra_CrsMatrix *>(&inputright.trilinos_matrix()), false);
      }
    else
      {
        mod_B = Teuchos::rcp(
          new Epetra_CrsMatrix(Copy, inputright.trilinos_sparsity_pattern()),
          true);
        mod_B->FillComplete(inputright.trilinos_matrix().DomainMap(),
                            inputright.trilinos_matrix().RangeMap());
        Assert(
          inputright.local_range() == V.local_range(),
          ExcMessage(
            "Parallel distribution of matrix B and vector V does not  match."));

        const int local_N = inputright.local_size();
        for (int i = 0; i < local_N; ++i)
          {
            int     N_entries = -1;
            double *new_data, *B_data;
            mod_B->ExtractMyRowView(i, N_entries, new_data);
            inputright.trilinos_matrix().ExtractMyRowView(i, N_entries, B_data);
            double value = V.trilinos_vector()[0][i];
            for (TrilinosWrappers::types::int_type j = 0; j < N_entries; ++j)
              new_data[j] = value * B_data[j];
          }
      }


    TrilinosWrappers::SparseMatrix tmp_result(
      transpose_left ? inputleft.locally_owned_domain_indices() :
                       inputleft.locally_owned_range_indices(),
      inputright.locally_owned_domain_indices(),
      inputleft.get_mpi_communicator());

#ifdef DEAL_II_TRILINOS_WITH_EPETRAEXT
    EpetraExt::MatrixMatrix::Multiply(inputleft.trilinos_matrix(),
                                      transpose_left,
                                      *mod_B,
                                      false,
                                      const_cast<Epetra_CrsMatrix &>(
                                        tmp_result.trilinos_matrix()),
                                      false /*
                                      call_FillComplete_on_result */
                                      ,
                                      true /* keep_all_hard_zeros */);
#else
    Assert(false,
           ExcMessage("This function requires that the Trilinos "
                      "installation found while running the deal.II "
                      "CMake scripts contains the optional Trilinos "
                      "package 'EpetraExt'. However, this optional "
                      "part of Trilinos was not found."));
#endif

    for (auto &index : indices)
      tmp_result.set(index, index, 1.0);
    tmp_result.compress(VectorOperation::insert);

    result.reinit(tmp_result.trilinos_matrix());
  }


  /**
   * computes and prints n_values singular values of E = I - matrix *
   * preconditioner
   * @param[in] unsigned int n_values number of singular values
   * @param[in] unsigned int shift shifts the constrainted indices to start from
   * 0
   * @param[in] unsigned int size number of indices included in E (e.g. number
   * of pressure dofs)
   * @param[in] dealii::AffineConstraints<double> constraints constraints for
   * the considered dofs
   * @param[in] MatrixType matrix matrix
   * @param[in] PreconditionerType preconditioner preconditioner
   */
  template <typename MatrixType, typename PreconditionerType>
  inline void
  analyze_singular_values(const unsigned int               n_values,
                          const unsigned int               shift,
                          const unsigned int               size,
                          const AffineConstraints<double> &constraints,
                          const MatrixType &               matrix,
                          const PreconditionerType &       preconditioner)
  {
    Preconditioners::ConstrainedIdentityMatrix constrained_identity(
      size /*vector size*/, shift /*shift */, constraints);

    Preconditioners::ErrorMatrix<MatrixType, PreconditionerType> error_matrix(
      matrix, preconditioner, constrained_identity);

    Assert(n_values > 0,
           ExcMessage("choose a positive number of update vectors"));


    typename RandomSVD<
      Preconditioners::ErrorMatrix<MatrixType,
                                   PreconditionerType>>::AdditionalData data2;
    data2.rank               = n_values;
    data2.size_subproblem    = data2.rank + 20;
    data2.n_rows             = size;
    data2.n_cols             = size;
    data2.n_power_iterations = 3;
    RandomSVD<Preconditioners::ErrorMatrix<MatrixType, PreconditionerType>>
      rsvd_solver(error_matrix, data2);
    rsvd_solver.solve();
  }



} // namespace dealii

#endif
