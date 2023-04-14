/**
 * @file ata.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef ATA_H
#define ATA_H

// Deal ii
#include <deal.II/base/index_set.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <preconditioners/wrappers.h>

namespace dealii
{
  namespace Preconditioners
  {
    /**
     * @brief Describes matrices of type A^T M A.
     *
     * Matrix class that wraps the matrix product A^T M A where @ref A is given as
     * TrilinosWrappers::SparseMatrix and M is a diagonal matrix given in @ref scaling_vector.
     */
    class TrilinosATA : public dealii::Subscriptor
    {
    public:
      using VectorType = dealii::TrilinosWrappers::MPI::Vector;
      using MatrixType = dealii::TrilinosWrappers::SparseMatrix;
      using real_type  = double;
      using size_type  = unsigned int;

      /**
       * Default constructor.
       */
      TrilinosATA() = default;

      /**
       * Constructor from matrix, initialize the scaling later with @ref initialize().
       *
       * @param A matrix
       * @param constrained_dofs Vector that contains all constrained dofs, needed to set the corresponding diagonal entries to 1 (to prevent the matrix product A^T M A from being singular).
       */
      TrilinosATA(const MatrixType &               A,
                  const std::vector<unsigned int> &constrained_dofs =
                    std::vector<unsigned int>(0))
        : Subscriptor()
        , A(A)
        , constrained_dofs(constrained_dofs)
      {
        scaling_vector.reinit(complete_index_set(A.m()));
        scaling_vector = 1.0;
        tmp.reinit(complete_index_set(A.m()));
      }


      /**
       * @brief Constructor
       * @param A Matrix A.
       * @param scaling_vector Diagonal entries of M.
       * @param constrained_dofs Constrained dofs, needed to set the corresponding diagonal entries to 1 (to prevent the matrix product A^T M A from being singular).
       */
      TrilinosATA(const MatrixType &               A,
                  const VectorType &               scaling_vector,
                  const std::vector<unsigned int> &constrained_dofs =
                    std::vector<unsigned int>(0))
        : Subscriptor()
        , A(A)
        , scaling_vector(scaling_vector)
        , constrained_dofs(constrained_dofs)
      {
        tmp.reinit(complete_index_set(A.m()));
      }



      /**
       * @brief Destructor
       */
      ~TrilinosATA()
      {}


      /**
       * @brief Initialize the scaling M.
       * @param vector Diagonal entries of M
       * @param shift Shift of constrained dofs (shift smallest index to 0)
       * @param in_constrained_dofs constrained dofs, needed to set corresponding diagonal entries to 1 (to prevent the matrix product A^T M A from being singular)
       */
      void
      initialize(const VectorType &               vector,
                 const unsigned int               shift = 0,
                 const std::vector<unsigned int> &in_constrained_dofs =
                   std::vector<unsigned int>(0))
      {
        scaling_vector.reinit(vector);
        scaling_vector = vector;

        constrained_dofs = in_constrained_dofs;
        for (auto &index : constrained_dofs)
          index -= shift;
      }

      /**
       * @brief Initialize directly with the scaling.
       * @param matrix Matrix A, we obtain A^T M A (M is a diagonal scaling matrix)
       * @param vector Scaling vector that contains the diagonal entries of M
       * @param shift shift of constrained dofs (smallest index)
       * @param std::vector<unsigned int> in_constrained_dofs constrained dofs, needed to set corresponding diagonal entries to 1 (to prevent the matrix product A^T M A from being singular)
       */
      void
      initialize(const TrilinosWrappers::SparseMatrix &matrix,
                 const VectorType &                    vector,
                 const unsigned int                    shift,
                 const std::vector<unsigned int> &     in_constrained_dofs =
                   std::vector<unsigned int>(0))
      {
        A.reinit(matrix);

        scaling_vector.reinit(vector);
        scaling_vector = vector;

        constrained_dofs = in_constrained_dofs;
        for (auto &index : constrained_dofs)
          index -= shift;

        tmp.reinit(complete_index_set(matrix.m()));
      }

      /**
       * @brief Initialize without scaling.
       * @param matrix Matrix A, we obtain A^T M A (M is a diagonal scaling matrix)
       * @param shift shift of constrained dofs (smallest index)
       * @param std::vector<unsigned int> in_constrained_dofs constrained dofs, needed to set corresponding diagonal entries to 1 (to prevent the matrix product A^T M A from being singular)
       */
      void
      initialize(const TrilinosWrappers::SparseMatrix &matrix,
                 const unsigned int                    shift = 0,
                 const std::vector<unsigned int> &     in_constrained_dofs =
                   std::vector<unsigned int>(0))
      {
        A.reinit(matrix);

        constrained_dofs = in_constrained_dofs;
        for (auto &index : constrained_dofs)
          index -= shift;

        tmp.reinit(complete_index_set(matrix.m()));
      }



      /**
       * @brief Computes w = A^T M A v
       *
       * Computes the product w = A^T M A v, where M is a diagonal matrix whose diagonal entries are stored in @ref scaling_vector.
       * @param[out] trg Target vector w.
       * @param[in] src Source vector v.
       */
      void
      vmult(VectorType &trg, const VectorType &src) const
      {
        A.vmult(tmp, src);
        tmp.scale(scaling_vector);
        A.Tvmult(trg, tmp);

        for (auto &index : constrained_dofs)
          trg[index] += src[index];
      }

      /**
       * @brief Computes w = A^T M A v
       *
       * Computes the product w = A^T M A v where M is a diagonal matrix whose diagonal entries are stored in @ref scaling_vector.
       * @param[out] trg Target vector w.
       * @param[in] src Source vector v.
       */
      void
      vmult(dealii::Preconditioners::MultiVector2::Column &      trg,
            const dealii::Preconditioners::MultiVector2::Column &src) const
      {
        dealii::Preconditioners::MultiVector2::Column tmp(
          scaling_vector.trilinos_partitioner());
        A.vmult(tmp, src);
        tmp.scale(scaling_vector);
        A.Tvmult(trg, tmp);

        for (auto &index : constrained_dofs)
          trg[index] += src[index];
      }

      template <typename VectorType>
      void
      Tvmult(VectorType &trg, const VectorType &src) const
      {
        vmult(trg, src);
      }


      /**
       * @brief Computes the residual and its l2-norm.
       * @param[out] res residual vector
       * @param[in] sol solution vector
       * @param[in] rhs right-hand side
       * @return l2-norm of residual
       */
      real_type
      residual(VectorType &      res,
               const VectorType &sol,
               const VectorType &rhs) const
      {
        res = 0;
        this->vmult(res, sol);
        res -= rhs;
        res *= -1.0;

        return res.l2_norm();
      }

      /**
       * @brief Returns the size of the resulting matrix.
       */
      size_type
      n() const
      {
        return A.n();
      }


    private:
      WrapperMatrix A;              /** m x n matrix, m >> n */
      VectorType    scaling_vector; /** contains diagonal entries of M */

      mutable std::vector<unsigned int>
        constrained_dofs; /** contains constrained dofs */

      mutable VectorType tmp; /** temporary vector of size m */
    };

  } // namespace Preconditioners
} // namespace dealii
#endif
