/**
 * @file constrained_identity_matrix.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef CONSTRAINED_IDENTITY
#define CONSTRAINED_IDENTITY

// deal.II
#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/lapack_templates.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>

// STL
#include <memory>

// preconditioners
#include <preconditioners/multivector2.h>

namespace dealii
{
  namespace Preconditioners
  {
    /**
     * @brief Matrix class that describes an identity matrix where diagonal entries
     * corresponding to constrained dofs are set to zero. It is meant to be
     * multiplied with the pressure block of the system vector.
     */
    class ConstrainedIdentityMatrix : public Subscriptor
    {
    public:
      /**
       * @brief Constructor.
       * @param size dimension of the identity matrix
       * @param n_u number of velocity dofs (shifts the constraints to the right index)
       * @param constraints contains constrained dofs
       */
      ConstrainedIdentityMatrix(const int                        size,
                                const int                        n_u,
                                const AffineConstraints<double> &constraints)
        : Subscriptor()
        , size(size)
        , n_u(n_u)
        , constraints(constraints)
      {}

      /**
       * @brief Copy constructor.
       * @param size dimension of the identity matrix
       * @param n_u number of velocity dofs (shifts the constraints to the right index)
       * @param constraints contains constrained dofs
       */
      ConstrainedIdentityMatrix(
        const ConstrainedIdentityMatrix &constrained_identity)
        : Subscriptor()
        , size(constrained_identity.size)
        , n_u(constrained_identity.n_u)
        , constraints(constrained_identity.constraints)
      {}

      ~ConstrainedIdentityMatrix()
      {}

      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[out] trg result
       * @param[in] src source vector
       */
      void
      vmult(TrilinosWrappers::MPI::Vector &      trg,
            const TrilinosWrappers::MPI::Vector &src) const
      {
        trg = src;

        for (unsigned int i = 0; i < trg.size(); ++i)
          if (constraints.is_constrained(i + n_u))
            trg(i) = 0.0;
      }

      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[out] trg result
       * @param[in] src source vector
       */
      void
      vmult(TrilinosWrappers::MPI::BlockVector &      trg,
            const TrilinosWrappers::MPI::BlockVector &src) const
      {
        trg = src;

        for (unsigned int i = 0; i < trg.size(); ++i)
          if (constraints.is_constrained(i))
            trg(i) = 0.0;
      }

      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[out] trg result
       * @param[in] src source vector
       */
      void
      vmult(Preconditioners::MultiVector2 &      trg,
            const Preconditioners::MultiVector2 &src) const
      {
        trg = src;


        for (unsigned int i = 0; i < src.size(); ++i)
          if (constraints.is_constrained(i + n_u))
            trg.set_row(0.0, i);
      }

      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[out] trg result
       * @param[in] src source vector
       */
      void
      vmult(Preconditioners::MultiVector2::Column &      trg,
            const Preconditioners::MultiVector2::Column &src) const
      {
        trg = src;


        for (int i = 0; i < src.size(); ++i)
          if (constraints.is_constrained(i + n_u))
            trg[i] = 0.0;
      }



      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[in,out] trgsrc source vector that is overwritten with the
       * result.
       */
      void
      vmult(TrilinosWrappers::MPI::Vector &trgsrc) const
      {
        for (unsigned int i = 0; i < trgsrc.size(); ++i)
          if (constraints.is_constrained(i + n_u))
            trgsrc(i) = 0.0;
      }

      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[in,out] trgsrc source vector that is overwritten with the
       * result.
       */
      void
      vmult(Preconditioners::MultiVector2 &trgsrc) const
      {
        for (unsigned int i = 0; i < trgsrc.size(); ++i)
          if (constraints.is_constrained(i + n_u))
            trgsrc.set_row(0.0, i);
      }

      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[in,out] trgsrc source vector that is overwritten with the
       * result.
       */
      void
      vmult(TrilinosWrappers::MPI::BlockVector &trgsrc) const
      {
        for (unsigned int i = 0; i < trgsrc.size(); ++i)
          if (constraints.is_constrained(i))
            trgsrc(i) = 0.0;
      }

      /**
       * @brief Computes the matrix-vector product with the 'constrained' identity
       * matrix. DoFs corresponding to @ref constraints are set to zero.
       *
       * @param[in,out] trgsrc source vector that is overwritten with the
       * result.
       */
      void
      vmult(Preconditioners::MultiVector2::Column &trgsrc) const
      {
        for (int i = 0; i < trgsrc.size(); ++i)
          if (constraints.is_constrained(i))
            trgsrc[i] = 0.0;
      }

    private:
      const int                       size;
      const int                       n_u;
      const AffineConstraints<double> constraints;
    };
  } // namespace Preconditioners
} // namespace dealii
#endif