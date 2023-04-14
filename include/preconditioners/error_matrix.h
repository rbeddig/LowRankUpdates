/**
 * @file error_matrix.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef ERRORMATRIX
#define ERRORMATRIX

// deal.II
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/trilinos_vector.h>

// Trilinos

// STL
#include <memory>

// preconditioners
#include <preconditioners/constrained_identity_matrix.h>
#include <preconditioners/multivector2.h>

namespace dealii
{
  namespace Preconditioners
  {
    /**
     * @brief Error matrix class for the computation of low-rank updates.
     * Describes the matrix \f$ E = I - \alpha S P_S \f$.
     */
    template <typename MatrixType, typename PreconditionerType>
    class ErrorMatrix : public Subscriptor
    {
    public:
      /**
       * Constructor
       * @param schur_complement Describes S.
       * @param schur_preconditioner Describes P_S.
       * @param in_constrained_identity (optional)
       */
      ErrorMatrix(const MatrixType &               schur_complement,
                  const PreconditionerType &       schur_preconditioner,
                  const double                     weight = 1.0,
                  const ConstrainedIdentityMatrix &in_constrained_identity =
                    ConstrainedIdentityMatrix(0,
                                              0,
                                              AffineConstraints<double>()))
        : Subscriptor()
        , schur_complement(&schur_complement)
        , schur_preconditioner(&schur_preconditioner)
        , weight(weight)
      {
        constrained_identity =
          std::make_shared<ConstrainedIdentityMatrix>(in_constrained_identity);
      }

      /**
       * Computes the matrix-vector product with E = I - S P_S.
       *
       * @param[out] trg result
       * @param[in] src source vector
       */
      template <typename VectorType>
      void
      vmult(VectorType &trg, const VectorType &src) const
      {
        VectorType tmp(src);

        // S \hat{S}^-1 v
        schur_preconditioner->vmult(tmp, src);
        schur_complement->vmult(trg, tmp);

        // (I - w S \hat{S}^-1) v
        trg *= -weight;
        trg.add(1.0, src);
        constrained_identity->vmult(trg);
      }

      /**
       * Computes the transposed matrix-vector product with E = I - S P_S.
       *
       * @param[out] trg result
       * @param[in] src source vector
       */
      template <typename VectorType>
      void
      Tvmult(VectorType &trg, const VectorType &src) const
      {
        VectorType tmp(src);

        // \hat{S}^-T S^T v
        schur_complement->Tvmult(tmp, src);
        schur_preconditioner->Tvmult(trg, tmp);


        // (I - w \hat{S}^-T S^T) v
        trg *= -weight;
        trg.add(1.0, src);
        constrained_identity->vmult(trg);
      }


    private:
      SmartPointer<const MatrixType>             schur_complement;
      SmartPointer<const PreconditionerType>     schur_preconditioner;
      double                                     weight;
      std::shared_ptr<ConstrainedIdentityMatrix> constrained_identity;
    };
  } // namespace Preconditioners
} // namespace dealii
#endif