/**
 * @file arnoldi.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef R_ARNOLDI
#define R_ARNOLDI

// deal.II
#include <deal.II/base/smartpointer.h>

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

// STL
#include <random>

// preconditioners
#include <preconditioners/multivector2.h>

namespace dealii
{
  /**
   * @brief Class used to compute a low-rank Arnoldi decomposition.
   *
   * @tparam MatrixType
   */
  template <typename MatrixType>
  class Arnoldi : public Subscriptor
  {
  public:
    /**
     * @brief Struct for the options of the Arnoldi method.
     *
     */
    struct AdditionalData
    {
      /**
       * @brief Construct a new AdditionalData object
       *
       * @param rank Target rank of the Arnoldi decomposition.
       * @param dim Size of the Arnoldi vectors.
       */
      AdditionalData(const unsigned int rank = 0, const unsigned int dim = 0)
        : rank(rank)
        , dim(dim)
      {}

      /**
       * @brief Construct a new AdditionalData object
       *
       * @param data
       */
      AdditionalData(const AdditionalData &data)
        : rank(data.rank)
        , dim(data.dim)
      {}

      unsigned int rank;
      unsigned int dim;
    };

    /**
     * @brief Construct a new Arnoldi object
     *
     * @param matrix Matrix to be decomposed. Needs only to be defined by its action on a vector.
     * @param data Options for the Arnoldi decomposition.
     */
    Arnoldi(const MatrixType &matrix, const AdditionalData &data)
      : matrix(&matrix)
      , data(data)
    {}

    /**
     * @brief Compute an Arnoldi decomposition of rank @ref rank.
     *
     * @param lowrank_matrix Contains @ref rank Arnoldi vectors.
     * @param hessenberg_matrix rank x rank Hessenberg matrix.
     */
    void
    solve(Preconditioners::MultiVector2 &lowrank_matrix,
          LAPACKFullMatrix<double> &     hessenberg_matrix) const
    {
      Assert(data.rank > 0, ExcMessage("Number of columns should be >0."));
      lowrank_matrix.reinit(data.rank + 1, data.dim);
      hessenberg_matrix.reinit(data.rank + 1, data.rank);

      Preconditioners::MultiVector2::Column tmp(
        lowrank_matrix.trilinos_vector(), 0, Copy);
      Preconditioners::MultiVector2::Column tmp2(tmp);


      double value = 0;

      tmp2.trilinos_vector().Random();
      matrix->vmult(tmp, tmp2);
      value = tmp.l2_norm();
      value = 1.0 / value;
      tmp *= value;

      tmp.set_column(lowrank_matrix, 0);

      // Perform rank r steps of the Arnoldi iteration
      for (unsigned int i = 1; i <= data.rank; ++i)
        {
          Preconditioners::MultiVector2::Column vector_i(
            lowrank_matrix.trilinos_vector(), i, Copy);
          tmp.reinit(lowrank_matrix.trilinos_vector(), i - 1, Copy);
          matrix->vmult(vector_i, tmp);
          double start_value = vector_i.l2_norm();
          for (unsigned int j = 0; j < i; ++j)
            {
              tmp.reinit(lowrank_matrix, j, Copy);
              value = tmp * vector_i;
              hessenberg_matrix.set(j, i - 1, value);
              vector_i.add(-value, tmp);
            }
          value = vector_i.l2_norm();
          hessenberg_matrix.set(i, i - 1, value);

          // Check whether we want to reorthogonalize
          if (value > 10. * start_value *
                        std::sqrt(std::numeric_limits<double>::epsilon()))
            { // Assert(value != 0, ExcDivideByZero());
              value = 1. / value;
              vector_i *= value;
              vector_i.set_column(lowrank_matrix, i);
            }
          else
            {
              // reorthogonalize
              std::cout << "Reorthogonalize Arnoldi vectors." << std::endl;

              for (unsigned int j = 0; j < i; ++j)
                {
                  tmp.reinit(lowrank_matrix, j, Copy);
                  value = tmp * vector_i;
                  hessenberg_matrix(j, i - 1) += value;
                  vector_i.add(-value, tmp);
                }
              value = vector_i.l2_norm();
              hessenberg_matrix(i, i - 1) += value;
              value = 1. / value;
              vector_i *= value;
              vector_i.set_column(lowrank_matrix, i);
            }
        }

      hessenberg_matrix.grow_or_shrink(data.rank);
      Epetra_MultiVector tmp_matrix(lowrank_matrix.get_columns());
      lowrank_matrix.reinit(tmp_matrix, data.rank);
    }


  private:
    const SmartPointer<const MatrixType> matrix;

    AdditionalData data;
  };


} // namespace dealii

#endif