/**
 * @file random_svd.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef RANDOM_SVD
#define RANDOM_SVD

// deal.II
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/lapack_templates.h>
#include <deal.II/lac/trilinos_vector.h>

// Trilinos
#include <Epetra_MultiVector.h>

// preconditioners
#include <preconditioners/multivector2.h>
#include <preconditioners/wrappers.h>

namespace dealii
{
  /**
   * Implements the randomized singular value decomposition based on
   * \cite{MarRT11} and \cite{HalMT11}:
   *
   * The procedure is as follows:
   *     1. To find \f$Q\f$, we compute \f$ M = (E^T E)^q E^T G \f$ for a
   *        random matrix \f$G \in \mathbb{R}^{n \times l}, l \geq r\f$.
   *        A small number \f$q\f$ of power iterations (\f$q \leq 3\f$) is
   *        usually sufficient. We orthogonalize after each multiplication
   *        with \f$E\f$ or \f$E^T\f$ to improve accuracy and robustness.
   *        The first \f$r\f$ columns of the result form \f$Q\f$. The
   *        application of a few power iterations and orthogonalization
   *        of the columns after each matrix-matrix-multiplication with
   *        \f$E\f$ or \f$E^T\f$ improves accuracy and robustness.
   *     2. We compute the economy-size SVD of \f$C = EQ = U_r \Sigma_r
   *        \hat{V}_r^T\f$.
   *     3. We find the approximate truncated SVD of \f$E\f$ as
   *        \f[
   *                E = (E^T)^T
   *                \approx (Q Q^T E^T)^T = (Q C^T)^T = C Q^T
   *                = U_r \Sigma_r \hat{V}_r^T Q^T
   *                = U_r \Sigma_r V_r^T.
   *        \f]
   */
  template <typename MatrixType>
  class RandomSVD : public Subscriptor
  {
  public:
    /**
     * @brief Containts the options for the randomized SVD.
     */


    struct AdditionalData
    {
      /**
       * @brief Construct a new Additional Data object
       *
       * @param rank Rank of the randomized SVD.
       * @param size_subproblem Sum of rank and number of oversampling vectors.
       * @param n_rows Number of matrix rows.
       * @param n_cols Number of matrix columns.
       * @param n_power_iterations Number of power iteratons.
       */
      AdditionalData(const int rank               = 0,
                     const int size_subproblem    = 0,
                     const int n_rows             = 0,
                     const int n_cols             = 0,
                     const int n_power_iterations = 0)
        : rank(rank)
        , size_subproblem(size_subproblem)
        , n_rows(n_rows)
        , n_cols(n_cols)
        , n_power_iterations(n_power_iterations)
      {}

      int rank;               // k
      int size_subproblem;    // l
      int n_rows;             // m
      int n_cols;             // n
      int n_power_iterations; // number of power iterations
    };

    /** @brief Default constructor */
    RandomSVD()
      : Subscriptor()
    {}

    /**
     * @brief Constructor.
     * @param matrix Matrix.
     * @param data Options for the SVD solver.
     */
    RandomSVD(const MatrixType &    matrix,
              const AdditionalData &data = AdditionalData())
      : Subscriptor()
      , matrix(&matrix)
      , data(data)
    {}


    /** Destructor */
    ~RandomSVD()
    {}

    /**
     * @brief Computes the truncated randomized SVD.
     * @param precomputed_vectors Can be used for repeated updates based on \f$E = I - S P_S\f$, stores
     *        the product \f$P_S^T S^T G\f$. (optional)
     */
    void
    solve(const Preconditioners::MultiVector2 &precomputed_vectors =
            Preconditioners::MultiVector2())
    {
      // Create G
      if (precomputed_vectors.size() == 0)
        create_gaussmatrix();

      left_singular_vectors = std::make_unique<Epetra_MultiVector>(
        Epetra_Map(
          data.n_cols, data.n_cols, 0, Utilities::Trilinos::comm_self()),
        data.size_subproblem);
      Preconditioners::MultiVector2 q_matrix(*left_singular_vectors, data.rank);

      power_range_finder(q_matrix, data.size_subproblem, precomputed_vectors);

      Epetra_FEVector               tmp3(Epetra_Map(data.n_cols,
                                      data.n_cols,
                                      0,
                                      Utilities::Trilinos::comm_self()),
                           data.size_subproblem);
      Preconditioners::MultiVector2 t_matrix(tmp3);
      matrix->vmult(t_matrix, q_matrix);

      // svd of T
      char                jobz  = 'S';
      int                 m     = data.n_rows;
      int                 n     = data.size_subproblem;
      int                 lda   = m;
      int                 ldu   = m;
      int                 ldvt  = n;
      int                 lwork = -1;
      std::vector<double> work(1);
      std::vector<double> left_view(m * n);
      std::vector<double> right_view(n * n);
      std::vector<int>    iwork(8 * std::min(m, n));
      singular_values.resize(n);
      int info = 0;

      // determine needed size of working vector
      dgesdd_(&jobz,
              &m,
              &n,
              t_matrix.begin(),
              &lda,
              singular_values.data(),
              left_view.data(),
              &ldu,
              right_view.data(),
              &ldvt,
              work.data(),
              &lwork,
              iwork.data(),
              &info);

      // compute SVD
      lwork = std::abs(work[0]) + 1;
      work.resize(lwork);
      Epetra_MultiVector tmp_left_singular_vectors(
        Epetra_Map(m, m, 0, Utilities::Trilinos::comm_self()), n);
      Epetra_MultiVector tmp_vector(
        Epetra_Map(n, n, 0, Utilities::Trilinos::comm_self()), n);
      dgesdd_(&jobz,
              &m,
              &n,
              t_matrix.begin(),
              &lda,
              singular_values.data(),
              tmp_left_singular_vectors[0],
              &ldu,
              tmp_vector[0],
              &ldvt,
              work.data(),
              &lwork,
              iwork.data(),
              &info);

      // compute V
      Epetra_MultiVector tmp_right_singular_vectors(
        Epetra_Map(m, m, 0, Utilities::Trilinos::comm_self()), n);

      int ierr = tmp_right_singular_vectors.Multiply(
        'N', 'T', 1.0, q_matrix.get_columns(), tmp_vector, 0.0);
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));

      right_singular_vectors = std::make_unique<Epetra_MultiVector>(
        Copy, tmp_right_singular_vectors, 0, data.rank);
      left_singular_vectors = std::make_unique<Epetra_MultiVector>(
        Copy, tmp_left_singular_vectors, 0, data.rank);

      std::cout << "   singular values: " << std::endl;
      for (auto &s : singular_values)
        {
          std::cout << "      " << s << std::endl;
        }

      singular_values.resize(data.rank);
    }


    /** @brief Creates the random test matrix \f$ G \f$. */
    void
    create_gaussmatrix()
    {
      gauss_matrix.reinit(data.size_subproblem, data.n_cols);
      gauss_matrix.fill_random();
    }


    /**
     * @brief Returns @ref data.rank right singular vectors of @ref matrix in @ref output_vector.
     *
     * @param output_vector Will be filled with the computed singular values.
     */
    void
    get_right_lowrank_matrix(Preconditioners::MultiVector2 &output_vector) const
    {
      output_vector.reinit(*right_singular_vectors, data.rank);
    }


    /**
     * @brief Returns @ref data.rank left singular vectors, scaled with the corresponding singular values,
     * of @ref matrix in @ref output_vector.
     *
     * @param output_vector Left singular vectors scales with singular values.
     */
    void
    get_left_lowrank_matrix(Preconditioners::MultiVector2 &output_vector) const
    {
      output_vector.reinit(*left_singular_vectors, data.rank);
      output_vector.scale(singular_values);
    }

    /**
     * @brief Precompute P_S^{-T} S^T G.
     *
     * Needed for repeated updates based on the error matrix \f$E = I - S P_S\f$
     *
     * @param precomputed_vectors Precomputed vectors.
     * @param schur_complement Object describing the (approximate) Schur complement \f$S\f$.
     * @param preconditioner Initial preconditioner \f$P_S\f$.
     *
     */
    template <typename SchurComplementType, typename PreconditionerType>
    void
    precompute_vectors(
      Preconditioners::MultiVector2 &precomputed_vectors,
      const SchurComplementType &    schur_complement,
      const PreconditionerType &     preconditioner,
      const bool transposed = true /* for right preconditioning*/)
    {
      create_gaussmatrix();

      // \hat{S}^{-1} S G or  \hat{S}^{-T} S^T G
      // TrilinosWrappers::MPI::Vector tmp(precomputed_vectors[0]);
      Preconditioners::MultiVector2 tmp(gauss_matrix);
      if (transposed)
        {
          schur_complement.Tvmult(tmp, gauss_matrix);
          preconditioner.Tvmult(precomputed_vectors, tmp);
        }
      else
        {
          Assert(false,
                 ExcNotImplemented()); // we need here the transposed version of
                                       // the randSVD-algorithm
          schur_complement.vmult(tmp, gauss_matrix);
          preconditioner.vmult(precomputed_vectors, tmp);
        }
    }

    /**
     * @brief Power range finder from \cite{HalMT11}
     *
     * @param[out] q_matrix @ref n_columns orthogonal vectors that approximate the range of E^T
     * @param[in] n_columns dimension of approximate range of E^T
     * @param[in] precomputed_vectors precomputed vectors \f$= P_S^T S^T G\f$
     */
    void
    power_range_finder(Preconditioners::MultiVector2 &      q_matrix,
                       const int                            n_columns,
                       const Preconditioners::MultiVector2 &precomputed_vectors)
    {
      enum PowerRangeFinder
      {
        basic,
        ortho
      };
      PowerRangeFinder range_finder_method = ortho; // basic;
      switch (range_finder_method)
        {
            case basic: { // Compute R^T
              Epetra_FEVector tmp(gauss_matrix.get_columns().Map(),
                                  data.size_subproblem);
              Preconditioners::MultiVector2 r_matrix(tmp);
              Preconditioners::MultiVector2 tmp_matrix(r_matrix);

              if (precomputed_vectors.size() != 0)
                {
                  r_matrix = 0.0;
                  r_matrix.add(-1.0, precomputed_vectors, 1.0, gauss_matrix);
                }
              else
                matrix->Tvmult(r_matrix, gauss_matrix);
              for (int j = 0; j < data.n_power_iterations; ++j)
                {
                  matrix->vmult(tmp_matrix, r_matrix);
                  matrix->Tvmult(r_matrix, tmp_matrix);
                }
              r_matrix.orthogonalize();
              q_matrix.reinit(r_matrix, n_columns);

              break;
            }
            case ortho: { // Algorithm 4.4 in HalMT11

              // Compute R^T
              Epetra_FEVector               tmp(left_singular_vectors->Map(),
                                  data.size_subproblem);
              Preconditioners::MultiVector2 r_matrix(tmp);
              Preconditioners::MultiVector2 q_matrix2(r_matrix);

              // R^T = A^T G
              if (precomputed_vectors.size() != 0)
                {
                  r_matrix = 0.0;
                  r_matrix.add(-1.0, precomputed_vectors, 1.0, gauss_matrix);
                }
              else
                {
                  matrix->Tvmult(r_matrix, gauss_matrix);
                }
              r_matrix.orthogonalize(); // R^T = Q R, R^T <- Q.

              // power iterations
              for (int j = 0; j < data.n_power_iterations; ++j)
                { /**************** A Q ****************************/
                  // R^T = A Q
                  matrix->vmult(q_matrix2, r_matrix);
                  q_matrix2.orthogonalize();


                  //************* A^T Q ************//
                  // R^T = A^T Q
                  matrix->Tvmult(r_matrix, q_matrix2);
                  r_matrix.orthogonalize();
                }
              q_matrix.reinit(r_matrix, n_columns); // TODO

              break;
            }
        }
    }


    /**
     * @brief Sets the random test matrix.
     */
    void
    set_gaussmatrix(const Preconditioners::MultiVector2 &matrix)
    {
      gauss_matrix = matrix;
    }

    /**
     * @brief Returns the random test matrix.
     */
    void
    get_gaussmatrix(Preconditioners::MultiVector2 &matrix) const
    {
      matrix = gauss_matrix;
    }

  private:
    Preconditioners::MultiVector2       gauss_matrix;
    std::unique_ptr<Epetra_MultiVector> left_singular_vectors;
    std::unique_ptr<Epetra_MultiVector> right_singular_vectors;
    std::vector<double>                 singular_values;

    const SmartPointer<const MatrixType>
      matrix; /** Points to an object that describes the error matrix. */

    AdditionalData data; /** Contains options */
  };
} // namespace dealii


#endif