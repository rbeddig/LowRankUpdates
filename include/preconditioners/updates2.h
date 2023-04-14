/**
 * @file updates2.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef UPDATES2
#define UPDATES2

// deal.II
#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/subscriptor.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_support.h>
#include <deal.II/lac/lapack_templates.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_vector.h>

// Trilinos
#include <Epetra_FEVector.h>

// STL
#include <memory>

//
#include <linear_algebra/block_schur_preconditioner.hpp>

// preconditioners
#include <preconditioners/arnoldi.h>
#include <preconditioners/constrained_identity_matrix.h>
#include <preconditioners/error_matrix.h>
#include <preconditioners/multivector2.h>
#include <preconditioners/random_svd.h>

namespace dealii
{
  namespace Preconditioners
  {
    class PreconditionerSelector;

    enum SVDSolverType
    {
      random,
      arnoldi
    };

    template <typename PreconditionerType>
    struct IsBlock00PreconditionerType
    {
    private:
      /**
       * @brief Overload returning true if the class is derived from
       * PreconditionBase.
       */
      static std::true_type
      check_preconditioner_type(const TrilinosWrappers::PreconditionBase *);


      /**
       * @brief Overload returning true if the class is derived from
       * PreconditionerSelector.
       */
      static std::true_type
      check_preconditioner_type(
        const Preconditioners::PreconditionerSelector *);

      /**
       * @brief Catch all for all other potential preconditioner types.
       */
      static std::false_type
      check_preconditioner_type(...);

    public:
      /**
       * @brief A statically computable value that indicates whether the template
       * argument to this class is derived from PreconditionBase (then we use
       * the block update).
       */
      static const bool value =
        std::is_same<decltype(check_preconditioner_type(
                       std::declval<PreconditionerType *>())),
                     std::true_type>::value;
    };

    template <typename PreconditionerType>
    const bool IsBlock00PreconditionerType<PreconditionerType>::value;

    template <typename MatrixType, typename PreconditionerType>
    class BlockUpdateBase : public Subscriptor
    {
    public:
      BlockUpdateBase()
        : Subscriptor()
      {}

    protected:
      /**
       * @brief Compute update vectors for the Schur complement preconditioner based on the block matrix (not yet implemented).
       *
       * @tparam DataType
       * @tparam ErrorMatrixType
       * @tparam ErrorMatrixType2
       * @tparam VectorType
       * @param n_new_vectors Number of update vectors
       * @param constraints Constraints on pressure dofs
       * @param number The number of the update
       * @param owned_partitioning Owned indices.
       * @param error_matrix Error matrix for the first update
       * @param error_matrix_2 Erro rmatrix for the following update.
       * @param precomputed_vectors Precomputed sampling vectors for the repeated update.
       * @param gaussmatrix Random test matrix.
       * @param matrix Schur complement (approximation).
       * @param initial_preconditioner Initial Schur complement preconditioner.
       * @param left_lowrank_matrix Left update vectors.
       * @param right_lowrank_matrix Right update vectors.
       * @param hessenberg_matrix Hessenberg matrix (needed for Arnoldi-Update)
       * @param data Parameters for the update.
       */
      template <typename DataType,
                typename ErrorMatrixType,
                typename ErrorMatrixType2,
                typename BlockVectorType,
                typename BlockPreconditionerType>
      void
      compute_update_vectors_block(
        const unsigned int                         n_new_vectors,
        const AffineConstraints<double> &          constraints,
        const unsigned int                         number,
        const std::vector<IndexSet> &              owned_partitioning,
        const ErrorMatrixType &                    error_matrix,
        const ErrorMatrixType2 &                   error_matrix_2,
        BlockVectorType &                          block_precomputed_vectors,
        BlockVectorType &                          block_gaussmatrix,
        const TrilinosWrappers::BlockSparseMatrix &matrix,
        const BlockPreconditionerType &            block_preconditioner,
        std::vector<MultiVector2> &                left_lowrank_matrix,
        std::vector<MultiVector2> &                right_lowrank_matrix,
        std::vector<LAPACKFullMatrix<double>> &    hessenberg_matrix,
        const DataType &                           data);


      /**
       * @brief Compute update vectors for the Schur complement preconditioner.
       *
       * @tparam DataType
       * @tparam ErrorMatrixType
       * @tparam ErrorMatrixType2
       * @tparam VectorType
       * @param n_new_vectors Number of update vectors
       * @param constraints Constraints on pressure dofs
       * @param number The number of the update
       * @param owned_partitioning Owned indices.
       * @param error_matrix Error matrix for the first update
       * @param error_matrix_2 Erro rmatrix for the following update.
       * @param precomputed_vectors Precomputed sampling vectors for the repeated update.
       * @param gaussmatrix Random test matrix.
       * @param matrix Schur complement (approximation).
       * @param initial_preconditioner Initial Schur complement preconditioner.
       * @param left_lowrank_matrix Left update vectors.
       * @param right_lowrank_matrix Right update vectors.
       * @param hessenberg_matrix Hessenberg matrix (needed for Arnoldi-Update)
       * @param data Parameters for the update.
       */
      template <typename DataType,
                typename ErrorMatrixType,
                typename ErrorMatrixType2,
                typename VectorType>
      void
      compute_update_vectors_nonblock(
        const unsigned int                     n_new_vectors,
        const AffineConstraints<double> &      constraints,
        const unsigned int                     number,
        const std::vector<IndexSet> &          owned_partitioning,
        const ErrorMatrixType &                error_matrix,
        const ErrorMatrixType2 &               error_matrix_2,
        VectorType &                           precomputed_vectors,
        VectorType &                           gaussmatrix,
        const MatrixType &                     matrix,
        const PreconditionerType &             initial_preconditioner,
        std::vector<MultiVector2> &            left_lowrank_matrix,
        std::vector<MultiVector2> &            right_lowrank_matrix,
        std::vector<LAPACKFullMatrix<double>> &hessenberg_matrix,
        const DataType &                       data);
    };

    ///////////////////////////////////////////
    // Implementation
    //////////////////////////////////////////
    template <typename MatrixType, typename PreconditionerType>
    template <typename DataType,
              typename ErrorMatrixType,
              typename ErrorMatrixType2,
              typename VectorType>
    void
    BlockUpdateBase<MatrixType, PreconditionerType>::
      compute_update_vectors_nonblock(
        const unsigned int                     n_new_vectors,
        const AffineConstraints<double> &      constraints,
        const unsigned int                     number,
        const std::vector<IndexSet> &          owned_partitioning,
        const ErrorMatrixType &                error_matrix,
        const ErrorMatrixType2 &               error_matrix_2,
        VectorType &                           precomputed_vectors,
        VectorType &                           gaussmatrix,
        const MatrixType &                     matrix,
        const PreconditionerType &             initial_preconditioner,
        std::vector<MultiVector2> &            left_lowrank_matrix,
        std::vector<MultiVector2> &            right_lowrank_matrix,
        std::vector<LAPACKFullMatrix<double>> &hessenberg_matrix,
        const DataType &                       data)
    {
      (void)n_new_vectors;
      (void)constraints;

      switch (data.solver_type)
        {
            case random: {
              // Use a randomized SVD for construction the update.
              if (number == 0)
                {
                  typename RandomSVD<ErrorMatrixType>::AdditionalData data2;
                  data2.rank            = data.rank;
                  data2.size_subproblem = std::max(data2.rank + 10, 15);
                  data2.n_rows          = owned_partitioning[data.block].size();
                  data2.n_cols          = owned_partitioning[data.block].size();
                  data2.n_power_iterations = 3;

                  RandomSVD<ErrorMatrixType> rsvd_solver(error_matrix, data2);

                  TrilinosWrappers::MPI::Vector tmp(
                    owned_partitioning[data.block]);
                  precomputed_vectors.resize(data2.size_subproblem, tmp);

                  rsvd_solver.precompute_vectors(precomputed_vectors,
                                                 matrix,
                                                 initial_preconditioner,
                                                 true);
                  precomputed_vectors *= data.weight;

                  rsvd_solver.solve(precomputed_vectors);
                  gaussmatrix.resize(data2.size_subproblem, tmp);
                  rsvd_solver.get_gaussmatrix(gaussmatrix);

                  rsvd_solver.get_right_lowrank_matrix(
                    right_lowrank_matrix[number]);
                  rsvd_solver.get_left_lowrank_matrix(
                    left_lowrank_matrix[number]);
                }
              else
                {
                  typename RandomSVD<ErrorMatrixType2>::AdditionalData data2;
                  data2.rank            = data.rank;
                  data2.size_subproblem = std::max(data2.rank + 10, 15);
                  data2.n_rows          = owned_partitioning[data.block].size();
                  data2.n_cols          = owned_partitioning[data.block].size();
                  data2.n_power_iterations = 3;

                  RandomSVD<ErrorMatrixType2> rsvd_solver(error_matrix_2,
                                                          data2);


                  rsvd_solver.set_gaussmatrix(gaussmatrix);
                  rsvd_solver.solve(precomputed_vectors);

                  rsvd_solver.get_right_lowrank_matrix(
                    right_lowrank_matrix[number]);
                  rsvd_solver.get_left_lowrank_matrix(
                    left_lowrank_matrix[number]);
                }

              break;
            }


            case arnoldi: {
              // Use Arnoldi vectors for constructing the update.
              if (number == 0)
                {
                  typename Arnoldi<ErrorMatrixType>::AdditionalData data2;
                  data2.rank = data.rank;
                  data2.dim  = owned_partitioning[data.block].size();

                  Arnoldi<ErrorMatrixType> arnoldi_solver(error_matrix, data2);
                  arnoldi_solver.solve(right_lowrank_matrix[number],
                                       hessenberg_matrix[number]);
                }
              else
                {
                  typename Arnoldi<ErrorMatrixType2>::AdditionalData data2;
                  data2.rank = data.rank;
                  data2.dim  = owned_partitioning[data.block].size();

                  Arnoldi<ErrorMatrixType2> arnoldi_solver(error_matrix_2,
                                                           data2);
                  arnoldi_solver.solve(right_lowrank_matrix[number],
                                       hessenberg_matrix[number]);
                }

              left_lowrank_matrix[number].reinit(right_lowrank_matrix[number]);

              break;
            }
        }
    }



    template <typename MatrixType, typename PreconditionerType>
    template <typename DataType,
              typename ErrorMatrixType,
              typename ErrorMatrixType2,
              typename BlockVectorType,
              typename BlockPreconditionerType>
    void
    BlockUpdateBase<MatrixType, PreconditionerType>::
      compute_update_vectors_block(
        const unsigned int                         n_new_vectors,
        const AffineConstraints<double> &          constraints,
        const unsigned int                         number,
        const std::vector<IndexSet> &              owned_partitioning,
        const ErrorMatrixType &                    error_matrix,
        const ErrorMatrixType2 &                   error_matrix_2,
        BlockVectorType &                          block_precomputed_vectors,
        BlockVectorType &                          block_gaussmatrix,
        const TrilinosWrappers::BlockSparseMatrix &matrix,
        const BlockPreconditionerType &            block_preconditioner,
        std::vector<MultiVector2> &                left_lowrank_matrix,
        std::vector<MultiVector2> &                right_lowrank_matrix,
        std::vector<LAPACKFullMatrix<double>> &    hessenberg_matrix,
        const DataType &                           data)
    {
      Assert(false, ExcNotImplemented());
    }



    /**
     * @brief Implements error-based low-rank updates for Schur complement
     * preconditioners as in \cite proceedings.
     *
     * @tparam PreconditionerType: class of the preconditioner that is used
     * @tparam MatrixType: class that implements the vmult method (matrix-vector
     * product) of the 'matrix' that will be preconditioned
     * */
    template <typename MatrixType, typename PreconditionerType>
    class LowRankUpdates
      : public BlockUpdateBase<MatrixType, PreconditionerType>
    {
    public:
      /** @brief Update options. */
      struct AdditionalData
      {
      public:
        /**
         * @brief Constructor.
         *
         * @param block Refers to the block that should be updated (=1 for the Schur complement preconditioner in our case).
         * @param rank Update rank.
         * @param weight Scaling for the update.
         * @param symmetric Use a symmetric update (non implemented in this version)
         * @param solver_type (random,arnoldi): Use a randomized SVD or Arnoldi decomposition for the update.
         * @param number Number of updates.
         */
        AdditionalData(
          const unsigned int  block       = 1, /* chosen block to be updated */
          const unsigned int  rank        = 15,
          const double        weight      = 1.0,
          const bool          symmetric   = false,
          const SVDSolverType solver_type = random,
          const unsigned int  number      = 1)
          : block(block)
          , rank(rank)
          , weight(weight)
          , symmetric(symmetric)
          , solver_type(solver_type)
          , number(number)
        {}


        unsigned int block;
        unsigned int rank;   /** rank of update */
        double       weight; /** to weight the error matrix E = I - weight * S
                                \hat{S}^{-1} */
        bool          symmetric;   /** use a symmetric update */
        SVDSolverType solver_type; /** solver type for the SVD */
        unsigned int  number;      /** number of updates in a row */
      };


      /** Constructor. */
      LowRankUpdates(const AdditionalData &data = AdditionalData())
        : BlockUpdateBase<MatrixType, PreconditionerType>()
        , data(data)
      {}

      /** Destructor. */
      ~LowRankUpdates()
      {}

      /**
       * @brief Initialize the update based on the error
       * E = I - in_matrix * in_preconditioner
       *
       * Computes the update vectors and precomputes the small inverse.
       */
      void
      initialize(
        const Preconditioners::WrapperMatrix &        in_matrix,
        const Preconditioners::WrapperPreconditioner &in_preconditioner,
        const std::vector<IndexSet> &                 in_partitioning,
        const AdditionalData &in_data = AdditionalData())
      {
        Assert(
          data.block == 0,
          ExcMessage(
            "This method is currently only implemented for the (0,0) block."));
        Timer timer;

        data = in_data;
        data.block_update =
          false; // we initialize the Schur update if we call this method

        // Initialize the underlying matrix + preconditioner
        matrix = &in_matrix;
        initial_preconditioner =
          const_cast<PreconditionerType *>(&in_preconditioner);
        error_matrix =
          std::make_shared<ErrorMatrix<MatrixType, PreconditionerType>>(
            in_matrix, in_preconditioner, data.weight);
        constrained_identity = std::make_unique<ConstrainedIdentityMatrix>(
          in_partitioning[data.block].size(), 0, AffineConstraints<double>() /* do not need to consider the constraints for the velocity block */);
        owned_partitioning = in_partitioning;

        // Compute the update vectors

        unsigned int needed_number = data.number;
        right_lowrank_matrix.resize(
          needed_number,
          MultiVector2(Epetra_FEVector(
            Epetra_Map((int)owned_partitioning[data.block].size(),
                       0,
                       Utilities::Trilinos::comm_self()),
            data.rank)));
        left_lowrank_matrix.resize(
          needed_number,
          MultiVector2(Epetra_FEVector(
            Epetra_Map((int)owned_partitioning[data.block].size(),
                       0,
                       Utilities::Trilinos::comm_self()),
            data.rank)));

        factorization.resize(
          needed_number,
          MultiVector2(Epetra_FEVector(
            Epetra_Map((int)data.rank, 0, Utilities::Trilinos::comm_self()),
            data.rank)));
        hessenberg_matrix.resize(needed_number,
                                 LAPACKFullMatrix<double>((int)(data.rank + 1),
                                                          (int)data.rank));
        ipiv.resize(needed_number, std::vector<int>((int)data.rank));


        for (unsigned int number = 0; number < needed_number; ++number)
          {
            timer.start();
            compute_update_vectors(data.rank,
                                   AffineConstraints<double>(),
                                   number);
            timer.stop();
            std::cout << " Computation of update vectors: " << timer.wall_time()
                      << " s" << std::endl;

            // Precompute the factorization
            timer.restart();
            precompute_factorization(number);
            timer.stop();
            std::cout << " Computation of the factorization: "
                      << timer.wall_time() << " s" << std::endl;
            current_max_number = number + 1;
          }
      }

      /**
       * @brief Initialize the update based on the error
       * E = I - in_matrix * in_preconditioner
       *
       * Computes the update vectors and precomputes the small inverse.
       */
      void
      initialize(const MatrixType &               in_matrix,
                 const PreconditionerType &       in_preconditioner,
                 const std::vector<IndexSet> &    in_partitioning,
                 const AffineConstraints<double> &constraints,
                 const AdditionalData &           in_data = AdditionalData())
      {
        Timer timer;

        data = in_data;

        // Initialize the underlying matrix + preconditioner
        matrix = &in_matrix;
        initial_preconditioner =
          const_cast<PreconditionerType *>(&in_preconditioner);
        constrained_identity =
          std::make_unique<ConstrainedIdentityMatrix>(in_partitioning[1].size(),
                                                      in_partitioning[0].size(),
                                                      constraints);
        error_matrix =
          std::make_shared<ErrorMatrix<MatrixType, PreconditionerType>>(
            in_matrix, in_preconditioner, data.weight, *constrained_identity);
        owned_partitioning = in_partitioning;

        // Compute the update vectors
        // const unsigned int number = data.number - 1;

        unsigned int needed_number = data.number;
        right_lowrank_matrix.resize(
          needed_number,
          MultiVector2(Epetra_FEVector(
            Epetra_Map((int)owned_partitioning[data.block].size(),
                       0,
                       Utilities::Trilinos::comm_self()),
            data.rank)));
        left_lowrank_matrix.resize(
          needed_number,
          MultiVector2(Epetra_FEVector(
            Epetra_Map((int)owned_partitioning[data.block].size(),
                       0,
                       Utilities::Trilinos::comm_self()),
            data.rank)));

        factorization.resize(
          needed_number,
          MultiVector2(Epetra_FEVector(
            Epetra_Map((int)data.rank, 0, Utilities::Trilinos::comm_self()),
            data.rank)));
        hessenberg_matrix.resize(needed_number,
                                 LAPACKFullMatrix<double>((int)(data.rank + 1),
                                                          (int)data.rank));
        ipiv.resize(needed_number, std::vector<int>((int)data.rank));


        for (unsigned int number = 0; number < needed_number; ++number)
          {
            timer.start();
            compute_update_vectors(data.rank, constraints, number);
            timer.stop();
            std::cout << " Computation of update vectors: " << timer.wall_time()
                      << " s" << std::endl;

            // Precompute the factorization
            timer.restart();
            precompute_factorization(number);
            timer.stop();
            std::cout << " Computation of the factorization: "
                      << timer.wall_time() << " s" << std::endl;
            current_max_number = number + 1;
          }
      }


      void
      reinit_A(const TrilinosWrappers::SparseMatrix &block00)
      {
        initial_preconditioner->reinit_A(block00);
      }

      /**
       * @brief Apply the update to a vector.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Source vector.
       */
      template <typename VectorType>
      void
      vmult(VectorType &trg, const VectorType &src) const;

      /**
       * @brief Apply the transposed update to a vector.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Source vector.
       */
      template <typename VectorType>
      void
      Tvmult(VectorType &trg, const VectorType &src) const;



    private:
      /**
       * @brief Apply the updated preconditioner.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Source vector.
       * @param max_number Number of applied updates.
       */
      template <typename VectorType>
      void
      apply_updated_preconditioner(VectorType &       trg,
                                   const VectorType & src,
                                   const unsigned int max_number) const;

      /**
       * @brief Apply the transposed updated preconditioner.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Source vector.
       * @param max_number Number of applied updates.
       */
      template <typename VectorType>
      void
      apply_updated_preconditioner_transposed(
        VectorType &       trg,
        const VectorType & src,
        const unsigned int max_number) const;



      /** applies the update */
      void
      apply_update(TrilinosWrappers::MPI::Vector &      trg,
                   const TrilinosWrappers::MPI::Vector &src,
                   const unsigned int                   number,
                   const bool                           transposed) const;

      /** applies the update */
      void
      apply_update(Preconditioners::MultiVector2 &      trg,
                   const Preconditioners::MultiVector2 &src,
                   const unsigned int                   number,
                   const bool                           transposed) const;

      /** applies the update */
      void
      apply_update(Preconditioners::MultiVector2::Column &      trg,
                   const Preconditioners::MultiVector2::Column &src,
                   const unsigned int                           number,
                   const bool transposed) const;

      /** Compute new update vectors. Calls the block (not yet implemented)
       *  or non-block version of @ref compute_update_vectors(). */
      template <typename BlockPreconditionerType = PreconditionIdentity>
      void
      compute_update_vectors(const unsigned int               n_new_vectors,
                             const AffineConstraints<double> &constraints,
                             const unsigned int               number,
                             const SmartPointer<const BlockPreconditionerType>
                               &block_preconditioner = nullptr);


      template <typename BlockPreconditionerType>
      void
      compute_update_vectors(const unsigned int               n_new_vectors,
                             const AffineConstraints<double> &constraints,
                             const unsigned int               number,
                             std::integral_constant<bool, true>,
                             const SmartPointer<const BlockPreconditionerType>
                               &block_preconditioner = nullptr);

      template <typename BlockPreconditionerType>
      void
      compute_update_vectors(const unsigned int               n_new_vectors,
                             const AffineConstraints<double> &constraints,
                             const unsigned int               number,
                             std::integral_constant<bool, false>,
                             const SmartPointer<const BlockPreconditionerType>
                               &block_preconditioner = nullptr);

      /** precompute the small inverse for the update */
      void
      precompute_factorization(const unsigned int number);

      /** apply the small inverse of the update */
      void
      solve_factorization(
        TrilinosWrappers::MPI::Vector &trg,
        const unsigned int number) const; //, const VectorType &src) const;

      /** apply the small inverse of the update */
      void
      solve_factorization(
        Preconditioners::MultiVector2 &trg,
        const unsigned int number) const; //, const VectorType &src) const;

      /** apply the small inverse of the update */
      void
      solve_factorization(
        Preconditioners::MultiVector2::Column &trg,
        const unsigned int number) const; //, const VectorType &src) const;

      /** apply the small inverse of the transposed update */
      void
      solve_transposed_factorization(TrilinosWrappers::MPI::Vector &trg,
                                     const unsigned int number) const;

      /** apply the small inverse of the transposed update */
      void
      solve_transposed_factorization(Preconditioners::MultiVector2 &trg,
                                     const unsigned int number) const;

      /** apply the small inverse of the transposed update */
      void
      solve_transposed_factorization(Preconditioners::MultiVector2::Column &trg,
                                     const unsigned int number) const;


      // members
      SmartPointer<PreconditionerType> initial_preconditioner; // P0
      SmartPointer<const MatrixType>   matrix;
      SmartPointer<const TrilinosWrappers::BlockSparseMatrix> block_matrix;

      std::shared_ptr<ErrorMatrix<MatrixType, PreconditionerType>> error_matrix;
      std::shared_ptr<
        ErrorMatrix<MatrixType, LowRankUpdates<MatrixType, PreconditionerType>>>
                                                 error_matrix_2;
      std::unique_ptr<ConstrainedIdentityMatrix> constrained_identity;

      Preconditioners::MultiVector2 precomputed_vectors;
      std::vector<MultiVector2>     right_lowrank_matrix; // V^T
      std::vector<MultiVector2>     left_lowrank_matrix;  // W

      std::vector<MultiVector2> factorization; // LU decomposition

      Preconditioners::MultiVector2         gaussmatrix;
      std::vector<LAPACKFullMatrix<double>> hessenberg_matrix;
      std::vector<std::vector<int>> ipiv; // permutation of the LU factorization

      std::vector<IndexSet> owned_partitioning; // of the pressure indices
      AdditionalData        data;
      unsigned int          current_max_number;

      // temporary vector
      mutable TrilinosWrappers::MPI::Vector tmp_v;
    };

    //////////////////////////////////////
    // Implementation
    //////////////////////////////////////
    template <typename MatrixType, typename PreconditionerType>
    template <typename VectorType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::vmult(
      VectorType &      trg,
      const VectorType &src) const
    {
      apply_updated_preconditioner(trg, src, current_max_number);
    }

    template <typename MatrixType, typename PreconditionerType>
    template <typename VectorType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::Tvmult(
      VectorType &      trg,
      const VectorType &src) const
    {
      apply_updated_preconditioner_transposed(trg, src, current_max_number);
    }



    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::apply_update(
      TrilinosWrappers::MPI::Vector &      trg,
      const TrilinosWrappers::MPI::Vector &src,
      const unsigned int                   number,
      const bool                           transposed) const
    {
      if (transposed)
        {
          TrilinosWrappers::MPI::Vector tmp(complete_index_set(data.rank));

          // tmp_v.reinit(src);
          // E    =     I - S \hat{S}^-1
          //   \approx  U_r \Sigma_r V_r^T

          // U_r^T \hat{S}^-T v
          left_lowrank_matrix[number].Tvmult(tmp, src);

          // ( I - D_r V_r^T U_r )^-T U_r^T \hat{S}^-T v
          solve_transposed_factorization(tmp, number);

          // (D_r V_r^T)^T ( I - D_r V_r^T U_r )^-T U_r^T \hat{S}^-T v
          right_lowrank_matrix[number].vmult(trg, tmp);

          // [ I + (D_r V_r^T)^T ( I - D_r V_r^T U_r )^-T U_r^T ] \hat{S}^-T v
          trg.add(1.0, src);
          // set value at constrained index to zero
          constrained_identity->vmult(trg);
        }
      else
        {
          TrilinosWrappers::MPI::Vector tmp(complete_index_set(data.rank));
          // TrilinosWrappers::MPI::Vector tmp2(
          //  tmp); // small ones, so there is no need to distribute
          // them over the processors
          // tmp_v.reinit(src);
          // E    =     I - S \hat{S}^-1
          //   \approx  U_r \Sigma_r V_r^T

          // D_r V_r^T v
          right_lowrank_matrix[number].Tvmult(tmp, src);

          // ( I - D_r V_r^T U_r )^-1 D_r V_r^T v
          solve_factorization(tmp, number);

          // U_r ( I - D_r V_r^T U_r )^-1 D_r V_r^T v
          left_lowrank_matrix[number].vmult(trg, tmp); // FIXME Abbruch

          // [ I + U_r ( I - D_r V_r^T U_r )^-1 D_r V_r^T ] v
          trg.add(1.0, src);
          // set value at constrained index to zero
          constrained_identity->vmult(trg);
        }
    }



    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::apply_update(
      Preconditioners::MultiVector2 &      trg,
      const Preconditioners::MultiVector2 &src,
      const unsigned int                   number,
      const bool                           transposed) const
    {
      // Assert(false, ExcNotImplemented());
      if (transposed)
        {
          Preconditioners::MultiVector2 tmp;
          tmp.reinit(src.n_vectors(), data.rank);

          // tmp_v.reinit(src);
          // E    =     I - S \hat{S}^-1
          //   \approx  U_r \Sigma_r V_r^T

          // U_r^T \hat{S}^-T v
          left_lowrank_matrix[number].Tmmult(tmp, src);

          // ( I - D_r V_r^T U_r )^-T U_r^T \hat{S}^-T v
          solve_transposed_factorization(tmp, number);

          // (D_r V_r^T)^T ( I - D_r V_r^T U_r )^-T U_r^T \hat{S}^-T v
          right_lowrank_matrix[number].mmult(trg, tmp);

          // [ I + (D_r V_r^T)^T ( I - D_r V_r^T U_r )^-T U_r^T ] \hat{S}^-T
          trg.add(1.0, src);
          // set value at constrained index to zero
          constrained_identity->vmult(trg);
        }
      else
        {
          Preconditioners::MultiVector2 tmp;
          tmp.reinit(src.n_vectors(), data.rank);

          // E    =     I - S \hat{S}^-1
          //   \approx  U_r \Sigma_r V_r^T

          // D_r V_r^T v
          right_lowrank_matrix[number].Tmmult(tmp, src);

          // ( I - D_r V_r^T U_r )^-1 D_r V_r^T v
          solve_factorization(tmp, number);

          // U_r ( I - D_r V_r^T U_r )^-1 D_r V_r^T v
          left_lowrank_matrix[number].mmult(trg, tmp); // FIXME Abbruch

          // [ I + U_r ( I - D_r V_r^T U_r )^-1 D_r V_r^T ] v
          trg.add(1.0, src);
          // set value at constrained index to zero
          constrained_identity->vmult(trg);
        }
    }

    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::apply_update(
      Preconditioners::MultiVector2::Column &      trg,
      const Preconditioners::MultiVector2::Column &src,
      const unsigned int                           number,
      const bool                                   transposed) const
    {
      // Assert(false, ExcNotImplemented());
      if (transposed)
        {
          Preconditioners::MultiVector2::Column tmp;
          tmp.reinit(data.rank);

          // tmp_v.reinit(src);
          // E    =     I - S \hat{S}^-1
          //   \approx  U_r \Sigma_r V_r^T

          // U_r^T \hat{S}^-T v
          left_lowrank_matrix[number].Tvmult(tmp, src);

          // ( I - D_r V_r^T U_r )^-T U_r^T \hat{S}^-T v
          solve_transposed_factorization(tmp, number);

          // (D_r V_r^T)^T ( I - D_r V_r^T U_r )^-T U_r^T \hat{S}^-T v
          right_lowrank_matrix[number].vmult(trg, tmp);

          // [ I + (D_r V_r^T)^T ( I - D_r V_r^T U_r )^-T U_r^T ] \hat{S}^-T
          trg.add(1.0, src);
          // set value at constrained index to zero
          constrained_identity->vmult(trg);
        }
      else
        {
          Preconditioners::MultiVector2::Column tmp;
          tmp.reinit(data.rank);

          // E    =     I - S \hat{S}^-1
          //   \approx  U_r \Sigma_r V_r^T

          // D_r V_r^T v
          right_lowrank_matrix[number].Tvmult(tmp, src);

          // ( I - D_r V_r^T U_r )^-1 D_r V_r^T v
          solve_factorization(tmp, number);

          // U_r ( I - D_r V_r^T U_r )^-1 D_r V_r^T v
          left_lowrank_matrix[number].vmult(trg, tmp); // FIXME Abbruch

          // [ I + U_r ( I - D_r V_r^T U_r )^-1 D_r V_r^T ] v
          trg.add(1.0, src);
          // set value at constrained index to zero
          constrained_identity->vmult(trg);
        }
    }



    template <typename MatrixType, typename PreconditionerType>
    template <typename Block00PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::compute_update_vectors(
      const unsigned int               n_new_vectors,
      const AffineConstraints<double> &constraints,
      const unsigned int               number,
      // const std::shared_ptr<Block00PreconditionerType>
      // &block00_preconditioner)
      const SmartPointer<const Block00PreconditionerType>
        &block00_preconditioner)
    {
      compute_update_vectors<Block00PreconditionerType>(
        n_new_vectors,
        constraints,
        number,
        std::integral_constant<
          bool,
          IsBlock00PreconditionerType<Block00PreconditionerType>::
            value>(), // IsBlockMatrix<MatrixType>::value>(),
        block00_preconditioner);
    }


    template <typename MatrixType, typename PreconditionerType>
    template <typename Block00PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::compute_update_vectors(
      const unsigned int               n_new_vectors,
      const AffineConstraints<double> &constraints,
      const unsigned int               number,
      std::integral_constant<bool, true>,
      const SmartPointer<const Block00PreconditionerType>
        &block00_preconditioner)
    {
      Assert(false, ExcNotImplemented());
      std::ignore = n_new_vectors;
      std::ignore = constraints;
      std::ignore = number;
      std::ignore = block00_preconditioner;
    }


    template <typename MatrixType, typename PreconditionerType>
    template <typename BlockPreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::compute_update_vectors(
      const unsigned int               n_new_vectors,
      const AffineConstraints<double> &constraints,
      const unsigned int               number,
      std::integral_constant<bool, false>,
      const SmartPointer<const BlockPreconditionerType> &block_preconditioner)
    {
      (void)block_preconditioner;
      Assert(n_new_vectors > 0,
             ExcMessage("choose a positive number of update vectors"));

      if (number > 0)
        {
          error_matrix_2 = std::make_shared<
            ErrorMatrix<MatrixType,
                        LowRankUpdates<MatrixType, PreconditionerType>>>(
            // *matrix, *this, data.weight, *constrained_identity);
            *matrix,
            *this,
            1.0,
            *constrained_identity);
        }

      BlockUpdateBase<MatrixType, PreconditionerType>::
        compute_update_vectors_nonblock(n_new_vectors,
                                        constraints,
                                        number,
                                        owned_partitioning,
                                        *error_matrix,
                                        *error_matrix_2,
                                        precomputed_vectors,
                                        gaussmatrix,
                                        *matrix,
                                        *initial_preconditioner,
                                        left_lowrank_matrix,
                                        right_lowrank_matrix,
                                        hessenberg_matrix,
                                        data);
    }



    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::precompute_factorization(
      const unsigned int number)
    {
      switch (data.solver_type)
        {
            case random: { // Compute the transposed matrix, since Fortran
              // matrices are the transposed of C++ matrices
              // Compute I - V_r^T D_r U_r
              factorization[number].reinit(data.rank, data.rank);
              // factorization = V_r^T D_r U_r
              right_lowrank_matrix[number].Tmmult(factorization[number],
                                                  left_lowrank_matrix[number]);
              const char   type  = 'G';
              const int    kl    = data.rank;
              const int    ku    = data.rank;
              const double cfrom = 1.0;
              const double cto   = -1.0;
              const int    m     = data.rank;
              const int    n     = data.rank;
              const int    lda   = data.rank;
              int          info  = 0;
              const int    ierr  = 0;
              AssertThrow(ierr == 0, ExcTrilinosError(ierr));
              lascl(&type,
                    &kl,
                    &ku,
                    &cfrom,
                    &cto,
                    &m,
                    &n,
                    factorization[number].begin(), // result.data()
                    &lda,
                    &info);
              for (unsigned int i = 0; i < data.rank; ++i)
                {
                  factorization[number].get_columns()[i][i] += 1.0;
                }

              // Compute LU factorization
              ipiv[number].resize(n);
              getrf(&m,
                    &n,
                    factorization[number].begin(), // result.data(),
                    &lda,
                    ipiv[number].data(),
                    &info);


              Assert(info >= 0, ExcInternalError());
              Assert(info == 0, LACExceptions::ExcSingular());
              break;
            }
            case arnoldi: {
              // Compute I_r - H
              hessenberg_matrix[number] *= -1.0;

              for (unsigned int i = 0; i < data.rank; ++i)
                hessenberg_matrix[number](i, i) += 1.0;

              // Precompute LU factorization of I_r - H
              hessenberg_matrix[number].compute_lu_factorization();
              break;
            }
        }
    }

    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::solve_factorization(
      TrilinosWrappers::MPI::Vector &trg,
      const unsigned int             number) const
    {
      switch (data.solver_type)
        {
            case random: { // Assert(trg.size() == src.size(),
              //       ExcDimensionMismatch(trg.size(), src.size()));
              // Solve with precomputed LU factorization
              char trans = 'N'; // because Fortran use transposed order 'N';
              int  n     = data.rank; // factorization.n();
              int  nrhs  = 1;         // src.size();

              // Assert(n == nrhs, ExcDimensionMismatch(n, nrhs));
              Assert((unsigned int)n == ipiv[number].size(),
                     ExcDimensionMismatch(n, ipiv[number].size()));

              int lda = n; // length of rows since, the matrix entries are
                           // stored row-wise
              int ldb = n;

              int info = 0;

              getrs(&trans,
                    &n,
                    &nrhs,
                    factorization[number].begin(),
                    &lda,
                    ipiv[number].data(),
                    trg.begin(),
                    &ldb,
                    &info);

              Assert(info >= 0, ExcInternalError());
              Assert(info == 0, LACExceptions::ExcSingular());

              break;
            }
            case arnoldi: {
              // The methods of LAPACKFullMatrix are only for Vector<number>
              // vectors
              Vector<double> tmp(trg);
              Vector<double> tmp2(trg);

              // Apply (I-H)^{-1}
              hessenberg_matrix[number].solve(tmp);

              // Compute ((I-H)^-1 - I) v
              tmp.add(-1.0, tmp2);

              trg = tmp;
              break;
            }
        }
    }


    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::solve_factorization(
      Preconditioners::MultiVector2 &trg,
      const unsigned int             number) const
    {
      switch (data.solver_type)
        {
            case random: { // Assert(trg.size() == src.size(),
              //       ExcDimensionMismatch(trg.size(), src.size()));
              // Solve with precomputed LU factorization
              char trans = 'N'; // because Fortran use transposed order 'N';
              int  n     = data.rank;       // factorization.n();
              int  nrhs  = trg.n_vectors(); // src.size();

              Assert((unsigned int)n == ipiv[number].size(),
                     ExcDimensionMismatch(n, ipiv[number].size()));

              int lda = n; // length of rows since, the matrix entries are
                           // stored row-wise
              int ldb = n;

              int info = 0;



              getrs(&trans,
                    &n,
                    &nrhs,
                    factorization[number].begin(),
                    &lda,
                    ipiv[number].data(),
                    trg.begin(),
                    &ldb,
                    &info);

              Assert(info >= 0, ExcInternalError());
              Assert(info == 0, LACExceptions::ExcSingular());

              break;
            }
            case arnoldi: {
              // The methods of LAPACKFullMatrix are only for Vector<number>
              // vectors
              for (unsigned int col = 0; col < trg.n_vectors(); ++col)
                {
                  Vector<double> tmp(trg.size());
                  Vector<double> tmp2(trg.size());
                  for (unsigned int i = 0; i < trg.size(); ++i)
                    {
                      tmp[i]  = trg[col][i];
                      tmp2[i] = trg[col][i];
                    }

                  // Apply (I-H)^{-1}
                  hessenberg_matrix[number].solve(tmp);

                  // Compute ((I-H)^-1 - I) v
                  tmp.add(-1.0, tmp2);

                  for (unsigned int i = 0; i < trg.size(); ++i)
                    {
                      trg[col][i] = tmp[i];
                    }
                }
              break;
            }
        }
    }

    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::solve_factorization(
      Preconditioners::MultiVector2::Column &trg,
      const unsigned int                     number) const
    {
      switch (data.solver_type)
        {
            case random: { // Assert(trg.size() == src.size(),
              // Solve with precomputed LU factorization
              char trans = 'N'; // because Fortran use transposed order 'N';
              int  n     = data.rank; // factorization.n();
              int  nrhs  = 1;
              Assert((unsigned int)n == ipiv[number].size(),
                     ExcDimensionMismatch(n, ipiv[number].size()));

              int lda = n; // length of rows since, the matrix entries are
                           // stored row-wise
              int ldb = n;

              int info = 0;



              getrs(&trans,
                    &n,
                    &nrhs,
                    factorization[number].begin(),
                    &lda,
                    ipiv[number].data(),
                    trg.begin(),
                    &ldb,
                    &info);

              Assert(info >= 0, ExcInternalError());
              Assert(info == 0, LACExceptions::ExcSingular());

              break;
            }
            case arnoldi: {
              // Assert(false, ExcNotImplemented());
              // The methods of LAPACKFullMatrix are only for Vector<number>
              // vectors
              Vector<double> tmp(trg.size());
              Vector<double> tmp2(trg.size());

              for (int i = 0; i < trg.size(); ++i)
                {
                  tmp[i]  = trg[i];
                  tmp2[i] = trg[i];
                }

              // Apply (I-H)^{-1}
              hessenberg_matrix[number].solve(tmp);

              // Compute ((I-H)^-1 - I) v
              tmp.add(-1.0, tmp2);

              trg = tmp;
              break;
            }
        }
    }



    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::
      solve_transposed_factorization(TrilinosWrappers::MPI::Vector &trg,
                                     const unsigned int number) const
    {
      switch (data.solver_type)
        {
            case random: {
              // Solve with precomputed LU factorization
              char trans = 'T';       // 'T';
              int  n     = data.rank; // factorization.n();
              int  nrhs  = 1;         // src.size();

              Assert((unsigned int)n == ipiv[number].size(),
                     ExcDimensionMismatch(n, ipiv[number].size()));

              int lda = n; // length of rows since, the matrix entries are
                           // stored row-wise
              int ldb = n;

              int info = 0;

              getrs(&trans,
                    &n,
                    &nrhs,
                    factorization[number].begin(),
                    &lda,
                    ipiv[number].data(),
                    trg.begin(),
                    &ldb,
                    &info);

              Assert(info >= 0, ExcInternalError());
              Assert(info == 0, LACExceptions::ExcSingular());
              break;
            }
            case arnoldi: {
              // The methods of LAPACKFullMatrix are only for Vector<number>
              // vectors
              Vector<double> tmp(trg);
              Vector<double> tmp2(trg);

              // Apply (I-H)^{-T}
              hessenberg_matrix[number].solve(tmp, true /*transpose*/);

              // Compute ((I-H)^-T - I) v
              tmp.add(-1.0, tmp2);

              trg = tmp;
              break;
            }
        }
    }


    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::
      solve_transposed_factorization(Preconditioners::MultiVector2 &trg,
                                     const unsigned int number) const
    {
      switch (data.solver_type)
        {
            case random: {
              // Solve with precomputed LU factorization
              char trans = 'T';             // 'T';
              int  n     = data.rank;       // factorization.n();
              int  nrhs  = trg.n_vectors(); // src.size();

              Assert((unsigned int)n == ipiv[number].size(),
                     ExcDimensionMismatch(n, ipiv[number].size()));

              int lda = n; // length of rows since, the matrix entries are
                           // stored row-wise
              int ldb = n;

              int info = 0;

              getrs(&trans,
                    &n,
                    &nrhs,
                    factorization[number].begin(),
                    &lda,
                    ipiv[number].data(),
                    trg.begin(),
                    &ldb,
                    &info);

              Assert(info >= 0, ExcInternalError());
              Assert(info == 0, LACExceptions::ExcSingular());
              break;
            }
            case arnoldi: {
              // Assert(false, ExcNotImplemented());
              // The methods of LAPACKFullMatrix are only for
              // Vector<number> vectors
              for (unsigned int col = 0; col < trg.n_vectors(); ++col)
                {
                  Vector<double> tmp(trg.size());
                  Vector<double> tmp2(trg.size());
                  for (unsigned int i = 0; i < trg.size(); ++i)
                    {
                      tmp[i]  = trg[col][i];
                      tmp2[i] = trg[col][i];
                    }

                  // Apply (I-H)^{-T}
                  hessenberg_matrix[number].solve(tmp, true /*transpose*/);

                  // Compute ((I-H)^-T - I) v
                  tmp.add(-1.0, tmp2);

                  for (unsigned int i = 0; i < trg.size(); ++i)
                    {
                      trg[col][i] = tmp[i];
                    }
                }
              break;
            }
        }
    }

    template <typename MatrixType, typename PreconditionerType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::
      solve_transposed_factorization(Preconditioners::MultiVector2::Column &trg,
                                     const unsigned int number) const
    {
      switch (data.solver_type)
        {
            case random: {
              // Solve with precomputed LU factorization
              char trans = 'T';       // 'T';
              int  n     = data.rank; // factorization.n();
              int  nrhs  = 1;

              Assert((unsigned int)n == ipiv[number].size(),
                     ExcDimensionMismatch(n, ipiv[number].size()));

              int lda = n; // length of rows since, the matrix entries are
                           // stored row-wise
              int ldb = n;

              int info = 0;

              getrs(&trans,
                    &n,
                    &nrhs,
                    factorization[number].begin(),
                    &lda,
                    ipiv[number].data(),
                    trg.begin(),
                    &ldb,
                    &info);

              Assert(info >= 0, ExcInternalError());
              Assert(info == 0, LACExceptions::ExcSingular());
              break;
            }
            case arnoldi: {
              // Assert(false, ExcNotImplemented());
              // The methods of LAPACKFullMatrix are only for Vector<number>
              // vectors
              Vector<double> tmp(trg.size());
              Vector<double> tmp2(trg.size());
              for (int i = 0; i < trg.size(); ++i)
                {
                  tmp[i]  = trg[i];
                  tmp2[i] = trg[i];
                }

              // Apply (I-H)^{-T}
              hessenberg_matrix[number].solve(tmp, true /*transpose*/);

              // Compute ((I-H)^-T - I) v
              tmp.add(-1.0, tmp2);

              trg = tmp;
              break;
            }
        }
    }



    template <typename MatrixType, typename PreconditionerType>
    template <typename VectorType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::
      apply_updated_preconditioner(VectorType &       trg,
                                   const VectorType & src,
                                   const unsigned int max_number) const
    {
      VectorType tmp_v; // TODO
      if (data.symmetric)
        {
          const unsigned int symmetric_version = 1; // in {1,2,3}
          if (symmetric_version == 1)
            {
              VectorType tmp(src);
              tmp_v.reinit(src);

              // Apply right update
              tmp = src;
              for (unsigned int i = max_number; i > 0; --i)
                {
                  apply_update(tmp_v, tmp, i - 1, false);
                  tmp = tmp_v;
                }

              // \hat{S}^-1_{upd,r} v
              initial_preconditioner->vmult(tmp, tmp_v);
              tmp *= data.weight;

              // S \hat{S}^-1_{upd,r} v
              matrix->vmult(tmp_v, tmp);

              // (I - S \hat{S}^-1_{upd,r}) v
              tmp_v.add(-1.0, src);
              tmp_v *= -1.0;

              // Apply left update
              // \hat{S}^{-1}_{upd,l} (I - S \hat{S}^-1_{upd,r}) v
              initial_preconditioner->Tvmult(trg, tmp_v);
              trg *= data.weight;

              for (unsigned int i = 0; i < max_number; ++i)
                {
                  apply_update(tmp_v, trg, i, true /* transposed */);
                  trg = tmp_v;
                }

              trg += tmp;
            }
          else if (symmetric_version == 2)
            {
              VectorType tmp(src);
              tmp_v.reinit(src);

              // Apply left update
              // \hat{S}^-1_{upd,l} v
              initial_preconditioner->Tvmult(tmp_v, src);
              tmp_v *= data.weight;

              for (unsigned int i = 0; i < max_number; ++i)
                {
                  apply_update(tmp, tmp_v, i, true /* transposed */);
                  tmp_v = tmp;
                }


              // S \hat{S}^-1_{upd,l} v
              matrix->vmult(tmp_v, tmp);

              // (I - S \hat{S}^-1_{upd,r}) v
              tmp_v.add(-1.0, src);
              tmp_v *= -1.0;

              // Apply right update
              for (unsigned int i = max_number; i > 0; --i)
                {
                  apply_update(trg, tmp_v, i - 1, false);
                  tmp_v = trg;
                }

              // \hat{S}^-1_{upd,r} v
              initial_preconditioner->vmult(trg, tmp_v);
              trg *= data.weight;

              trg += tmp;
            }
          else if (symmetric_version == 3)
            {
              VectorType tmp(src);
              tmp_v.reinit(src);

              // Apply right update
              tmp = src;
              for (unsigned int i = max_number; i > 0; --i)
                {
                  apply_update(tmp_v, tmp, i - 1, false);
                  tmp = tmp_v;
                }

              // \hat{S}^-1_{upd,r} v
              initial_preconditioner->vmult(tmp, tmp_v);
              tmp *= data.weight;

              // S \hat{S}^-1_{upd,r} v
              matrix->vmult(tmp_v, tmp);

              // Apply left update
              // \hat{S}^{-1}_{upd,l} S \hat{S}^-1_{upd,r} v
              initial_preconditioner->Tvmult(trg, tmp_v);
              trg *= data.weight;

              for (unsigned int i = 0; i < max_number; ++i)
                {
                  apply_update(tmp_v, trg, i, true /* transposed */);
                  trg = tmp_v;
                }
            }
        }
      else
        {
          VectorType tmp(src);
          tmp_v.reinit(src);
          tmp = src;
          for (unsigned int i = max_number; i > 0; --i)
            {
              apply_update(tmp_v, tmp, i - 1, false);
              tmp = tmp_v;
            }

          // \hat{S}^-1 [ I + U_r ( I - D_r V_r^T U_r )^-1 D_r V_r^T ] v
          initial_preconditioner->vmult(trg, tmp_v);
          trg *= data.weight;
        }
    }


    template <typename MatrixType, typename PreconditionerType>
    template <typename VectorType>
    void
    LowRankUpdates<MatrixType, PreconditionerType>::
      apply_updated_preconditioner_transposed(
        VectorType &       trg,
        const VectorType & src,
        const unsigned int max_number) const
    {
      // using VectorType = TrilinosWrappers::MPI::Vector;
      VectorType tmp_v; // FIXME
      if (data.symmetric)
        {
          bool symmetric_version_1 = true; // false;
          if (symmetric_version_1)
            {
              VectorType tmp(src);
              tmp_v.reinit(src);

              // Apply right update
              tmp = src;
              for (unsigned int i = max_number; i > 0; --i)
                {
                  apply_update(tmp_v, tmp, i - 1, false);
                  tmp = tmp_v;
                }

              // \hat{S}^-1_{upd,r} v
              initial_preconditioner->vmult(tmp, tmp_v);
              tmp *= data.weight;

              // S \hat{S}^-1_{upd,r} v
              matrix->vmult(tmp_v, tmp);

              // (I - S \hat{S}^-1_{upd,r}) v
              tmp_v.add(-1.0, src);
              tmp_v *= -1.0;

              // Apply left update
              // \hat{S}^{-1}_{upd,l} (I - S \hat{S}^-1_{upd,r}) v
              initial_preconditioner->Tvmult(trg, tmp_v);
              trg *= data.weight;

              for (unsigned int i = 0; i < max_number; ++i)
                {
                  apply_update(tmp_v, trg, i, true /* transposed */);
                  trg = tmp_v;
                }

              // \hat{S}^-1_{upd,r} v + \hat{S}^{-1}_{upd,l} (I - S
              // \hat{S}^-1_{upd,r}) v
              trg += tmp;
            }
          else
            {
              VectorType tmp(src);
              tmp_v.reinit(src);

              // Apply left update
              // \hat{S}^-1_{upd,l} v
              initial_preconditioner->Tvmult(tmp_v, src);
              tmp_v *= data.weight;

              for (unsigned int i = 0; i < max_number; ++i)
                {
                  apply_update(tmp, tmp_v, i, true /* transposed */);
                  tmp_v = tmp;
                }


              // S \hat{S}^-1_{upd,l} v
              matrix->vmult(tmp_v, tmp);

              // (I - S \hat{S}^-1_{upd,r}) v
              tmp_v.add(-1.0, src);
              tmp_v *= -1.0;

              // Apply right update
              for (unsigned int i = max_number; i > 0; --i)
                {
                  apply_update(trg, tmp_v, i - 1, false);
                  tmp_v = trg;
                }

              // \hat{S}^-1_{upd,r} v
              initial_preconditioner->vmult(trg, tmp_v);
              trg *= data.weight;

              trg += tmp;
            }
        }
      else
        {
          tmp_v.reinit(src);
          // E    =     I - S \hat{S}^-1
          //   \approx  U_r \Sigma_r V_r^T

          // \hat{S}^-T v
          initial_preconditioner->Tvmult(tmp_v, src);
          tmp_v *= data.weight;

          for (unsigned int i = 0; i < max_number; ++i)
            {
              apply_update(trg, tmp_v, i, true /* transposed */);
              tmp_v = trg;
            }
        }
    }



  } // namespace Preconditioners
} // namespace dealii
#endif
