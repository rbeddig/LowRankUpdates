/**
 * @file schur_preconditioner_selector.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef SCHUR_PRECONDITIONER_SELECTOR
#define SCHUR_PRECONDITIONER_SELECTOR

// Deal ii
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

// preconditioners
#include <linear_algebra/block_schur_preconditioner.hpp>
#include <linear_algebra/inverse_matrix.hpp>
#include <linear_algebra/schur_complement.hpp>
#include <preconditioners/bfbt.h>
#include <preconditioners/updates2.h>
#include <preconditioners/utilities.h>

DEAL_II_NAMESPACE_OPEN

namespace Preconditioners
{
  /**
   * Implements the (SIMPLE) Schur complement preconditioner given as
   * \f[  P_S = (B diag(A)^{-1} B^T)^{-1}   \f]
   * for a saddle-point system
   * \f[
   *     \begin{pmatrix} A & B^T \\ B & 0 \end{pmatrix}
   *     \begin{pmatrix} u \\ p \end{pmatrix}
   *     =
   *     \begin{pmatrix} f \\ 0 \end{pmatrix}
   * \f]
   */
  class OrigSchurPreconditioner : public Subscriptor
  {
  public:
    /**
     * Options for the Poisson-type solver needed in the preconditioner.
     */
    struct AdditionalData
    {
    public:
      AdditionalData(const bool   precondition_cg     = true,
                     const double tolerance           = 1e-6,
                     const bool   solve_only_with_amg = true,
                     const bool   use_ic              = false,
                     const std::vector<unsigned int> &constrained_dofs =
                       std::vector<unsigned int>(0))
        : precondition_cg(precondition_cg)
        , tolerance(tolerance)
        , solve_only_with_amg(solve_only_with_amg)
        , use_ic(use_ic)
        , constrained_dofs(constrained_dofs)
      {}

      bool   precondition_cg;     /** Solve with preconditioned CG. */
      double tolerance;           /** Relative solver tolerance. */
      bool   solve_only_with_amg; /** Use only 5 AMG cycles to approximate the
                                     inverse. */
      bool use_ic; /** Use an incomplete Cholesky factorization. */
      std::vector<unsigned int>
        constrained_dofs; /** Constrained pressure dofs. */
    };

    /** Constructor. Needs to be initialized before usage with @ref initialize(). */
    OrigSchurPreconditioner(const AdditionalData &data = AdditionalData())
      : Subscriptor()
      , data(data)
    {}

    /**
     * @brief Constructor. Needs to be initialized before usage with \ref initialize().
     * @param matrix saddle-point block matrix
     * @param data contains solver options
     */
    OrigSchurPreconditioner(const TrilinosWrappers::BlockSparseMatrix &matrix,
                            const AdditionalData &data = AdditionalData())
      : Subscriptor()
      , system_matrix(&matrix)
      , data(data)
    {}

    /**
     * @brief Initialize the preconditioner,
     * @param matrix saddle-point block matrix
     * @param data contains solver options
     */
    void
    initialize(const TrilinosWrappers::BlockSparseMatrix &matrix,
               const AdditionalData &input_data = AdditionalData())
    {
      data = input_data;

      system_matrix = &matrix;

      // initialize scaling vector with diag(A)^-1
      scaling_vector.reinit(complete_index_set(system_matrix->block(0, 1).m()));


      // The functions for extracting the diagonal or inverse row sums
      // are only defined for Epetra_Vectors
      Epetra_Vector tmp_scaling(scaling_vector.trilinos_partitioner());

      system_matrix->block(0, 0).trilinos_matrix().ExtractDiagonalCopy(
        tmp_scaling);

      unsigned int N = system_matrix->block(0, 0).n();
      double       diag;
      for (unsigned int i = 0; i < N; ++i)
        {
          diag = tmp_scaling[i];
          if (diag != 0.0)
            tmp_scaling[i] = 1. / diag;
          else
            tmp_scaling[i] = 0.0;
        }

      // Fill an Epetra_FEVector from the Epetra_Vector such that we can
      // use the TrilinosWrappers::MPI::Vector class for the scaling
      // operations
      double *array_view = scaling_vector.trilinos_vector()[0];
      tmp_scaling.ExtractCopy(array_view, 1);

      if (data.precondition_cg || data.solve_only_with_amg || data.use_ic)
        { // compute B diag(A)^{-1} B^T
          std::vector<int> indices(std::begin(data.constrained_dofs),
                                   std::end(data.constrained_dofs));
          perform_modified_mmult(system_matrix->block(0, 1),
                                 system_matrix->block(0, 1),
                                 approx_schur_complement,
                                 scaling_vector,
                                 true,
                                 indices);
          wrap_approx_schur_complement.reinit(approx_schur_complement);
        }
      else
        {
          ata_approx_schur_complement.initialize(
            system_matrix->block(0, 1),
            scaling_vector,
            0, // /*already shifted*/scaling_vector.size() /*shift*/,
            data.constrained_dofs); // TODO constrained_dofs}
        }

      if (data.use_ic)
        {
          ic_preconditioner.initialize(approx_schur_complement);
          wrap_ic_preconditioner.reinit(ic_preconditioner);
        }
      else if (data.precondition_cg || data.solve_only_with_amg)
        {
          TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
          amg_data.elliptic              = true;
          amg_data.higher_order_elements = true;
          amg_data.smoother_sweeps       = 2;
          amg_data.aggregation_threshold = 0.02;
          amg_data.output_details        = false;
          if (data.solve_only_with_amg)
            amg_data.n_cycles = 5;
          else
            amg_data.n_cycles = 1;
          amg_preconditioner.initialize(approx_schur_complement, amg_data);
          wrap_amg_preconditioner.reinit(amg_preconditioner);
        }
    }

    /** for compiling */
    void
    reinit_A(const TrilinosWrappers::SparseMatrix &matrix)
    {
      std::ignore = matrix;
    }

    /**
     * @brief Applies the preconditioner to a vector.
     * trg = *this * src
     * @param trg Result.
     * @param src Source vector.
     */
    void
    vmult(TrilinosWrappers::MPI::Vector &      trg,
          const TrilinosWrappers::MPI::Vector &src) const
    {
      if (data.use_ic)
        {
          wrap_ic_preconditioner.vmult(trg, src);
        }
      else if (data.solve_only_with_amg)
        {
          wrap_amg_preconditioner.vmult(trg, src);
        }
      else
        {
          SolverControl             solver_control(100, // src.size(),
                                       data.tolerance * src.l2_norm());
          SolverCG<LA::MPI::Vector> solver(solver_control);
          try
            {
              if (data.precondition_cg)
                solver.solve(wrap_approx_schur_complement,
                             trg,
                             src,
                             wrap_amg_preconditioner);
              else
                solver.solve(ata_approx_schur_complement,
                             trg,
                             src,
                             PreconditionIdentity());
            }
          catch (SolverControl::NoConvergence &)
            {}
        }
    }

    /**
     * @brief Applies the transposed preconditioner to a vector.
     * trg = *this * src
     * @param trg Result.
     * @param src Source vector.
     */
    void
    Tvmult(TrilinosWrappers::MPI::Vector &      trg,
           const TrilinosWrappers::MPI::Vector &src) const
    {
      if (data.use_ic)
        {
          wrap_ic_preconditioner.vmult(trg, src);
        }
      else if (data.solve_only_with_amg)
        {
          wrap_amg_preconditioner.vmult(trg, src);
        }
      else
        {
          SolverControl             solver_control(100, // src.size(),
                                       data.tolerance * src.l2_norm());
          SolverCG<LA::MPI::Vector> solver(solver_control);
          try
            {
              if (data.precondition_cg)
                solver.solve(wrap_approx_schur_complement,
                             trg,
                             src,
                             wrap_amg_preconditioner);
              else
                solver.solve(ata_approx_schur_complement,
                             trg,
                             src,
                             PreconditionIdentity());
            }
          catch (SolverControl::NoConvergence &)
            {}
        }
    }

    /**
     * Matrix-vector product
     * trg = *this * src
     */
    void
    vmult(Preconditioners::MultiVector2 &      trg,
          const Preconditioners::MultiVector2 &src) const
    {
      if (data.use_ic)
        {
          wrap_ic_preconditioner.vmult(trg, src);
        }
      else if (data.solve_only_with_amg)
        {
          wrap_amg_preconditioner.vmult(trg, src);
        }
      else
        {
          for (unsigned int i = 0; i < src.n_vectors(); ++i)
            {
              Preconditioners::MultiVector2::Column tmp_src(
                src.trilinos_vector(), i, View);
              Preconditioners::MultiVector2::Column tmp_trg(
                trg.trilinos_vector(), i, View);
              SolverControl solver_control(100, // tmp_src.size(),
                                           data.tolerance * tmp_src.l2_norm());
              SolverCG<Preconditioners::MultiVector2::Column> solver(
                solver_control);
              try
                {
                  if (data.precondition_cg)
                    solver.solve(wrap_approx_schur_complement,
                                 tmp_trg,
                                 tmp_src,
                                 wrap_amg_preconditioner);
                  else
                    solver.solve(ata_approx_schur_complement,
                                 tmp_trg,
                                 tmp_src,
                                 PreconditionIdentity());
                }
              catch (SolverControl::NoConvergence &)
                {}
            }
        }
    }

    /**
     * Transposed matrix-vector product
     * trg = (*this)^T * src
     */
    void
    Tvmult(Preconditioners::MultiVector2 &      trg,
           const Preconditioners::MultiVector2 &src) const
    {
      if (data.use_ic)
        {
          wrap_ic_preconditioner.vmult(trg, src);
        }
      else if (data.solve_only_with_amg)
        {
          wrap_amg_preconditioner.vmult(trg, src);
        }
      else
        {
          for (unsigned int i = 0; i < src.n_vectors(); ++i)
            {
              Preconditioners::MultiVector2::Column tmp_src(
                src.trilinos_vector(), i, View);
              Preconditioners::MultiVector2::Column tmp_trg(
                trg.trilinos_vector(), i, View);
              SolverControl solver_control(100, // tmp_src.size(),
                                           data.tolerance * tmp_src.l2_norm());
              SolverCG<Preconditioners::MultiVector2::Column> solver(
                solver_control);
              try
                {
                  if (data.precondition_cg)
                    solver.solve(wrap_approx_schur_complement,
                                 tmp_trg,
                                 tmp_src,
                                 wrap_amg_preconditioner);
                  else
                    solver.solve(ata_approx_schur_complement,
                                 tmp_trg,
                                 tmp_src,
                                 PreconditionIdentity());
                }
              catch (SolverControl::NoConvergence &)
                {}
            }
        }
    }

    /**
     * Matrix-vector product
     * trg = *this * src
     */
    void
    vmult(Preconditioners::MultiVector2::Column &      trg,
          const Preconditioners::MultiVector2::Column &src) const
    {
      if (data.use_ic)
        {
          wrap_ic_preconditioner.vmult(trg, src);
        }
      else if (data.solve_only_with_amg)
        {
          wrap_amg_preconditioner.vmult(trg, src);
        }
      else
        {
          {
            SolverControl solver_control(100, // src.size(),
                                         data.tolerance * src.l2_norm());
            SolverCG<Preconditioners::MultiVector2::Column> solver(
              solver_control);
            try
              {
                if (data.precondition_cg)
                  solver.solve(wrap_approx_schur_complement,
                               trg,
                               src,
                               wrap_amg_preconditioner);
                else
                  solver.solve(ata_approx_schur_complement,
                               trg,
                               src,
                               PreconditionIdentity());
              }
            catch (SolverControl::NoConvergence &)
              {}
          }
        }
    }

    /**
     * Transposed matrix-vector product
     * trg = (*this)^T * src
     */
    void
    Tvmult(Preconditioners::MultiVector2::Column &      trg,
           const Preconditioners::MultiVector2::Column &src) const
    {
      if (data.use_ic)
        {
          wrap_ic_preconditioner.vmult(trg, src);
        }
      else if (data.solve_only_with_amg)
        {
          wrap_amg_preconditioner.vmult(trg, src);
        }
      else
        {
          {
            SolverControl solver_control(100, // src.size(),
                                         data.tolerance * src.l2_norm());
            SolverCG<Preconditioners::MultiVector2::Column> solver(
              solver_control);
            try
              {
                if (data.precondition_cg)
                  solver.solve(wrap_approx_schur_complement,
                               trg,
                               src,
                               wrap_amg_preconditioner);
                else
                  solver.solve(ata_approx_schur_complement,
                               trg,
                               src,
                               PreconditionIdentity());
              }
            catch (SolverControl::NoConvergence &)
              {}
          }
        }
    }

  private:
    SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
    TrilinosWrappers::MPI::Vector                           scaling_vector;

    Preconditioners::TrilinosATA   ata_approx_schur_complement;
    TrilinosWrappers::SparseMatrix approx_schur_complement;
    WrapperMatrix                  wrap_approx_schur_complement;

    TrilinosWrappers::PreconditionAMG amg_preconditioner;
    WrapperPreconditioner             wrap_amg_preconditioner;

    TrilinosWrappers::PreconditionIC ic_preconditioner;
    WrapperPreconditioner            wrap_ic_preconditioner;

    AdditionalData data;
  };


  /**
   * @brief Wrapper class to select a Schur complement preconditioner.
   */
  class SchurPreconditionerSelector : public Subscriptor
  {
  public:
    using MatrixType = DyCorePlanet::LinearAlgebra::ApproxSchurComplement<
      TrilinosWrappers::BlockSparseMatrix,
      TrilinosWrappers::PreconditionBase>;

    /** Default constructor. */
    SchurPreconditionerSelector()
      : Subscriptor()
    {}

    /**
     * Constructor.
     * @param name Schur complement preconditioner name (bfbt, orig)
     * @param system_matrix saddle-point block matrix
     * @param partitioning Index sets for the velocity and pressure.
     */
    SchurPreconditionerSelector(
      const std::string &                        name,
      const TrilinosWrappers::BlockSparseMatrix &system_matrix,
      const std::vector<IndexSet> &              partitioning)
      : preconditioner_name(name)
      , system_matrix(&system_matrix)
      , owned_partitioning(partitioning)
      , rank(0)
      , update_initialized(false)
    {}

    /** Destructor. */
    ~SchurPreconditionerSelector()
    {}

    /** Select a Schur complement preconditioner ('bfbt' for BFBt/LSC,'orig' for
     * SIMPLE). */
    void
    select(const std::string &name)
    {
      preconditioner_name = name;
    }


    /**
     * @brief Applies the preconditioner to a vector.
     * trg = *this * src
     * @param trg Result.
     * @param src Source vector.
     */
    template <typename VectorType>
    void
    vmult(VectorType &trg, const VectorType &src) const
    {
      if (preconditioner_name == "bfbt")
        if (bfbt_update_data.rank == 0 || !update_initialized) // no update
          bfbt_preconditioner.vmult(trg, src);
        else
          {
            bfbt_updated_preconditioner.vmult(trg, src);
          }
      else if (preconditioner_name == "orig")
        {
          if (orig_update_data.rank == 0 ||
              !update_initialized) // FIXME: richtig Bedingung? ||
                                   // !update_initialized)
            {
              orig_preconditioner.vmult(trg, src);
            }
          else
            {
              orig_updated_preconditioner.vmult(trg, src);
            }
        }
    }


    /**
     * @brief Applies the transposed preconditioner to a vector.
     * trg = *this * src
     * @param trg Result.
     * @param src Source vector.
     */
    template <typename VectorType>
    void
    Tvmult(VectorType &trg, const VectorType &src) const
    {
      if (preconditioner_name == "bfbt")
        if (bfbt_update_data.rank == 0 || !update_initialized) // no update
          bfbt_preconditioner.Tvmult(trg, src);
        else
          bfbt_updated_preconditioner.Tvmult(trg, src);
      else if (preconditioner_name == "orig")
        {
          if (orig_update_data.rank == 0 || !update_initialized)
            {
              orig_preconditioner.Tvmult(trg, src);
            }
          else
            {
              orig_updated_preconditioner.Tvmult(trg, src);
            }
        }
    }


    /**
     * @brief Applies the preconditioner to a vector.
     * trg = *this * src
     * @param trg Result.
     * @param src Source vector.
     */
    void
    vmult(Preconditioners::MultiVector2 &      trg,
          const Preconditioners::MultiVector2 &src) const
    {
      for (unsigned int i = 0; i < src.n_vectors(); ++i)
        {
          if (preconditioner_name == "bfbt")
            {
              if (bfbt_update_data.rank == 0 ||
                  !update_initialized) // no update
                bfbt_preconditioner.vmult(trg, src);
              else
                bfbt_updated_preconditioner.vmult(trg, src);
            }
          else if (preconditioner_name == "orig")
            {
              if (orig_update_data.rank == 0 || !update_initialized)
                orig_preconditioner.vmult(trg, src);
              else
                orig_updated_preconditioner.vmult(trg, src);
            }
        }
    }


    /**
     * @brief Applies the transposed preconditioner to a vector.
     * trg = *this * src
     * @param trg Result.
     * @param src Source vector.
     */
    void
    Tvmult(Preconditioners::MultiVector2 &      trg,
           const Preconditioners::MultiVector2 &src) const
    {
      for (unsigned int i = 0; i < src.n_vectors(); ++i)
        {
          if (preconditioner_name == "bfbt")
            {
              if (bfbt_update_data.rank == 0 ||
                  !update_initialized) // no update
                bfbt_preconditioner.Tvmult(trg, src);
              else
                bfbt_updated_preconditioner.Tvmult(trg, src);
            }
          else if (preconditioner_name == "orig")
            {
              if (orig_update_data.rank == 0 || !update_initialized)
                orig_preconditioner.Tvmult(trg, src);
              else
                orig_updated_preconditioner.Tvmult(trg, src);
            }
        }
    }

    /**
     * @brief Initializes the Schur complement preconditioner.
     */
    void
    initialize()
    {
      Assert(system_matrix->block(0, 0).trilinos_matrix().Filled(),
             ExcInternalError());
      if (preconditioner_name == "bfbt")
        {
          bfbt_preconditioner.initialize(*system_matrix, bfbt_data);
        }
      else if (preconditioner_name == "orig")
        {
          orig_preconditioner.initialize(*system_matrix, orig_data);
        }
    }


    /**
     * @brief Initializes the BFBt/LSC method with a scaling using the @ref preconditioner_matrix.
     */
    void
    initialize(const TrilinosWrappers::SparseMatrix &preconditioner_matrix)
    {
      if (preconditioner_name == "bfbt")
        {
          // preconditioner_matrix defines the scaling
          bfbt_preconditioner.initialize(*system_matrix,
                                         preconditioner_matrix,
                                         bfbt_data);
        }
    }



    /** @brief Initializes the update. */
    void
    initialize_update(const MatrixType &               in_schur_complement,
                      const AffineConstraints<double> &constraints)
    {
      block_update = false;
      if (rank != 0) // else: no update
        {
          if (preconditioner_name == "bfbt")
            {
              const std::vector<IndexSet> partitioning =
                in_schur_complement.get_partitioning();
              bfbt_updated_preconditioner.initialize(in_schur_complement,
                                                     bfbt_preconditioner,
                                                     partitioning,
                                                     constraints,
                                                     bfbt_update_data);
              update_initialized = true;
            }
          else if (preconditioner_name == "orig")
            {
              const std::vector<IndexSet> partitioning =
                in_schur_complement.get_partitioning();
              orig_updated_preconditioner.initialize(in_schur_complement,
                                                     orig_preconditioner,
                                                     partitioning,
                                                     constraints,
                                                     orig_update_data);
              update_initialized = true;
            }
          else
            {
              Assert(false, ExcNotImplemented());
            }
        }
    }


    /**
     * @brief Initialize the update.
     *
     * @tparam Block00PreconditionerType
     * @param block_matrix Saddle-point matrix.
     * @param schur_complement (Approximate) Schur complement.
     * @param block00_preconditioner Initial preconditioner.
     * @param constraints Constraints.
     */
    template <typename Block00PreconditionerType>
    void
    initialize_update(const TrilinosWrappers::BlockSparseMatrix &block_matrix,
                      const MatrixType &               schur_complement,
                      const Block00PreconditionerType &block00_preconditioner,
                      const AffineConstraints<double> &constraints)
    {
      block_update = true;
      // TODO fill data
      if (rank != 0) // else: no update
        {
          if (preconditioner_name == "bfbt")
            {
              bfbt_updated_preconditioner.initialize(block_matrix,
                                                     schur_complement,
                                                     block00_preconditioner,
                                                     bfbt_preconditioner,
                                                     owned_partitioning,
                                                     constraints,
                                                     bfbt_update_data);
            }
          else if (preconditioner_name == "orig")
            {
              orig_updated_preconditioner.initialize(block_matrix,
                                                     schur_complement,
                                                     block00_preconditioner,
                                                     orig_preconditioner,
                                                     owned_partitioning,
                                                     constraints,
                                                     orig_update_data);
            }
          else
            {
              Assert(false, ExcNotImplemented());
            }
          update_initialized = true;
        }
    }

    /**
     * @brief reinits the upper left block of the saddle-point matrix used in the BFBt/LSC
     * preconditioner
     */
    void
    reinit_A(const TrilinosWrappers::SparseMatrix &matrix)
    {
      if (preconditioner_name == "bfbt")
        {
          bfbt_preconditioner.reinit_A(matrix);
          if (update_initialized)
            bfbt_updated_preconditioner.reinit_A(matrix);
        }
      else if (preconditioner_name == "orig")
        {
          orig_preconditioner.reinit_A(matrix);
          if (update_initialized)
            orig_updated_preconditioner.reinit_A(matrix);
        }
      else
        Assert(false, ExcNotImplemented());
    }


    /** @brief Sets the parameters for the BFBt/LSC preconditioner. */
    void
    set_data(const Preconditioners::TrilinosBFBt::AdditionalData &data)
    {
      bfbt_data = data;
    }


    /**
     * @brief Sets the parameters for the Schur complement preconditioner \f$P_S = (B
     * diag(A)^{-1} B^T)^{-1}\f$.
     */
    void
    set_data(
      const Preconditioners::OrigSchurPreconditioner::AdditionalData &data)
    {
      orig_data = data;
    }


    /**
     * @brief Sets the update data.
     *
     * @tparam UpdateDataType
     * @param data Struct with update data.
     */
    template <typename UpdateDataType>
    void
    set_data(const UpdateDataType &data)
    {
      if (preconditioner_name == "bfbt")
        {
          bfbt_update_data.rank        = data.rank;
          bfbt_update_data.solver_type = data.solver_type;
          bfbt_update_data.number      = data.number;
          bfbt_update_data.symmetric   = data.symmetric;
          bfbt_update_data.weight      = data.weight;
        }
      else if (preconditioner_name == "orig")
        {
          orig_update_data.rank        = data.rank;
          orig_update_data.solver_type = data.solver_type;
          orig_update_data.number      = data.number;
          orig_update_data.symmetric   = data.symmetric;
          orig_update_data.weight      = data.weight;
        }
      rank = data.rank; // TODO only use rank once without the data
      // structures
    }

  private:
    std::string                                             preconditioner_name;
    SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
    std::vector<IndexSet>                                   owned_partitioning;

    // Optional preconditioners
    Preconditioners::TrilinosBFBt            bfbt_preconditioner;
    Preconditioners::OrigSchurPreconditioner orig_preconditioner;

    // and the corresponding data structs.
    Preconditioners::TrilinosBFBt::AdditionalData            bfbt_data;
    Preconditioners::OrigSchurPreconditioner::AdditionalData orig_data;

    using BFBtUpdateType = LowRankUpdates<MatrixType, TrilinosBFBt>;
    BFBtUpdateType::AdditionalData bfbt_update_data;
    BFBtUpdateType                 bfbt_updated_preconditioner;

    using OrigUpdateType = LowRankUpdates<MatrixType, OrigSchurPreconditioner>;
    OrigUpdateType::AdditionalData orig_update_data;
    OrigUpdateType                 orig_updated_preconditioner;

    unsigned int rank;
    bool         block_update;
    bool         update_initialized;
  };

} // namespace Preconditioners
DEAL_II_NAMESPACE_CLOSE
#endif