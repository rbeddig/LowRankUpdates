/**
 * @file bfbt.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef BFBT
#define BFBT

// Deal ii
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>


// STL

// Trilinos
#include <Epetra_CrsMatrix.h>
#include <Epetra_FEVector.h>
#include <Epetra_Vector.h>



// own headers
#include <preconditioners/ata.h>
#include <preconditioners/utilities.h>
#include <preconditioners/wrappers.h>



namespace dealii
{
  namespace Preconditioners
  {
    enum PoissonTypeSolver
    {
      Direct,
      IC,
      AMG,
      CG,
      PCG_AMG
    };

    /**
     * @brief BFBt or LSC preconditioner for systems where the matrix blocks are stored as @ref dealii::TrilinosWrappers::SparseMatrix
     *
     * For a system
     * \f[
     *     \begin{pmatrix} A & B^T \\ B & 0 \end{pmatrix}
     *     \begin{pmatrix} u \\ p \end{pmatrix}
     *     =
     *     \begin{pmatrix} f \\ 0 \end{pmatrix}
     * \f]
     * the BFBt Schur complement preconditioner is given as
     * \f[
     *     P_S = (B B^T)^{-1} B A B^T (B B^T)^{-1}.
     * \f]
     *
     * \cite ElmSW14
     */
    class TrilinosBFBt : public dealii::Subscriptor
    {
    public:
      using MatrixType      = dealii::TrilinosWrappers::SparseMatrix;
      using BlockMatrixType = dealii::TrilinosWrappers::BlockSparseMatrix;

      /**
       * @brief Options for the BFBt/LSC preconditioner
       */
      struct AdditionalData
      {
      public:
        /**
         * @brief Construct a new Additional Data object
         *
         * @param solver_type Solver for the inner Poisson-type problems.
         * @param tolerance Solver tolerance for the inner iterative solver for the Poisson-type problems.
         * @param maxiters Maximum of inner iterations.
         * @param use_scaling Use a scaling for the preconditioner (Set to true for the LSC preconditioner.)
         * @param scale_with_inv_row_sum Use the inverse row sum of the scaling matrix.
         * @param constrained_dofs Constrained pressure dofs (required to obtain an invertible preconditioner)
         * @param AMG_aggregation_threshold Aggregation threshold for the algebraic multigrid that can be optionally used for the Poisson-type problems.
         */
        AdditionalData(const PoissonTypeSolver solver_type            = CG,
                       const double            tolerance              = 1e-6,
                       const unsigned int      maxiters               = 100,
                       const bool              use_scaling            = true,
                       const bool              scale_with_inv_row_sum = false,
                       const std::vector<unsigned int> &constrained_dofs =
                         std::vector<unsigned int>(0),
                       const double AMG_aggregation_threshold = 0.02)
          : solver_type(solver_type)
          , tolerance(tolerance)
          , maxiters(maxiters)
          , use_scaling(use_scaling)
          , scale_with_inv_row_sum(scale_with_inv_row_sum)
          , constrained_dofs(constrained_dofs)
          , AMG_aggregation_threshold(AMG_aggregation_threshold)
        {}

        PoissonTypeSolver
          solver_type; /** Solver type for the Poisson-type problems. */
        double
                     tolerance; /** relative tolerance for the CG solver for (B B^T)^-1 */
        unsigned int maxiters; /** maximal number of CG iterations to
                                  approximate (B B^T)^-1 */
        bool use_scaling;      /** use a scaled poisson-type problem B M B^T */
        bool scale_with_inv_row_sum; // else: scale with inv_diagonal
        mutable std::vector<unsigned int> constrained_dofs;
        double AMG_aggregation_threshold; /** aggregation threshold for the AMG
                                             for B B^T */
      };

      /**
       * @brief Default constructor. Needs to be initialized with @ref initialize() before usage.
       */
      TrilinosBFBt()
        : Subscriptor()
        , solver_control_lu()
        , LU_additional_data(true, "Amesos_Klu")
        , LU_BMBT(solver_control_lu, LU_additional_data)
      {}

      /**
       * @brief Constructs the preconditioner with the given block system matrix.
       *
       * Needs to be initialized with @ref initialize() before usage.
       *
       * @param system_matrix Saddle-point system matrix of the fluid-flow problem.
       * @param data Struct containing the parameters for the preconditioner.
       */
      TrilinosBFBt(const BlockMatrixType &system_matrix,
                   const AdditionalData & data = AdditionalData())
        : Subscriptor()
        , block00(system_matrix.block(0, 0))
        , block01(system_matrix.block(0, 1))
        , BBT(system_matrix.block(0, 1))
        , solver_control_lu()
        , LU_additional_data(true, "Amesos_Klu")
        , LU_BMBT(solver_control_lu, LU_additional_data)
        , data(data)
      {
        tmp.reinit(complete_index_set(system_matrix.block(0, 1).n()));
        tmp2.reinit(complete_index_set(system_matrix.block(0, 1).m()));
      }



      /**
       * @brief Constructor where the matrix blocks A, B^T are given in @ref block00, @ref block01.
       *
       * @param block00 (0,0)-block of the saddle-point matrix.
       * @param block01 (0,1)-block of the saddle-point matrix.
       * @param data Struct containing the parameters for the preconditioner.
       */
      TrilinosBFBt(const MatrixType &    block00,
                   const MatrixType &    block01,
                   const AdditionalData &data = AdditionalData())
        : Subscriptor()
        , block00(block00)
        , block01(block01)
        , BBT(block01)
        , solver_control_lu()
        , LU_additional_data(true, "Amesos_Klu")
        , LU_BMBT(solver_control_lu, LU_additional_data)
        , data(data)
      {
        tmp.reinit(complete_index_set(block01.n()));
        tmp2.reinit(complete_index_set(block01.m()));
      }

      /**
       * @brief Destructor
       */
      ~TrilinosBFBt()
      {}


      /**
       * @brief Inner initialization of the BFBt preconditioner
       * @param[in] matrix matrix that is used
       * for the scaling
       */
      void
      initialize_inner_data(const TrilinosWrappers::SparseMatrix &matrix)
      {
        /////////////////////////////////////
        // initialize the scaling M (if we use (B M B^T)^-1 instead of (B
        // B^T)^-1)
        /////////////////////////////////////

        // initialize scaling_vector
        scaling_vector.reinit(
          complete_index_set(block01.get_matrix_ptr()->m()));

        // The functions for extracting the diagonal or inverse row sums are
        // only defined for Epetra_Vectors
        Epetra_Vector tmp_scaling(scaling_vector.trilinos_partitioner());
        if (data.scale_with_inv_row_sum)
          {
            // Adapted from the method InvRowSums for Epetra_CrsMatrix
            double *vector_entries;
            tmp_scaling.ExtractView(&vector_entries);
            int n_rows = matrix.trilinos_matrix().NumMyRows();
            for (int i = 0; i < n_rows; i++)
              {
                int     n_entries = matrix.trilinos_matrix().NumMyEntries(i);
                double *row_entries;
                matrix.trilinos_matrix().ExtractMyRowView(i,
                                                          n_entries,
                                                          row_entries);
                double scale = 0.0;
                for (int j = 0; j < n_entries; j++)
                  {
                    scale += std::abs(row_entries[j]);
                  }
                if (scale < Epetra_MinDouble)
                  {
                    // ignore zero rows
                    vector_entries[i] = 0.0;
                  }
                else
                  vector_entries[i] = 1.0 / scale;
              }
          }
        else
          {
            matrix.trilinos_matrix().ExtractDiagonalCopy(tmp_scaling);
            // invert diagonal entries
            unsigned int N = matrix.m();
            double       diag;
            for (unsigned int i = 0; i < N; ++i)
              {
                diag = tmp_scaling[i];
                if (diag > Epetra_MinDouble)
                  tmp_scaling[i] = 1. / diag;
                else
                  tmp_scaling[i] = 0.0;
              }
          }
        // Fill an Epetra_FEVector from the Epetra_Vector such that we can
        // use the TrilinosWrappers::MPI::Vector class for the scaling
        // operations
        double *array_view = scaling_vector.trilinos_vector()[0];
        tmp_scaling.ExtractCopy(array_view, 1);

        /////////////////////////////////////
        // initialize TrilinosATA objects with the new scaling vector
        /////////////////////////////////////
        BBT.initialize(scaling_vector,
                       0, /*already shifted */
                       data.constrained_dofs);

        /////////////////////////////////////
        // Compute the modified matrix-matrix product B B^T (s.th. the result is
        // invertible) if we need it
        /////////////////////////////////////
        switch (data.solver_type)
          {
            case Direct:
            case IC:
            case AMG:
              case PCG_AMG: {
                // Compute matrix-matrix product
                std::vector<int> indices(std::begin(data.constrained_dofs),
                                         std::end(data.constrained_dofs));
                perform_modified_mmult(*block01.get_matrix_ptr(),
                                       *block01.get_matrix_ptr(),
                                       BMBT,
                                       scaling_vector,
                                       true,
                                       indices);
                wrap_BMBT.reinit(BMBT);

                break;
              }
            case CG:
              break;
          }

        /////////////////////////////////////
        // initialize solvers for the Poisson-type problems
        /////////////////////////////////////
        switch (data.solver_type)
          {
              case Direct: {
                // Factorize B M B^T
                LU_BMBT.initialize(BMBT);

                break;
              }
              case IC: {
                // Compute IC(0) of BMBT
                ic_preconditioner.initialize(BMBT);
                wrap_ic_preconditioner.reinit(ic_preconditioner);

                break;
              }
            case AMG:
              case PCG_AMG: {
                // Initialize AMG for CG

                TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;
                amg_data.elliptic              = true;
                amg_data.higher_order_elements = true;
                amg_data.smoother_sweeps       = 2;
                amg_data.aggregation_threshold =
                  data.AMG_aggregation_threshold; // default: 0.02
                amg_data.output_details = false;  // true;
                if (data.solver_type == AMG)
                  amg_data.n_cycles = 2;
                else
                  amg_data.n_cycles = 1;
                Amg_preconditioner.initialize(BMBT, amg_data);
                wrap_amg_preconditioner.reinit(Amg_preconditioner);

                break;
              }
            case CG:
              break;
          }
      } // initialize



      /**
       * @brief Inner initialization of the BFBt preconditioner
       */
      void
      initialize_inner_data()
      {
        /////////////////////////////////////
        // initialize the scaling vector as a one vector (since we do not scale)
        /////////////////////////////////////
        scaling_vector.reinit(
          complete_index_set(block01.get_matrix_ptr()->m()));
        scaling_vector = 1.0; // no scaling

        /////////////////////////////////////
        // Initialize the TrilinosATA objects for B M B^T
        /////////////////////////////////////
        BBT.initialize(scaling_vector,
                       0, // /*already shifted*/scaling_vector.size() /*shift*/,
                       data.constrained_dofs);


        /////////////////////////////////////
        // Compute matrix-matrix product B M B^T if we need it
        /////////////////////////////////////
        switch (data.solver_type)
          {
            case Direct:
            case IC:
            case AMG:
              case PCG_AMG: {
                std::vector<int> indices(std::begin(data.constrained_dofs),
                                         std::end(data.constrained_dofs));
                perform_modified_mmult(*block01.get_matrix_ptr(),
                                       *block01.get_matrix_ptr(),
                                       BMBT,
                                       scaling_vector,
                                       true,
                                       indices);
                wrap_BMBT.reinit(BMBT);

                break;
              }
            case CG:
              break;
          }

        switch (data.solver_type)
          {
              case Direct: {
                // Factorize B M B^T
                std::cout << "factorize BMBT " << std::endl;
                LU_BMBT.initialize(BMBT);

                break;
              }
              case IC: {
                ic_preconditioner.initialize(BMBT);
                wrap_ic_preconditioner.reinit(ic_preconditioner);

                break;
              }
            case AMG:
              case PCG_AMG: {
                // Initialize AMG as preconditioner for the CG solver
                TrilinosWrappers::PreconditionAMG::AdditionalData amg_data;

                amg_data.elliptic              = true;
                amg_data.higher_order_elements = true;
                amg_data.smoother_sweeps       = 2;
                amg_data.aggregation_threshold =
                  data.AMG_aggregation_threshold; // default: 0.02
                amg_data.output_details = false;  // true;
                if (data.solver_type == AMG)
                  amg_data.n_cycles = 2;
                else
                  amg_data.n_cycles = 1;
                Amg_preconditioner.initialize(BMBT, amg_data);
                wrap_amg_preconditioner.reinit(Amg_preconditioner);

                break;
              }
            case CG:
              break;
          }
      } // initialize



      /**
       * @brief Initializes the BFBt preconditioner with scaling
       * @param system_matrix Block matrix of the (Navier-)Stokes problem
       * @param u_preconditioner_matrix matrix that is used to initialize the scaling (e.g. velocity mass matrix)
       * @param input_data sets options of the BFBt preconditioner
       */
      void
      initialize(const TrilinosWrappers::BlockSparseMatrix &system_matrix,
                 const TrilinosWrappers::SparseMatrix &u_preconditioner_matrix,
                 const AdditionalData &                input_data)
      {
        data = input_data;

        // set the pointers to the needed matrices
        block00.reinit(system_matrix.block(0, 0));
        block01.reinit(system_matrix.block(0, 1));


        BBT.initialize(system_matrix.block(0, 1),
                       0, // scaling_vector.size() /*shift*/,
                       data.constrained_dofs);

        // initialize the needed preconditioners
        initialize_inner_data(u_preconditioner_matrix);

        // initialize temporary vectors
        tmp.reinit(system_matrix.block(0, 1).locally_owned_domain_indices());
        tmp2.reinit(system_matrix.block(0, 1).locally_owned_range_indices());
      }

      /**
       * @brief Initialize BFBt preconditioner without scaling
       * @param system_matrix block matrix that from the (Navier-)Stokes problem
       * @param input_data options for the BFBt preconditioner
       */
      void
      initialize(const TrilinosWrappers::BlockSparseMatrix &system_matrix,
                 const AdditionalData &                     input_data)
      {
        data = input_data;

        // set the pointers to the needed matrices
        block00.reinit(system_matrix.block(0, 0));
        block01.reinit(system_matrix.block(0, 1));

        // initialize the TrilinosATA objects that describe B M B^T
        BBT.initialize(system_matrix.block(0, 1),
                       0, // scaling_vector.size() /*shift*/,
                       data.constrained_dofs);

        // initialize the needed inner preconditioners
        initialize_inner_data();

        // initialize temporary vectors
        tmp.reinit(system_matrix.block(0, 1).locally_owned_domain_indices());
        tmp2.reinit(system_matrix.block(0, 1).locally_owned_range_indices());
      }

      /**
       * @brief Solve the Poisson-type problems.
       *
       * @param trg Result.
       * @param src Input vector.
       */
      void
      solve_poissontype_problem(MultiVector2 &      trg,
                                const MultiVector2 &src) const
      {
        switch (data.solver_type)
          {
              case Direct: {
                Assert(false, ExcNotImplemented());
                break;
              }
              case IC: {
                wrap_ic_preconditioner.vmult(trg, src);
                break;
              }
              case AMG: {
                wrap_amg_preconditioner.vmult(trg, src);
                break;
              }
            case CG:
              for (unsigned int i = 0; i < src.n_vectors(); ++i)
                {
                  Preconditioners::MultiVector2::Column tmp_src(
                    src.trilinos_vector(), i, View);
                  Preconditioners::MultiVector2::Column tmp_trg(
                    trg.trilinos_vector(), i, View);
                  // tmp_trg = 0;
                  SolverControl solver_control(data.maxiters,
                                               data.tolerance *
                                                 tmp_src.l2_norm());
                  SolverCG<Preconditioners::MultiVector2::Column> solver(
                    solver_control);
                  try
                    {
                      solver.solve(BBT,
                                   tmp_trg,
                                   tmp_src,
                                   PreconditionIdentity());
                    }
                  catch (SolverControl::NoConvergence &)
                    {}
                  // tmp_trg.set_column(trg, i);
                }
              break;
              case PCG_AMG: {
                for (unsigned int i = 0; i < src.n_vectors(); ++i)
                  {
                    Preconditioners::MultiVector2::Column tmp_src(
                      src.trilinos_vector(), i, View);
                    Preconditioners::MultiVector2::Column tmp_trg(
                      trg.trilinos_vector(), i, View);
                    // tmp_trg = 0;
                    SolverControl solver_control(data.maxiters,
                                                 data.tolerance *
                                                   tmp_src.l2_norm());
                    SolverCG<Preconditioners::MultiVector2::Column> solver(
                      solver_control);
                    try
                      {
                        solver.solve(wrap_BMBT,
                                     tmp_trg,
                                     tmp_src,
                                     wrap_amg_preconditioner);
                      }
                    catch (SolverControl::NoConvergence &)
                      {}
                    // tmp_trg.set_column(trg, i);
                  }
                break;
              }
          }
      }



      /**
       * @brief Solve the Poisson-type problems.
       *
       * @param trg Result.
       * @param src Input vector.
       */
      void
      solve_poissontype_problem(MultiVector2::Column &      trg,
                                const MultiVector2::Column &src) const
      {
        switch (data.solver_type)
          {
              case Direct: {
                Assert(false, ExcNotImplemented());
                break;
              }
              case IC: {
                wrap_ic_preconditioner.vmult(trg, src);
                break;
              }
              case AMG: {
                wrap_amg_preconditioner.vmult(trg, src);
                break;
              }
              case CG: {
                SolverControl solver_control(data.maxiters,
                                             data.tolerance * src.l2_norm());
                SolverCG<Preconditioners::MultiVector2::Column> solver(
                  solver_control);
                try
                  {
                    solver.solve(BBT, trg, src, PreconditionIdentity());
                  }
                catch (SolverControl::NoConvergence &)
                  {}

                break;
              }
              case PCG_AMG: {
                SolverControl solver_control(data.maxiters,
                                             data.tolerance * src.l2_norm());
                SolverCG<Preconditioners::MultiVector2::Column> solver(
                  solver_control);
                try
                  {
                    solver.solve(wrap_BMBT, trg, src, wrap_amg_preconditioner);
                  }
                catch (SolverControl::NoConvergence &)
                  {}

                break;
              }
          }
      }

      /**
       * @brief Solve the Poisson-type problems.
       *
       * @param trg Result.
       * @param src Input vector.
       */
      void
      solve_poissontype_problem(TrilinosWrappers::MPI::Vector &      trg,
                                const TrilinosWrappers::MPI::Vector &src) const
      {
        switch (data.solver_type)
          {
              case Direct: {
                Assert(false, ExcNotImplemented());
                break;
              }
              case IC: {
                wrap_ic_preconditioner.vmult(trg, src);
                break;
              }
              case AMG: {
                wrap_amg_preconditioner.vmult(trg, src);
                break;
              }
              case CG: {
                SolverControl solver_control(data.maxiters,
                                             data.tolerance * src.l2_norm());
                SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
                try
                  {
                    solver.solve(BBT, trg, src, PreconditionIdentity());
                  }
                catch (SolverControl::NoConvergence &)
                  {}
                break;
              }
              case PCG_AMG: {
                SolverControl solver_control(data.maxiters,
                                             data.tolerance * src.l2_norm());
                SolverCG<TrilinosWrappers::MPI::Vector> solver(solver_control);
                try
                  {
                    solver.solve(wrap_BMBT, trg, src, wrap_amg_preconditioner);
                  }
                catch (SolverControl::NoConvergence &)
                  {}
                break;
              }
          }
      }


      /**
       * @brief Computes the matrix-vector product with the BFBt/LSC preconditioner.
       *
       * trg = this * src
       *
       * @param[out] trg result
       * @param[in] src input vector
       */
      void
      vmult(Preconditioners::MultiVector2 &      trg,
            const Preconditioners::MultiVector2 &src) const
      {
        Preconditioners::MultiVector2 mv_tmp2;
        mv_tmp2.reinit(src.n_vectors(), block00.get_matrix_ptr()->m());
        Preconditioners::MultiVector2 mv_tmp3(mv_tmp2);
        Preconditioners::MultiVector2 mv_tmp(src);



        // Apply (B M B^T)^-1
        solve_poissontype_problem(mv_tmp, src);


        // Compute B M A M B^T (B M B^T)^-1
        block01.vmult(mv_tmp2, mv_tmp);
        mv_tmp2.scale(scaling_vector);
        block00.vmult(mv_tmp3, mv_tmp2);
        mv_tmp3.scale(scaling_vector);
        block01.Tvmult(mv_tmp, mv_tmp3);

        // Apply (B M BT)^-1;
        solve_poissontype_problem(trg, mv_tmp);
      } // vmult

      /**
       * @brief Computes the matrix-vector product with the BFBt/LSC preconditioner.
       *
       * trg = this * src
       *
       * @param[out] trg result
       * @param[in] src input vector
       */
      void
      vmult(Preconditioners::MultiVector2::Column &      trg,
            const Preconditioners::MultiVector2::Column &src) const
      {
        Preconditioners::MultiVector2::Column mv_tmp2(
          scaling_vector.trilinos_partitioner());
        Preconditioners::MultiVector2::Column mv_tmp3(mv_tmp2);
        Preconditioners::MultiVector2::Column mv_tmp(src);



        // Apply (B M B^T)^-1
        solve_poissontype_problem(mv_tmp, src);


        // Compute B M A M B^T (B M B^T)^-1
        block01.vmult(mv_tmp2, mv_tmp);
        mv_tmp2.scale(scaling_vector);
        block00.vmult(mv_tmp3, mv_tmp2);
        mv_tmp3.scale(scaling_vector);
        block01.Tvmult(mv_tmp, mv_tmp3);

        // Apply (B M BT)^-1;
        solve_poissontype_problem(trg, mv_tmp);

      } // vmult


      /**
       * @brief Computes the matrix-vector product with the BFBt/LSC preconditioner.
       *
       * trg = this * src
       *
       * @param[out] trg result
       * @param[in] src input vector
       */
      void
      vmult(TrilinosWrappers::MPI::Vector &      trg,
            const TrilinosWrappers::MPI::Vector &src) const
      {
        TrilinosWrappers::MPI::Vector tmp3(tmp2);


        // Apply (B M B^T)^-1
        solve_poissontype_problem(tmp, src);


        // Compute B M A M B^T (B M B^T)^-1
        block01.vmult(tmp2, tmp);
        tmp2.scale(scaling_vector);
        block00.vmult(tmp3, tmp2);
        tmp3.scale(scaling_vector);
        block01.Tvmult(tmp, tmp3);

        // Apply (B M BT)^-1
        solve_poissontype_problem(trg, tmp);

      } // vmult

      /**
       * @brief Computes the transposed matrix-vector product with the BFBt/LSC preconditioner.
       *
       * trg = this^T * src
       *
       * @param[out] trg result
       * @param[in] src input vector
       */
      void
      Tvmult(Preconditioners::MultiVector2 &      trg,
             const Preconditioners::MultiVector2 &src) const
      {
        Preconditioners::MultiVector2 mv_tmp2;
        mv_tmp2.reinit(src.n_vectors(), block00.get_matrix_ptr()->m());
        Preconditioners::MultiVector2 mv_tmp3(mv_tmp2);
        Preconditioners::MultiVector2 mv_tmp(src);

        // Apply (B M B^T)^-1
        solve_poissontype_problem(mv_tmp, src);

        // Compute B M A^T M B^T (B M B^T)^-1
        block01.vmult(mv_tmp2, mv_tmp);
        mv_tmp2.scale(scaling_vector);
        block00.Tvmult(mv_tmp3, mv_tmp2);
        mv_tmp3.scale(scaling_vector);
        block01.Tvmult(mv_tmp, mv_tmp3);

        // Apply (B M BT)^-1
        solve_poissontype_problem(trg, mv_tmp);
      } // Tvmult

      /**
       * @brief Computes the transposed matrix-vector product with the BFBt/LSC preconditioner.
       *
       * trg = this^T * src
       *
       * @param[out] trg result
       * @param[in] src input vector
       */
      void
      Tvmult(Preconditioners::MultiVector2::Column &      trg,
             const Preconditioners::MultiVector2::Column &src) const
      {
        Preconditioners::MultiVector2::Column mv_tmp2(
          scaling_vector.trilinos_partitioner());
        Preconditioners::MultiVector2::Column mv_tmp3(mv_tmp2);
        Preconditioners::MultiVector2::Column mv_tmp(src);

        // Apply (B M B^T)^-1
        solve_poissontype_problem(mv_tmp, src);


        // Compute B M A^T M B^T (B M B^T)^-1
        block01.vmult(mv_tmp2, mv_tmp);
        mv_tmp2.scale(scaling_vector);
        block00.Tvmult(mv_tmp3, mv_tmp2);
        mv_tmp3.scale(scaling_vector);
        block01.Tvmult(mv_tmp, mv_tmp3);

        // Apply (B M BT)^-1
        solve_poissontype_problem(trg, mv_tmp);

      } // Tvmult


      /**
       * @brief Computes theh transposed matrix-vector product with the BFBt/LSC preconditioner.
       *
       * trg = this^T * src
       *
       * @param[out] trg result
       * @param[in] src input vector
       */
      void
      Tvmult(TrilinosWrappers::MPI::Vector &      trg,
             const TrilinosWrappers::MPI::Vector &src) const
      {
        TrilinosWrappers::MPI::Vector tmp3(tmp2);
        tmp.reinit(src);

        // Apply (B M B^T)^-1
        solve_poissontype_problem(tmp, src);


        // Compute B M A^T M B^T (B M B^T)^-1
        block01.vmult(tmp2, tmp);
        tmp2.scale(scaling_vector);
        block00.Tvmult(tmp3, tmp2);
        tmp3.scale(scaling_vector);
        block01.Tvmult(tmp, tmp3);

        // Apply (B M BT)^-1
        solve_poissontype_problem(trg, tmp);

      } // Tvmult

      /**
       * reinit the upper left block of the saddle-point matrix
       */
      void
      reinit_A(const TrilinosWrappers::SparseMatrix &matrix)
      {
        block00.reinit(matrix);
      }



    private:
      WrapperMatrix block00; /** points to the upper left block A of the
                                saddle-point matrix */
      WrapperMatrix block01; /** points to the upper right block B^T of
                                the saddle-point matrix */
      TrilinosATA BBT; /** describes B M B^T for a diagonal scaling matrix M */
      TrilinosWrappers::MPI::Vector
        scaling_vector; /** contains the diagonal of M */

      TrilinosWrappers::PreconditionAMG Amg_preconditioner;
      WrapperPreconditioner             wrap_amg_preconditioner;

      TrilinosWrappers::PreconditionIC ic_preconditioner;
      WrapperPreconditioner            wrap_ic_preconditioner;

      MatrixType    BMBT;      // also needed for AMG
      WrapperMatrix wrap_BMBT; // also needed for AMG

      // only needed for direct solver
      SolverControl                                  solver_control_lu;
      TrilinosWrappers::SolverDirect::AdditionalData LU_additional_data;
      mutable TrilinosWrappers::SolverDirect         LU_BMBT;

      AdditionalData data; /** preconditioner options */

      // temporary vectors for computations
      mutable TrilinosWrappers::MPI::Vector tmp;
      mutable TrilinosWrappers::MPI::Vector tmp2;
    }; // end class TrilinosBFBT

  } // namespace Preconditioners

} // namespace dealii
#endif
