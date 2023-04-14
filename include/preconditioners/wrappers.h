/**
 * @file wrappers.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef PRECONDITIONER_WRAPPERS
#define PRECONDITIONER_WRAPPERS

#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

namespace dealii
{
  namespace Preconditioners
  {
    /**
     * @brief Wrapper for TrilinosWrappers::Sparsematrix. Used to compute
     * matrix-vector products with own VectorTypes.
     */
    class WrapperMatrix : public Subscriptor
    {
    public:
      using value_type = double;
      using size_type  = types::global_dof_index;

      /**
       * @brief Default constructor.
       */
      WrapperMatrix()
        : Subscriptor()
      {}

      /**
       * @brief Constructor.
       *
       * @param matrix Matrix.
       */
      WrapperMatrix(const TrilinosWrappers::SparseMatrix &matrix)
        : Subscriptor()
        , matrix(&matrix)
      {}


      /** @brief Destructor. */
      ~WrapperMatrix()
      {}

      /**
       * @brief Reinit with another matrix.
       *
       * @param in_matrix Set the SmartPointer to another matrix.
       */
      void
      reinit(const TrilinosWrappers::SparseMatrix &in_matrix)
      {
        matrix = &in_matrix;
      }

      /**
       * @brief Returns a pointer to the underlying matrix.
       *
       */
      const SmartPointer<const TrilinosWrappers::SparseMatrix>
      get_matrix_ptr() const
      {
        return matrix;
      }

      /**
       * @brief Returns a reference to the underlying Epetra_CrsMatrix.
       */
      const Epetra_CrsMatrix &
      trilinos_matrix() const
      {
        return matrix->trilinos_matrix();
      }

      /**
       * @brief Computes trg += (*this) * src.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Input vector.
       */
      template <typename VectorType>
      void
      vmult_add(VectorType &trg, const VectorType &src) const;

      /**
       * @brief Computes trg += (*this)^T * src.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Input vector.
       */
      template <typename VectorType>
      void
      Tvmult_add(VectorType &trg, const VectorType &src) const;

      /**
       * @brief Matrix-vector product trg = (*this) * src
       *
       * @param trg Result.
       * @param src Input vector.
       */
      void
      vmult(Preconditioners::MultiVector2 &      trg,
            const Preconditioners::MultiVector2 &src) const;

      /**
       * @brief Transposed matrix-vector product
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Input vector.
       */
      void
      Tvmult(Preconditioners::MultiVector2 &      trg,
             const Preconditioners::MultiVector2 &src) const;

      /**
       * @brief Matrix-vector product trg = (*this) * src
       *
       * @param trg Result.
       * @param src Input vector.
       */
      void
      vmult(Preconditioners::MultiVector2::Column &      trg,
            const Preconditioners::MultiVector2::Column &src) const;

      /**
       * @brief Transposed matrix-vector product
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Input vector.
       */
      void
      Tvmult(Preconditioners::MultiVector2::Column &      trg,
             const Preconditioners::MultiVector2::Column &src) const;

      /**
       * @brief Matrix-vector product trg = (*this) * src
       *
       * @param trg Result.
       * @param src Input vector.
       */
      void
      vmult(Epetra_Vector &trg, const Epetra_Vector &src) const;

      /**
       * @brief Transposed matrix-vector product
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Input vector.
       */
      void
      Tvmult(Epetra_Vector &trg, const Epetra_Vector &src) const;

      /**
       * @brief Matrix-vector product trg = (*this) * src
       *
       * @param trg Result.
       * @param src Input vector.
       */
      void
      vmult(TrilinosWrappers::MPI::Vector &      trg,
            const TrilinosWrappers::MPI::Vector &src) const;

      /**
       * @brief Transposed matrix-vector product
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Input vector.
       */
      void
      Tvmult(TrilinosWrappers::MPI::Vector &      trg,
             const TrilinosWrappers::MPI::Vector &src) const;

      /**
       * @brief Returns the number of rows of the underlying matrix.
       *
       * @return size_type Number of matrix rows.
       */
      size_type
      m() const
      {
        return matrix->m();
      }

      /**
       * @brief Returns the number of columnss of the underlying matrix.
       *
       * @return size_type Number of matrix columns.
       */
      size_type
      n() const
      {
        return matrix->n();
      }

    private:
      SmartPointer<const TrilinosWrappers::SparseMatrix> matrix;
    };



    /**
     * @brief Wrapper for TrilinosWrappers::BlockSparseMatrix. Used to compute
     * matrix-vector products with own VectorTypes.
     */
    class BlockWrapperMatrix : public BlockMatrixBase<WrapperMatrix>
    {
    public:
      using BaseClass = BlockMatrixBase<WrapperMatrix>;

      using BlockType = BaseClass::BlockType;

      using value_type      = BaseClass::value_type;
      using pointer         = BaseClass::pointer;
      using const_pointer   = BaseClass::const_pointer;
      using reference       = BaseClass::reference;
      using const_reference = BaseClass::const_reference;
      using size_type       = BaseClass::size_type;
      using iterator        = BaseClass::iterator;
      using const_iterator  = BaseClass::const_iterator;

      BlockWrapperMatrix() = default;


      BlockWrapperMatrix(const TrilinosWrappers::BlockSparseMatrix &matrix)
        : BlockWrapperMatrix()
      {
        unsigned int n_block_rows    = matrix.n_block_rows();
        unsigned int n_block_columns = matrix.n_block_cols();

        // resize. set sizes of blocks to
        // zero. user will later have to call
        // collect_sizes for this
        this->sub_objects.reinit(n_block_rows, n_block_columns);
        this->row_block_indices.reinit(n_block_rows, 0);
        this->column_block_indices.reinit(n_block_columns, 0);

        // and reinitialize the blocks
        for (size_type r = 0; r < this->n_block_rows(); ++r)
          for (size_type c = 0; c < this->n_block_cols(); ++c)
            {
              BlockType *p = new BlockType();

              Assert(this->sub_objects[r][c] == nullptr, ExcInternalError());
              this->sub_objects[r][c] = p;
              this->sub_objects[r][c]->reinit(matrix.block(r, c));
            }
      }

      ~BlockWrapperMatrix()
      { // delete previous content of
        // the subobjects array
        try
          {
            clear();
          }
        catch (...)
          {}
      }

      using BlockMatrixBase<WrapperMatrix>::clear;

      BlockWrapperMatrix &
      operator=(const BlockWrapperMatrix &) = default;

      void
      reinit(const TrilinosWrappers::BlockSparseMatrix &matrix)
      {
        // resize. set sizes of blocks to
        // zero. user will later have to call
        // collect_sizes for this
        reinit(matrix.n_block_rows(), matrix.n_block_cols());

        // and reinitialize the blocks
        for (size_type r = 0; r < this->n_block_rows(); ++r)
          for (size_type c = 0; c < this->n_block_cols(); ++c)
            {
              this->sub_objects[r][c]->reinit(matrix.block(r, c));
            }
      }

      void
      reinit(const unsigned int n_block_rows, const unsigned int n_block_cols)
      {
        // first delete previous content of
        // the subobjects array
        clear();

        // then resize. set sizes of blocks to
        // zero. user will later have to call
        // collect_sizes for this
        this->sub_objects.reinit(n_block_rows, n_block_cols);
        this->row_block_indices.reinit(n_block_rows, 0);
        this->column_block_indices.reinit(n_block_cols, 0);

        // and reinitialize the blocks
        for (size_type r = 0; r < this->n_block_rows(); ++r)
          for (size_type c = 0; c < this->n_block_cols(); ++c)
            {
              BlockType *p = new BlockType();

              Assert(this->sub_objects[r][c] == nullptr, ExcInternalError());
              this->sub_objects[r][c] = p;
            }
      }



      /**
       * @brief Matrix vector product with the underlying block matrix.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Source vector.
       */
      template <typename VectorType>
      void
      vmult(VectorType &trg, const VectorType &src) const
      {
        Assert(trg.n_blocks() == n_block_rows(),
               ExcDimensionMismatch(trg.n_blocks(), n_block_rows()));
        Assert(src.n_blocks() == n_block_cols(),
               ExcDimensionMismatch(src.n_blocks(), n_block_cols()));

        for (size_type row = 0; row < n_block_rows(); ++row)
          {
            block(row, 0).vmult(trg.block(row), src.block(0));
            for (size_type col = 1; col < n_block_cols(); ++col)
              block(row, col).vmult_add(trg.block(row), src.block(col));
          };
      }

      /**
       * @brief Transposed matrix vector product with the underlying block matrix.
       *
       * @tparam VectorType
       * @param trg Result.
       * @param src Source vector.
       */
      template <typename VectorType>
      void
      Tvmult(VectorType &trg, const VectorType &src) const
      {
        Assert(trg.n_blocks() == n_block_cols(),
               ExcDimensionMismatch(trg.n_blocks(), n_block_cols()));
        Assert(src.n_blocks() == n_block_rows(),
               ExcDimensionMismatch(src.n_blocks(), n_block_rows()));

        trg = 0.;

        for (unsigned int row = 0; row < n_block_rows(); ++row)
          {
            for (unsigned int col = 0; col < n_block_cols(); ++col)
              block(row, col).Tvmult_add(trg.block(col), src.block(row));
          }
      }
    };



    /**
     * @brief Wrapper for preconditioners based on
     * TrilinosWrappers::PreconditionBase. Used to compute matrix-vector
     * products with own VectorTypes.
     */
    class WrapperPreconditioner : public Subscriptor
    {
    public:
      /** @brief Default constructor. */
      WrapperPreconditioner()
        : Subscriptor()
      {}

      /** @brief Constructor.
       *
       * @param preconditioner Preconditioner derived from TrilinosWrappers::PreconditionBase.
       */
      WrapperPreconditioner(
        const TrilinosWrappers::PreconditionBase &preconditioner)
        : Subscriptor()
        , preconditioner(&preconditioner)
      {}



      /** @brief Destructor */
      ~WrapperPreconditioner()
      {}

      /** @brief Reinit the Wrapper with another preconditioner. */
      void
      reinit(const TrilinosWrappers::PreconditionBase &in_preconditioner)
      {
        preconditioner = &in_preconditioner;
      }

      /** @brief Applies the preconditioner to a vector.
       * trg = *this * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      vmult(Preconditioners::MultiVector2 &      trg,
            const Preconditioners::MultiVector2 &src) const;

      /** @brief Applies the transposed preconditioner (if defined) to a vector.
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      Tvmult(Preconditioners::MultiVector2 &      trg,
             const Preconditioners::MultiVector2 &src) const;

      /** @brief Applies the preconditioner to a vector.
       * trg = *this * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      vmult(Preconditioners::MultiVector2::Column &      trg,
            const Preconditioners::MultiVector2::Column &src) const;

      /** @brief Applies the transposed preconditioner (if defined) to a vector.
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      Tvmult(Preconditioners::MultiVector2::Column &      trg,
             const Preconditioners::MultiVector2::Column &src) const;

      /** @brief Applies the preconditioner to a vector.
       * trg = *this * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      vmult(Epetra_Vector &trg, const Epetra_Vector &src) const;

      /** @brief Applies the transposed preconditioner (if defined) to a vector.
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      Tvmult(Epetra_Vector &trg, const Epetra_Vector &src) const;

      /** @brief Applies the preconditioner to a vector.
       * trg = *this * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      vmult(TrilinosWrappers::MPI::Vector &      trg,
            const TrilinosWrappers::MPI::Vector &src) const;

      /** @brief Applies the transposed preconditioner (if defined) to a vector.
       * trg = (*this)^T * src
       * @param trg Result.
       * @param src Source vector.
       */
      void
      Tvmult(TrilinosWrappers::MPI::Vector &      trg,
             const TrilinosWrappers::MPI::Vector &src) const;


    private:
      SmartPointer<const TrilinosWrappers::PreconditionBase> preconditioner;
    };


    //////////////////////////////////////////////////////
    // Implementation
    //////////////////////////////////////////////////////
    template <typename VectorType>
    inline void
    WrapperMatrix::vmult_add(VectorType &trg, const VectorType &src) const
    {
      // Assert(&src != &trg, ExcSourceEqualsDestination());

      // Reinit a temporary vector with fast argument set, which does not
      // overwrite the content (to save time).
      VectorType tmp_vector(trg);
      vmult(tmp_vector, src);
      trg += tmp_vector;
    }

    template <typename VectorType>
    inline void
    WrapperMatrix::Tvmult_add(VectorType &trg, const VectorType &src) const
    {
      // Assert(&src != &trg, ExcSourceEqualsDestination());

      // Reinit a temporary vector with fast argument set, which does not
      // overwrite the content (to save time).
      VectorType tmp_vector(trg);
      Tvmult(tmp_vector, src);
      trg += tmp_vector;
    }


    inline void
    WrapperMatrix::vmult(Preconditioners::MultiVector2 &      trg,
                         const Preconditioners::MultiVector2 &src) const
    {
      // Assert(&trg != &src, ExcSourceEqualsDestination());
      // dimensions are not checked

      const int ierr =
        matrix->trilinos_matrix().Multiply(false,
                                           src.trilinos_vector(),
                                           trg.trilinos_vector());
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }

    inline void
    WrapperMatrix::Tvmult(Preconditioners::MultiVector2 &      trg,
                          const Preconditioners::MultiVector2 &src) const
    {
      // Assert(&trg != &src, ExcSourceEqualsDestination());
      // dimensions are not checked

      const int ierr =
        matrix->trilinos_matrix().Multiply(true,
                                           src.trilinos_vector(),
                                           trg.trilinos_vector());
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }

    inline void
    WrapperMatrix::vmult(Preconditioners::MultiVector2::Column &      trg,
                         const Preconditioners::MultiVector2::Column &src) const
    {
      Assert(&trg != &src,
             TrilinosWrappers::SparseMatrix::ExcSourceEqualsDestination());
      Assert(matrix->trilinos_matrix().Filled(),
             TrilinosWrappers::SparseMatrix::ExcMatrixNotCompressed());
      (void)src;
      (void)trg;


      Assert(src.trilinos_vector().Map().SameAs(
               matrix->trilinos_matrix().DomainMap()) == true,
             ExcMessage("Column map of matrix does not fit with vector map!"));
      Assert(trg.trilinos_vector().Map().SameAs(
               matrix->trilinos_matrix().RangeMap()) == true,
             ExcMessage("Row map of matrix does not fit with vector map!"));

      const int ierr =
        matrix->trilinos_matrix().Multiply(false,
                                           src.trilinos_vector(),
                                           trg.trilinos_vector());
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }

    inline void
    WrapperMatrix::Tvmult(
      Preconditioners::MultiVector2::Column &      trg,
      const Preconditioners::MultiVector2::Column &src) const
    {
      Assert(&trg != &src,
             TrilinosWrappers::SparseMatrix::ExcSourceEqualsDestination());
      Assert(matrix->trilinos_matrix().Filled(),
             TrilinosWrappers::SparseMatrix::ExcMatrixNotCompressed());
      // dimensions are not checked
      Assert(trg.trilinos_vector().Map().SameAs(
               matrix->trilinos_matrix().DomainMap()) == true,
             ExcMessage("Column map of matrix does not fit with vector map!"));
      Assert(src.trilinos_vector().Map().SameAs(
               matrix->trilinos_matrix().RangeMap()) == true,
             ExcMessage("Row map of matrix does not fit with vector map!"));

      const int ierr =
        matrix->trilinos_matrix().Multiply(true,
                                           src.trilinos_vector(),
                                           trg.trilinos_vector());
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }

    inline void
    WrapperMatrix::vmult(Epetra_Vector &trg, const Epetra_Vector &src) const
    {
      Assert(&trg != &src,
             TrilinosWrappers::SparseMatrix::ExcSourceEqualsDestination());
      Assert(matrix->trilinos_matrix().Filled(),
             TrilinosWrappers::SparseMatrix::ExcMatrixNotCompressed());
      // dimensions are not checked

      const int ierr = matrix->trilinos_matrix().Multiply(false, src, trg);
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }

    inline void
    WrapperMatrix::Tvmult(Epetra_Vector &trg, const Epetra_Vector &src) const
    {
      // Assert(&trg != &src, ExcSourceEqualsDestination());
      // dimensions are not checked

      const int ierr = matrix->trilinos_matrix().Multiply(true, src, trg);
      Assert(ierr == 0, ExcTrilinosError(ierr));
      (void)ierr; // removes -Wunused-variable in optimized mode
    }

    inline void
    WrapperMatrix::vmult(TrilinosWrappers::MPI::Vector &      trg,
                         const TrilinosWrappers::MPI::Vector &src) const
    {
      matrix->vmult(trg, src);
    }

    inline void
    WrapperMatrix::Tvmult(TrilinosWrappers::MPI::Vector &      trg,
                          const TrilinosWrappers::MPI::Vector &src) const
    {
      matrix->Tvmult(trg, src);
    }



    inline void
    WrapperPreconditioner::vmult(Preconditioners::MultiVector2 &      trg,
                                 const Preconditioners::MultiVector2 &src) const
    {
      const int ierr =
        preconditioner->trilinos_operator().ApplyInverse(src.trilinos_vector(),
                                                         trg.trilinos_vector());
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));
    }

    inline void
    WrapperPreconditioner::Tvmult(
      Preconditioners::MultiVector2 &      trg,
      const Preconditioners::MultiVector2 &src) const
    {
      // preconditioner->transpose();
      int ierr = preconditioner->trilinos_operator().SetUseTranspose(true);
      (void)ierr;
      Assert(ierr == 0, ExcTrilinosError(ierr));
      ierr =
        preconditioner->trilinos_operator().ApplyInverse(src.trilinos_vector(),
                                                         trg.trilinos_vector());
      Assert(ierr == 0, ExcTrilinosError(ierr));
      // preconditioner->transpose();
      ierr = preconditioner->trilinos_operator().SetUseTranspose(false);
      Assert(ierr == 0, ExcTrilinosError(ierr));
    }

    inline void
    WrapperPreconditioner::vmult(
      Preconditioners::MultiVector2::Column &      trg,
      const Preconditioners::MultiVector2::Column &src) const
    {
      const int ierr =
        preconditioner->trilinos_operator().ApplyInverse(src.trilinos_vector(),
                                                         trg.trilinos_vector());
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));
    }

    inline void
    WrapperPreconditioner::Tvmult(
      Preconditioners::MultiVector2::Column &      trg,
      const Preconditioners::MultiVector2::Column &src) const
    {
      // preconditioner->transpose();
      int ierr = preconditioner->trilinos_operator().SetUseTranspose(true);
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      ierr =
        preconditioner->trilinos_operator().ApplyInverse(src.trilinos_vector(),
                                                         trg.trilinos_vector());
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      // preconditioner->transpose();
      ierr = preconditioner->trilinos_operator().SetUseTranspose(false);
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));
    }

    inline void
    WrapperPreconditioner::vmult(Epetra_Vector &      trg,
                                 const Epetra_Vector &src) const
    {
      const int ierr =
        preconditioner->trilinos_operator().ApplyInverse(src, trg);
      AssertThrow(ierr == 0, ExcTrilinosError(ierr));
    }

    inline void
    WrapperPreconditioner::Tvmult(Epetra_Vector &      trg,
                                  const Epetra_Vector &src) const
    {
      (void)trg;
      (void)src;
      Assert(false, ExcNotImplemented());
      // preconditioner->transpose();
      // const int ierr =
      //   preconditioner->trilinos_operator().ApplyInverse(src, trg);
      // AssertThrow(ierr == 0, ExcTrilinosError(ierr));
      // preconditioner->transpose();
    }

    inline void
    WrapperPreconditioner::vmult(TrilinosWrappers::MPI::Vector &      trg,
                                 const TrilinosWrappers::MPI::Vector &src) const
    {
      preconditioner->vmult(trg, src);
    }

    inline void
    WrapperPreconditioner::Tvmult(
      TrilinosWrappers::MPI::Vector &      trg,
      const TrilinosWrappers::MPI::Vector &src) const
    {
      preconditioner->Tvmult(trg, src);
    }

  } // namespace Preconditioners
} // namespace dealii
#endif