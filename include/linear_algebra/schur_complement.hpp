/**
 * @file schur_complement.hpp
 * @author Konrad Simon, Rebekka Beddig
 * @version 0.1
 */
#ifndef SCHURCOMPLEMENT
#define SCHURCOMPLEMENT

// Deal.ii
#include <deal.II/base/subscriptor.h>

// STL
#include <memory>
#include <vector>

// My headers
#include <base/config.h>

// preconditioner
#include <preconditioners/multivector2.h>
#include <preconditioners/wrappers.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace LinearAlgebra
{
  /*!
   * @class SchurComplement
   *
   * @brief Implements a MPI parallel Schur complement
   *
   * Implements a parallel Schur complement through the use of an inner inverse
   * matrix, i.e., if we want to solve
   * \f{eqnarray}{
   *	\left(
   *	\begin{array}{cc}
   *		A & B^T \\
   *		B & 0
   *	\end{array}
   *	\right)
   *	\left(
   *	\begin{array}{c}
   *		\sigma \\
   *		u
   *	\end{array}
   *	\right)
   *	=
   *	\left(
   *	\begin{array}{c}
   *		f \\
   *		0
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$A\f$ is invertible then we first define the inverse and
   *define the Schur complement as \f{eqnarray}{ \tilde S = BP_A^{-1}B^T \f}
   *to solve for \f$u\f$. The inverse must be separately given to the class as
   *an input argument.
   */
  template <typename BlockMatrixType, typename InverseMatrixType>
  class SchurComplement : public Subscriptor
  {
  private:
    using BlockType  = typename BlockMatrixType::BlockType;
    using VectorType = TrilinosWrappers::MPI::Vector;

  public:
    /*!
     * Constructor. The user must take care to pass the correct inverse of the
     * upper left block of the system matrix.
     *
     * @param system_matrix
     * 	Block Matrix
     * @param relevant_inverse_matrix
     * 	Inverse of upper left block of the system matrix.
     * @param owned_partitioning
     * @param mpi_communicator
     */
    SchurComplement(const BlockMatrixType &      system_matrix,
                    const InverseMatrixType &    relevant_inverse_matrix,
                    const std::vector<IndexSet> &owned_partitioning,
                    MPI_Comm                     mpi_communicator);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

    /*!
     * Transposed matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    Tvmult(VectorType &dst, const VectorType &src) const;

    const std::vector<IndexSet> &
    get_partitioning() const
    {
      return owned_partitioning;
    }

    unsigned int
    m() const
    {
      return block_01->n();
    }

    unsigned int
    n() const
    {
      return block_01->n();
    }

  private:
    /*!
     * Smart pointer to system matrix block 01.
     */
    const SmartPointer<const BlockType> block_01;

    /*!
     * Smart pointer to system matrix block 10.
     */
    const SmartPointer<const BlockType> block_10;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrixType> relevant_inverse_matrix;



    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////



  template <typename BlockMatrixType, typename InverseMatrixType>
  SchurComplement<BlockMatrixType, InverseMatrixType>::SchurComplement(
    const BlockMatrixType &      system_matrix,
    const InverseMatrixType &    relevant_inverse_matrix,
    const std::vector<IndexSet> &owned_partitioning,
    MPI_Comm                     mpi_communicator)
    : block_01(&(system_matrix.block(0, 1)))
    , block_10(&(system_matrix.block(1, 0)))
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
  {}

  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  SchurComplement<BlockMatrixType, InverseMatrixType>::vmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    block_01->vmult(tmp1, src);
    relevant_inverse_matrix->vmult(tmp2, tmp1);
    block_10->vmult(dst, tmp2);
  }


  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  SchurComplement<BlockMatrixType, InverseMatrixType>::Tvmult(
    VectorType &      dst,
    const VectorType &src) const
  {
    block_01->vmult(tmp1, src);
    relevant_inverse_matrix->Tvmult(tmp2, tmp1);
    block_10->vmult(dst, tmp2);
  }



  /*!
   * @class ApproxSchurComplement
   *
   * @brief Implements an approximate MPI parallel Schur complement
   *
   * Implements a parallel Schur complement through the use of an inner inverse
   * matrix, i.e., if we want to solve
   * \f{eqnarray}{
   *	\left(
   *	\begin{array}{cc}
   *		A & B^T \\
   *		B & 0
   *	\end{array}
   *	\right)
   *	\left(
   *	\begin{array}{c}
   *		\sigma \\
   *		u
   *	\end{array}
   *	\right)
   *	=
   *	\left(
   *	\begin{array}{c}
   *		f \\
   *		0
   *	\end{array}
   *	\right)
   * \f}
   * and know that \f$A\f$ is invertible then we first define the inverse and
   *define the Schur complement as \f{eqnarray}{ \tilde S = BP_A^{-1}B^T \f}
   *to solve for \f$u\f$. The inverse must be separately given to the class as
   *an input argument.
   */
  template <typename BlockMatrixType, typename InverseMatrixType>
  class ApproxSchurComplement : public Subscriptor
  {
  private:
    using BlockType = dealii::Preconditioners::WrapperMatrix;
    // using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor. The user must take care to pass the correct inverse of the
     * upper left block of the system matrix.
     *
     * @param system_matrix
     * 	Block Matrix
     * @param relevant_inverse_matrix
     * 	Inverse of upper left block of the system matrix.
     * @param owned_partitioning
     * @param mpi_communicator
     */
    ApproxSchurComplement(const BlockMatrixType &      system_matrix,
                          const InverseMatrixType &    relevant_inverse_matrix,
                          const std::vector<IndexSet> &owned_partitioning,
                          MPI_Comm                     mpi_communicator);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(TrilinosWrappers::MPI::Vector &      dst,
          const TrilinosWrappers::MPI::Vector &src) const;

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(dealii::Preconditioners::MultiVector2 &      dst,
          const dealii::Preconditioners::MultiVector2 &src) const;

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(dealii::Preconditioners::MultiVector2::Column &      dst,
          const dealii::Preconditioners::MultiVector2::Column &src) const;


    /*!
     * Transposed matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    Tvmult(TrilinosWrappers::MPI::Vector &      dst,
           const TrilinosWrappers::MPI::Vector &src) const;

    /*!
     * Transposed matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    Tvmult(dealii::Preconditioners::MultiVector2 &      dst,
           const dealii::Preconditioners::MultiVector2 &src) const;

    /*!
     * Transposed matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    Tvmult(dealii::Preconditioners::MultiVector2::Column &      dst,
           const dealii::Preconditioners::MultiVector2::Column &src) const;

    const std::vector<IndexSet> &
    get_partitioning() const
    {
      return owned_partitioning;
    }

    unsigned int
    m() const
    {
      return owned_partitioning[1].size();
    }

    unsigned int
    n() const
    {
      return owned_partitioning[1].size();
    }

  private:
    /*!
     * Smart pointer to system matrix block 01.
     */
    // const SmartPointer<const BlockType> block_01;
    const dealii::Preconditioners::WrapperMatrix block_01;

    /*!
     * Smart pointer to system matrix block 10.
     */
    const dealii::Preconditioners::WrapperMatrix block_10;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const dealii::Preconditioners::WrapperPreconditioner
      relevant_inverse_matrix;


    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable TrilinosWrappers::MPI::Vector tmp1, tmp2;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename BlockMatrixType, typename InverseMatrixType>
  ApproxSchurComplement<BlockMatrixType, InverseMatrixType>::
    ApproxSchurComplement(const BlockMatrixType &      system_matrix,
                          const InverseMatrixType &    relevant_inverse_matrix,
                          const std::vector<IndexSet> &owned_partitioning,
                          MPI_Comm                     mpi_communicator)
    : block_01(system_matrix.block(0, 1))
    , block_10(system_matrix.block(1, 0))
    , relevant_inverse_matrix(relevant_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[0], mpi_communicator)
    , tmp2(owned_partitioning[0], mpi_communicator)
  {}

  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  ApproxSchurComplement<BlockMatrixType, InverseMatrixType>::vmult(
    TrilinosWrappers::MPI::Vector &      dst,
    const TrilinosWrappers::MPI::Vector &src) const
  {
    block_01.vmult(tmp1, src);
    relevant_inverse_matrix.vmult(tmp2, tmp1);
    block_10.vmult(dst, tmp2);
  }


  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  ApproxSchurComplement<BlockMatrixType, InverseMatrixType>::vmult(
    dealii::Preconditioners::MultiVector2 &      dst,
    const dealii::Preconditioners::MultiVector2 &src) const
  {
    dealii::Preconditioners::MultiVector2 mv_tmp1, mv_tmp2;
    mv_tmp1.reinit(src.n_vectors(), tmp1.size());
    mv_tmp2.reinit(src.n_vectors(), tmp1.size());

    block_01.vmult(mv_tmp1, src);
    relevant_inverse_matrix.vmult(mv_tmp2, mv_tmp1);
    block_10.vmult(dst, mv_tmp2);
  }


  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  ApproxSchurComplement<BlockMatrixType, InverseMatrixType>::vmult(
    dealii::Preconditioners::MultiVector2::Column &      dst,
    const dealii::Preconditioners::MultiVector2::Column &src) const
  {
    dealii::Preconditioners::MultiVector2::Column mv_tmp1(
      tmp1.trilinos_vector(), 0, Copy),
      mv_tmp2(mv_tmp1);

    block_01.vmult(mv_tmp1, src);
    relevant_inverse_matrix.vmult(mv_tmp2, mv_tmp1);
    block_10.vmult(dst, mv_tmp2);
  }



  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  ApproxSchurComplement<BlockMatrixType, InverseMatrixType>::Tvmult(
    TrilinosWrappers::MPI::Vector &      dst,
    const TrilinosWrappers::MPI::Vector &src) const
  {
    block_01.vmult(tmp1, src);
    relevant_inverse_matrix.Tvmult(tmp2, tmp1);
    block_10.vmult(dst, tmp2);
  }

  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  ApproxSchurComplement<BlockMatrixType, InverseMatrixType>::Tvmult(
    dealii::Preconditioners::MultiVector2 &      dst,
    const dealii::Preconditioners::MultiVector2 &src) const
  {
    dealii::Preconditioners::MultiVector2 mv_tmp1, mv_tmp2;
    mv_tmp1.reinit(src.n_vectors(), tmp1.size());
    mv_tmp2.reinit(src.n_vectors(), tmp1.size());

    block_01.vmult(mv_tmp1, src);
    relevant_inverse_matrix.Tvmult(mv_tmp2, mv_tmp1);
    block_10.vmult(dst, mv_tmp2);
  }

  template <typename BlockMatrixType, typename InverseMatrixType>
  void
  ApproxSchurComplement<BlockMatrixType, InverseMatrixType>::Tvmult(
    dealii::Preconditioners::MultiVector2::Column &      dst,
    const dealii::Preconditioners::MultiVector2::Column &src) const
  {
    dealii::Preconditioners::MultiVector2::Column mv_tmp1(
      tmp1.trilinos_vector(), 0, Copy),
      mv_tmp2(mv_tmp1);

    block_01.vmult(mv_tmp1, src);
    relevant_inverse_matrix.Tvmult(mv_tmp2, mv_tmp1);
    block_10.vmult(dst, mv_tmp2);
  }


  ////////////////////////////////////////////
  // SchurComplementLowerBlock
  ////////////////////////////////////////////
  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType,
            typename ApproxInverseMatrixType>
  class SchurComplementLowerBlock : public Subscriptor
  {
  private:
    using BlockType = typename BlockMatrixType::BlockType;

  public:
    /*!
     * Constructor.
     */
    SchurComplementLowerBlock(
      const BlockMatrixType &        system_matrix,
      const InverseMatrixType &      relevant_inverse_matrix,
      const ApproxInverseMatrixType &relevant_approx_inverse_matrix,
      const std::vector<IndexSet> &  owned_partitioning,
      const bool                     do_full_solve,
      MPI_Comm                       mpi_communicator);

    /*!
     * Matrix-vector product.
     *
     * @param dst
     * @param src
     */
    void
    vmult(VectorType &dst, const VectorType &src) const;

  private:
    /*!
     * Smart pointer to system matrix block 12.
     */
    const SmartPointer<const BlockType> block_12;

    /*!
     * Smart pointer to system matrix block 21.
     */
    const SmartPointer<const BlockType> block_21;

    /*!
     * Smart pointer to inverse upper left block of the system matrix.
     */
    const SmartPointer<const InverseMatrixType> relevant_inverse_matrix;

    const SmartPointer<const ApproxInverseMatrixType>
      relevant_approx_inverse_matrix;

    /*!
     * Index set to initialize tmp vectors using only locally owned partition.
     */
    const std::vector<IndexSet> &owned_partitioning;

    const bool do_full_solve;

    /*!
     * Current MPI communicator.
     */
    MPI_Comm mpi_communicator;

    /*!
     * Muatable types for temporary vectors.
     */
    mutable VectorType tmp1, tmp2;
  };


  ///////////////////////////////////////
  /// Implementation
  ///////////////////////////////////////


  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType,
            typename ApproxInverseMatrixType>
  SchurComplementLowerBlock<BlockMatrixType,
                            VectorType,
                            InverseMatrixType,
                            ApproxInverseMatrixType>::
    SchurComplementLowerBlock(
      const BlockMatrixType &        system_matrix,
      const InverseMatrixType &      relevant_inverse_matrix,
      const ApproxInverseMatrixType &relevant_approx_inverse_matrix,
      const std::vector<IndexSet> &  owned_partitioning,
      const bool                     do_full_solve,
      MPI_Comm                       mpi_communicator)
    : block_12(&(system_matrix.block(1, 2)))
    , block_21(&(system_matrix.block(2, 1)))
    , relevant_inverse_matrix(&relevant_inverse_matrix)
    , relevant_approx_inverse_matrix(&relevant_approx_inverse_matrix)
    , owned_partitioning(owned_partitioning)
    , do_full_solve(do_full_solve)
    , mpi_communicator(mpi_communicator)
    , tmp1(owned_partitioning[1], mpi_communicator)
    , tmp2(owned_partitioning[1], mpi_communicator)
  {}

  template <typename BlockMatrixType,
            typename VectorType,
            typename InverseMatrixType,
            typename ApproxInverseMatrixType>
  void
  SchurComplementLowerBlock<BlockMatrixType,
                            VectorType,
                            InverseMatrixType,
                            ApproxInverseMatrixType>::vmult(VectorType &dst,
                                                            const VectorType
                                                              &src) const
  {
    if (do_full_solve == true)
      {
        block_12->vmult(tmp1, src);
        relevant_inverse_matrix->vmult(tmp2, tmp1);
        block_21->vmult(dst, tmp2);
      }
    else
      {
        block_12->vmult(tmp1, src);
        relevant_approx_inverse_matrix->vmult(tmp2, tmp1);
        block_21->vmult(dst, tmp2);
      }
  }
} // end namespace LinearAlgebra

DYCOREPLANET_CLOSE_NAMESPACE
#endif