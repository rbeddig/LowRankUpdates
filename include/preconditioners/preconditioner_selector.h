/**
 * @file preconditioner_selector.h
 * @author Rebekka Beddig
 * @version 0.1
 */
#ifndef PRECONDITIONER_SELECTOR
#define PRECONDITIONER_SELECTOR

// Deal ii
#include <deal.II/base/subscriptor.h>

#include <deal.II/lac/trilinos_precondition.h>

// own
#include <preconditioners/updates2.h>
#include <preconditioners/wrappers.h>

DEAL_II_NAMESPACE_OPEN
namespace Preconditioners
{
  /**
   * @brief Class to select the preconditioner for the upper left block of the saddle
   * point matrix.
   */
  class PreconditionerSelector : public Subscriptor
  {
  public:
    using UpdateType =
      Preconditioners::LowRankUpdates<WrapperMatrix, WrapperPreconditioner>;

    /**
     * @brief Default constructor.
     */
    PreconditionerSelector() = default;

    /**
     * @brief Constructor, sets the preconditioner to @ref name (amg, ilu, jacobi)
     */
    PreconditionerSelector(const std::string &name)
      : Subscriptor()
      , preconditioner_name(name)
    {}

    /**
     * Destructor
     */
    ~PreconditionerSelector()
    {}

    /**
     * @brief Select the desired preconditioner.
     * @param name Choose 'jacobi','amg' or 'ilu'
     */
    void
    select(const std::string &name)
    {
      preconditioner_name = name;
    }


    /**
     * @brief Applies the preconditioner to a vector.
     *
     * @param[out] trg result
     * @param[in] src source vector
     */
    template <typename VectorType>
    void
    vmult(VectorType &trg, const VectorType &src) const
    {
      wrap_preconditioner.vmult(trg, src);
    }


    /**
     * @brief Applies the transposed preconditioner to a
     * vector (if defined)
     *
     * @param[out] trg result
     * @param[in] src source vector
     */
    template <typename VectorType>
    void
    Tvmult(VectorType &trg, const VectorType &src) const
    {
      wrap_preconditioner.Tvmult(trg, src);
    }

    /**
     * @brief Initializes the selected preconditioner.
     * @param matrix matrix for which we want to compute the preconditioner
     */
    void
    initialize(const TrilinosWrappers::SparseMatrix &matrix)
    {
      if (preconditioner_name == "amg")
        {
          amg_preconditioner.initialize(matrix, amg_data);
          wrap_preconditioner.reinit(amg_preconditioner);
        }
      else if (preconditioner_name == "ilu")
        {
          ilu_preconditioner.initialize(matrix, ilu_data);
          wrap_preconditioner.reinit(ilu_preconditioner);
        }
      else if (preconditioner_name == "jacobi")
        {
          jacobi_preconditioner.initialize(matrix, jacobi_data);
          wrap_preconditioner.reinit(jacobi_preconditioner);
        }
      else
        Assert(false, ExcNotImplemented());
    }

    /**
     * @brief Initializes an update of rank @ref rank (Not yet implemented)
     */
    void
    initialize_update(const TrilinosWrappers::SparseMatrix &matrix,
                      const std::vector<IndexSet>           partitioning)
    {
      Assert(false, ExcNotImplemented());
      std::ignore = matrix;
      std::ignore = partitioning;
    }

    /**
     * @brief Returns a reference to the underlying preconditioner.
     *
     * @return const TrilinosWrappers::PreconditionBase& Reference to the underlying preconditioner.
     */
    const TrilinosWrappers::PreconditionBase &
    dealii_preconditioner() const
    {
      if (preconditioner_name == "amg")
        return amg_preconditioner;
      else if (preconditioner_name == "ilu")
        return ilu_preconditioner;
      else if (preconditioner_name == "jacobi")
        return jacobi_preconditioner;
      else
        {
          Assert(false, ExcNotImplemented());
          return jacobi_preconditioner; // never reached, but removes the
                                        // compiler warning
        }
    }


    /** @brief Set the parameters for the AMG preconditioner. */
    void
    set_data(
      const typename TrilinosWrappers::PreconditionAMG::AdditionalData &data)
    {
      amg_data = data;
    }

    /** @brief Sets the parameters for the ILU preconditioner. */
    void
    set_data(
      const typename TrilinosWrappers::PreconditionILU::AdditionalData &data)
    {
      ilu_data = data;
    }

    /** @brief Sets the parameters for the Jacobi preconditioner. */
    void
    set_data(
      const typename TrilinosWrappers::PreconditionJacobi::AdditionalData &data)
    {
      jacobi_data = data;
    }

    /** @brief Set the parameters for the update. */
    void
    set_data(const typename UpdateType::AdditionalData &data)
    {
      update_data = data;
    }

    /** @brief transposes the preconditioner if possible */
    void
    transpose()
    {
      if (preconditioner_name == "amg")
        amg_preconditioner.transpose();
      else if (preconditioner_name == "ilu")
        ilu_preconditioner.transpose();
      else if (preconditioner_name == "jacobi")
        jacobi_preconditioner.transpose();
    }

  private:
    std::string preconditioner_name;

    // Optional preconditioners
    Preconditioners::WrapperPreconditioner wrap_preconditioner;
    TrilinosWrappers::PreconditionAMG      amg_preconditioner;
    TrilinosWrappers::PreconditionILU      ilu_preconditioner;
    TrilinosWrappers::PreconditionJacobi   jacobi_preconditioner;


    // and the corresponding data structs (that contain the preconditioner
    // parameters)
    TrilinosWrappers::PreconditionAMG::AdditionalData    amg_data;
    TrilinosWrappers::PreconditionILU::AdditionalData    ilu_data;
    TrilinosWrappers::PreconditionJacobi::AdditionalData jacobi_data;

    UpdateType                 updated_preconditioner;
    UpdateType::AdditionalData update_data;
  };

} // namespace Preconditioners
DEAL_II_NAMESPACE_CLOSE
#endif