/**
 * @file boussineq_model_assembly.h
 * @author Konrad Simon, Rebekka Beddig
 * @version 0.1
 */
#pragma once

// Deal.ii
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

// Aquaplanet3D
#include <base/config.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace Standard
{
  namespace Assembly
  {
    namespace Scratch
    {
      ////////////////////////////////////
      /// NSE preconditioner
      ////////////////////////////////////

      template <int dim>
      struct NSEPreconditioner
      {
        NSEPreconditioner(const double              time_step,
                          const double              time_index,
                          const FiniteElement<dim> &nse_fe,
                          const Quadrature<dim> &   nse_quadrature,
                          const Mapping<dim> &      mapping,
                          const UpdateFlags         update_flags);

        NSEPreconditioner(const NSEPreconditioner<dim> &data);

        FEValues<dim> nse_fe_values;

        std::vector<Tensor<1, dim>> phi_u;
        std::vector<Tensor<2, dim>> grad_phi_u;
        // std::vector<SymmetricTensor<2, dim>> grads_phi_u;
        std::vector<double> phi_p;

        // for the Picard iteration
        std::vector<Tensor<1, dim>> nse_picard_iterate;

        const double time_step;
        const double time_index;
      };


      ////////////////////////////////////
      /// NSE system
      ////////////////////////////////////

      template <int dim>
      struct NSESystem : public NSEPreconditioner<dim>
      {
        NSESystem(const double              time_step,
                  const double              time_index,
                  const FiniteElement<dim> &nse_fe,
                  const Mapping<dim> &      mapping,
                  const Quadrature<dim> &   nse_quadrature,
                  const UpdateFlags         nse_update_flags,
                  const FiniteElement<dim> &temperature_fe,
                  const UpdateFlags         temperature_update_flags);

        NSESystem(const NSESystem<dim> &data);

        FEValues<dim> temperature_fe_values;
        FEValues<dim> nse_fe_values;

        std::vector<Tensor<1, dim>>          phi_u;
        std::vector<SymmetricTensor<2, dim>> grads_phi_u;
        std::vector<double>                  div_phi_u;

        std::vector<double>         old_temperature_values;
        std::vector<Tensor<1, dim>> old_velocity_values;
        std::vector<Tensor<2, dim>> old_velocity_grads;

        // for the Picard iteration
        std::vector<Tensor<1, dim>> nse_picard_iterate;
      };


      ////////////////////////////////////
      /// Temperature matrix
      ////////////////////////////////////

      template <int dim>
      struct TemperatureMatrix
      {
        TemperatureMatrix(const double              time_step,
                          const double              time_index,
                          const FiniteElement<dim> &temperature_fe,
                          const Mapping<dim> &      mapping,
                          const Quadrature<dim> &   temperature_quadrature);

        TemperatureMatrix(const TemperatureMatrix<dim> &data);

        FEValues<dim> temperature_fe_values;

        std::vector<double>         phi_T;
        std::vector<Tensor<1, dim>> grad_phi_T;

        const double time_step;
        const double time_index;
      };


      ////////////////////////////////////
      /// Temperture RHS
      ////////////////////////////////////

      template <int dim>
      struct TemperatureRHS
      {
        TemperatureRHS(const double              time_step,
                       const double              time_index,
                       const FiniteElement<dim> &temperature_fe,
                       const FiniteElement<dim> &nse_fe,
                       const Mapping<dim> &      mapping,
                       const Quadrature<dim> &   quadrature);

        TemperatureRHS(const TemperatureRHS<dim> &data);

        FEValues<dim> temperature_fe_values;
        FEValues<dim> nse_fe_values;

        std::vector<double>         phi_T;
        std::vector<Tensor<1, dim>> grad_phi_T;

        std::vector<Tensor<1, dim>> old_velocity_values;
        std::vector<double>         old_temperature_values;
        std::vector<Tensor<1, dim>> old_temperature_grads;

        const double time_step;
        const double time_index;
      };



      ////////////////////////////////////
      /// velocity mass
      ////////////////////////////////////

      template <int dim>
      struct VelocityMass
      {
        VelocityMass(const double              time_step,
                     const double              time_index,
                     const FiniteElement<dim> &nse_fe,
                     const Quadrature<dim> &   nse_quadrature,
                     const Mapping<dim> &      mapping,
                     const UpdateFlags         update_flags);

        VelocityMass(const VelocityMass<dim> &data);

        FEValues<dim> nse_fe_values;

        std::vector<Tensor<1, dim>> phi_u;

        const double time_step;
        const double time_index;
      };



    } // namespace Scratch



    namespace CopyData
    {
      ////////////////////////////////////
      /// NSE preconditioner copy
      ////////////////////////////////////

      template <int dim>
      struct NSEPreconditioner
      {
        NSEPreconditioner(const FiniteElement<dim> &nse_fe);
        NSEPreconditioner(const NSEPreconditioner<dim> &data);

        NSEPreconditioner<dim> &
        operator=(const NSEPreconditioner<dim> &) = default;

        FullMatrix<double>                   local_matrix;
        FullMatrix<double>                   local_schur_matrix;
        std::vector<types::global_dof_index> local_dof_indices;
      };


      ////////////////////////////////////
      /// NSE system copy
      ////////////////////////////////////

      template <int dim>
      struct NSESystem : public NSEPreconditioner<dim>
      {
        NSESystem(const FiniteElement<dim> &nse_fe);

        NSESystem(const NSESystem<dim> &data);

        Vector<double> local_rhs;
        Vector<double> local_pressure_shape_function_integrals;

        // Needed to assemble the terms of the NSE separately
        // FullMatrix<double> local_matrix;
        FullMatrix<double> local_constant_matrix;
        FullMatrix<double> local_diffusion_matrix;
        FullMatrix<double> local_advection_matrix;
      };


      ////////////////////////////////////
      /// Temperature system copy
      ////////////////////////////////////

      template <int dim>
      struct TemperatureMatrix
      {
        TemperatureMatrix(const FiniteElement<dim> &temperature_fe);

        TemperatureMatrix(const TemperatureMatrix<dim> &data);

        FullMatrix<double> local_mass_matrix;
        FullMatrix<double> local_advection_matrix;
        FullMatrix<double> local_stiffness_matrix;

        std::vector<types::global_dof_index> local_dof_indices;
      };


      ////////////////////////////////////
      /// Temperature RHS copy
      ////////////////////////////////////

      template <int dim>
      struct TemperatureRHS
      {
        TemperatureRHS(const FiniteElement<dim> &temperature_fe);
        TemperatureRHS(const TemperatureRHS<dim> &data);
        Vector<double>                       local_rhs;
        std::vector<types::global_dof_index> local_dof_indices;
        FullMatrix<double>                   matrix_for_bc;
      };



      ////////////////////////////////////
      /// NSE velocity mass copy
      ////////////////////////////////////

      template <int dim>
      struct VelocityMass
      {
        VelocityMass(const FiniteElement<dim> &nse_fe);
        VelocityMass(const VelocityMass<dim> &data);

        VelocityMass<dim> &
        operator=(const VelocityMass<dim> &) = default;

        FullMatrix<double>                   local_matrix;
        std::vector<types::global_dof_index> local_dof_indices;
      };



    } // namespace CopyData
  }   // namespace Assembly
} // namespace Standard

// Extern template instantiations
extern template class Standard::Assembly::Scratch::NSEPreconditioner<2>;
extern template class Standard::Assembly::Scratch::NSESystem<2>;
extern template class Standard::Assembly::Scratch::TemperatureMatrix<2>;
extern template class Standard::Assembly::Scratch::TemperatureRHS<2>;
extern template class Standard::Assembly::Scratch::VelocityMass<2>;

extern template class Standard::Assembly::CopyData::NSEPreconditioner<2>;
extern template class Standard::Assembly::CopyData::NSESystem<2>;
extern template class Standard::Assembly::CopyData::TemperatureMatrix<2>;
extern template class Standard::Assembly::CopyData::TemperatureRHS<2>;
extern template class Standard::Assembly::CopyData::VelocityMass<2>;

extern template class Standard::Assembly::Scratch::NSEPreconditioner<3>;
extern template class Standard::Assembly::Scratch::NSESystem<3>;
extern template class Standard::Assembly::Scratch::TemperatureMatrix<3>;
extern template class Standard::Assembly::Scratch::TemperatureRHS<3>;
extern template class Standard::Assembly::Scratch::VelocityMass<3>;

extern template class Standard::Assembly::CopyData::NSEPreconditioner<3>;
extern template class Standard::Assembly::CopyData::NSESystem<3>;
extern template class Standard::Assembly::CopyData::TemperatureMatrix<3>;
extern template class Standard::Assembly::CopyData::TemperatureRHS<3>;
extern template class Standard::Assembly::CopyData::VelocityMass<3>;

DYCOREPLANET_CLOSE_NAMESPACE
