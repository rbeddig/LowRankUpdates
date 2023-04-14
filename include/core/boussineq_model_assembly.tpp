/**
 * @file boussineq_model_assembly.tpp
 * @author Konrad Simon, Rebekka Beddig
 * @version 0.1
 */
#pragma once

#include <core/boussineq_model_assembly.h>

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
      NSEPreconditioner<dim>::NSEPreconditioner(
        const double              time_step,
        const double              time_index,
        const FiniteElement<dim> &nse_fe,
        const Quadrature<dim> &   nse_quadrature,
        const Mapping<dim> &      mapping,
        const UpdateFlags         update_flags)
        : nse_fe_values(mapping, nse_fe, nse_quadrature, update_flags)
        , phi_u(nse_fe.dofs_per_cell)
        , grad_phi_u(nse_fe.dofs_per_cell)
        // , grads_phi_u(nse_fe.dofs_per_cell)
        , phi_p(nse_fe.dofs_per_cell)
        , nse_picard_iterate(nse_quadrature.size())
        , time_step(time_step)
        , time_index(time_index)
      {}


      template <int dim>
      NSEPreconditioner<dim>::NSEPreconditioner(
        const NSEPreconditioner<dim> &scratch)
        : nse_fe_values(scratch.nse_fe_values.get_mapping(),
                        scratch.nse_fe_values.get_fe(),
                        scratch.nse_fe_values.get_quadrature(),
                        scratch.nse_fe_values.get_update_flags())
        , phi_u(scratch.phi_u)
        , grad_phi_u(scratch.grad_phi_u)
        // , grads_phi_u(scratch.grads_phi_u)
        , phi_p(scratch.phi_p)
        , nse_picard_iterate(scratch.nse_picard_iterate)
        , time_step(scratch.time_step)
        , time_index(scratch.time_index)
      {}



      ////////////////////////////////////
      /// NSE system
      ////////////////////////////////////

      template <int dim>
      NSESystem<dim>::NSESystem(const double              time_step,
                                const double              time_index,
                                const FiniteElement<dim> &nse_fe,
                                const Mapping<dim> &      mapping,
                                const Quadrature<dim> &   nse_quadrature,
                                const UpdateFlags         nse_update_flags,
                                const FiniteElement<dim> &temperature_fe,
                                const UpdateFlags temperature_update_flags)
        : NSEPreconditioner<dim>(time_step,
                                 time_index,
                                 nse_fe,
                                 nse_quadrature,
                                 mapping,
                                 nse_update_flags)
        , temperature_fe_values(mapping,
                                temperature_fe,
                                nse_quadrature,
                                temperature_update_flags)
        , nse_fe_values(mapping, nse_fe, nse_quadrature, nse_update_flags)
        , phi_u(nse_fe.dofs_per_cell)
        , grads_phi_u(nse_fe.dofs_per_cell)
        , div_phi_u(nse_fe.dofs_per_cell)
        , old_temperature_values(nse_quadrature.size())
        , old_velocity_values(nse_quadrature.size())
        , old_velocity_grads(nse_quadrature.size())
        , nse_picard_iterate(nse_quadrature.size())
      {}


      template <int dim>
      NSESystem<dim>::NSESystem(const NSESystem<dim> &scratch)
        : NSEPreconditioner<dim>(scratch)
        , temperature_fe_values(
            scratch.temperature_fe_values.get_mapping(),
            scratch.temperature_fe_values.get_fe(),
            scratch.temperature_fe_values.get_quadrature(),
            scratch.temperature_fe_values.get_update_flags())
        , nse_fe_values(scratch.nse_fe_values.get_mapping(),
                        scratch.nse_fe_values.get_fe(),
                        scratch.nse_fe_values.get_quadrature(),
                        scratch.nse_fe_values.get_update_flags())
        , phi_u(scratch.phi_u)
        , grads_phi_u(scratch.grads_phi_u)
        , div_phi_u(scratch.div_phi_u)
        , old_temperature_values(scratch.old_temperature_values)
        , old_velocity_values(scratch.old_velocity_values)
        , old_velocity_grads(scratch.old_velocity_grads)
        , nse_picard_iterate(scratch.nse_picard_iterate)
      {}


      ////////////////////////////////////
      /// Temperature matrix
      ////////////////////////////////////

      template <int dim>
      TemperatureMatrix<dim>::TemperatureMatrix(
        const double              time_step,
        const double              time_index,
        const FiniteElement<dim> &temperature_fe,
        const Mapping<dim> &      mapping,
        const Quadrature<dim> &   temperature_quadrature)
        : temperature_fe_values(mapping,
                                temperature_fe,
                                temperature_quadrature,
                                update_values | update_gradients |
                                  update_JxW_values)
        , phi_T(temperature_fe.dofs_per_cell)
        , grad_phi_T(temperature_fe.dofs_per_cell)
        , time_step(time_step)
        , time_index(time_index)
      {}


      template <int dim>
      TemperatureMatrix<dim>::TemperatureMatrix(
        const TemperatureMatrix<dim> &scratch)
        : temperature_fe_values(
            scratch.temperature_fe_values.get_mapping(),
            scratch.temperature_fe_values.get_fe(),
            scratch.temperature_fe_values.get_quadrature(),
            scratch.temperature_fe_values.get_update_flags())
        , phi_T(scratch.phi_T)
        , grad_phi_T(scratch.grad_phi_T)
        , time_step(scratch.time_step)
        , time_index(scratch.time_index)
      {}


      ////////////////////////////////////
      /// Temperture RHS
      ////////////////////////////////////

      template <int dim>
      TemperatureRHS<dim>::TemperatureRHS(
        const double              time_step,
        const double              time_index,
        const FiniteElement<dim> &temperature_fe,
        const FiniteElement<dim> &nse_fe,
        const Mapping<dim> &      mapping,
        const Quadrature<dim> &   quadrature)
        : temperature_fe_values(mapping,
                                temperature_fe,
                                quadrature,
                                update_values | update_gradients |
                                  update_hessians | update_quadrature_points |
                                  update_JxW_values)
        , nse_fe_values(mapping,
                        nse_fe,
                        quadrature,
                        update_values | update_gradients)
        , phi_T(temperature_fe.dofs_per_cell)
        , grad_phi_T(temperature_fe.dofs_per_cell)
        , old_velocity_values(quadrature.size())
        , old_temperature_values(quadrature.size())
        , old_temperature_grads(quadrature.size())
        , time_step(time_step)
        , time_index(time_index)
      {}


      template <int dim>
      TemperatureRHS<dim>::TemperatureRHS(const TemperatureRHS<dim> &scratch)
        : temperature_fe_values(
            scratch.temperature_fe_values.get_mapping(),
            scratch.temperature_fe_values.get_fe(),
            scratch.temperature_fe_values.get_quadrature(),
            scratch.temperature_fe_values.get_update_flags())
        , nse_fe_values(scratch.nse_fe_values.get_mapping(),
                        scratch.nse_fe_values.get_fe(),
                        scratch.nse_fe_values.get_quadrature(),
                        scratch.nse_fe_values.get_update_flags())
        , phi_T(scratch.phi_T)
        , grad_phi_T(scratch.grad_phi_T)
        , old_velocity_values(scratch.old_velocity_values)
        , old_temperature_values(scratch.old_temperature_values)
        , old_temperature_grads(scratch.old_temperature_grads)
        , time_step(scratch.time_step)
        , time_index(scratch.time_index)
      {}


      ////////////////////////////////////
      /// NSE velocity mass
      ////////////////////////////////////

      template <int dim>
      VelocityMass<dim>::VelocityMass(const double              time_step,
                                      const double              time_index,
                                      const FiniteElement<dim> &nse_fe,
                                      const Quadrature<dim> &   nse_quadrature,
                                      const Mapping<dim> &      mapping,
                                      const UpdateFlags         update_flags)
        : nse_fe_values(mapping, nse_fe, nse_quadrature, update_flags)
        , phi_u(nse_fe.dofs_per_cell)
        , time_step(time_step)
        , time_index(time_index)
      {}


      template <int dim>
      VelocityMass<dim>::VelocityMass(const VelocityMass<dim> &scratch)
        : nse_fe_values(scratch.nse_fe_values.get_mapping(),
                        scratch.nse_fe_values.get_fe(),
                        scratch.nse_fe_values.get_quadrature(),
                        scratch.nse_fe_values.get_update_flags())
        , phi_u(scratch.phi_u)
        , time_step(scratch.time_step)
        , time_index(scratch.time_index)
      {}

    } // namespace Scratch



    namespace CopyData
    {
      ////////////////////////////////////
      /// NSE preconditioner copy
      ////////////////////////////////////

      template <int dim>
      NSEPreconditioner<dim>::NSEPreconditioner(
        const FiniteElement<dim> &nse_fe)
        : local_matrix(nse_fe.dofs_per_cell, nse_fe.dofs_per_cell)
        , local_schur_matrix(nse_fe.dofs_per_cell, nse_fe.dofs_per_cell)
        , local_dof_indices(nse_fe.dofs_per_cell)
      {}


      template <int dim>
      NSEPreconditioner<dim>::NSEPreconditioner(
        const NSEPreconditioner<dim> &data)
        : local_matrix(data.local_matrix)
        , local_schur_matrix(data.local_schur_matrix)
        , local_dof_indices(data.local_dof_indices)
      {}


      ////////////////////////////////////
      /// NSE system copy
      ////////////////////////////////////

      template <int dim>
      NSESystem<dim>::NSESystem(const FiniteElement<dim> &nse_fe)
        : NSEPreconditioner<dim>(nse_fe)
        , local_rhs(nse_fe.dofs_per_cell)
        , local_pressure_shape_function_integrals(nse_fe.dofs_per_cell)
        , local_constant_matrix(nse_fe.dofs_per_cell, nse_fe.dofs_per_cell)
        , local_diffusion_matrix(nse_fe.dofs_per_cell, nse_fe.dofs_per_cell)
        , local_advection_matrix(nse_fe.dofs_per_cell, nse_fe.dofs_per_cell)
      {}


      template <int dim>
      NSESystem<dim>::NSESystem(const NSESystem<dim> &data)
        : NSEPreconditioner<dim>(data)
        , local_rhs(data.local_rhs)
        , local_pressure_shape_function_integrals(
            data.local_pressure_shape_function_integrals)
        , local_constant_matrix(data.local_constant_matrix)
        , local_diffusion_matrix(data.local_diffusion_matrix)
        , local_advection_matrix(data.local_advection_matrix)
      {}


      ////////////////////////////////////
      /// Temperature system copy
      ////////////////////////////////////

      template <int dim>
      TemperatureMatrix<dim>::TemperatureMatrix(
        const FiniteElement<dim> &temperature_fe)
        : local_mass_matrix(temperature_fe.dofs_per_cell,
                            temperature_fe.dofs_per_cell)
        , local_advection_matrix(temperature_fe.dofs_per_cell,
                                 temperature_fe.dofs_per_cell)
        , local_stiffness_matrix(temperature_fe.dofs_per_cell,
                                 temperature_fe.dofs_per_cell)
        , local_dof_indices(temperature_fe.dofs_per_cell)
      {}


      template <int dim>
      TemperatureMatrix<dim>::TemperatureMatrix(
        const TemperatureMatrix<dim> &data)
        : local_mass_matrix(data.local_mass_matrix)
        , local_advection_matrix(data.local_advection_matrix)
        , local_stiffness_matrix(data.local_stiffness_matrix)
        , local_dof_indices(data.local_dof_indices)
      {}


      ////////////////////////////////////
      /// Temperature RHS copy
      ////////////////////////////////////

      template <int dim>
      TemperatureRHS<dim>::TemperatureRHS(
        const FiniteElement<dim> &temperature_fe)
        : local_rhs(temperature_fe.dofs_per_cell)
        , local_dof_indices(temperature_fe.dofs_per_cell)
        , matrix_for_bc(temperature_fe.dofs_per_cell,
                        temperature_fe.dofs_per_cell)
      {}


      template <int dim>
      TemperatureRHS<dim>::TemperatureRHS(const TemperatureRHS<dim> &data)
        : local_rhs(data.local_rhs)
        , local_dof_indices(data.local_dof_indices)
        , matrix_for_bc(data.matrix_for_bc)
      {}

      ////////////////////////////////////
      /// NSE velocity mass copy
      ////////////////////////////////////
      template <int dim>
      VelocityMass<dim>::VelocityMass(const FiniteElement<dim> &nse_fe)
        : local_matrix(nse_fe.dofs_per_cell, nse_fe.dofs_per_cell)
        , local_dof_indices(nse_fe.dofs_per_cell)
      {}


      template <int dim>
      VelocityMass<dim>::VelocityMass(const VelocityMass<dim> &data)
        : local_matrix(data.local_matrix)
        , local_dof_indices(data.local_dof_indices)
      {}



    } // namespace CopyData
  }   // namespace Assembly
} // namespace Standard

DYCOREPLANET_CLOSE_NAMESPACE
