/**
 * @file core_model_data.cc
 * @author Konrad Simon
 * @version 0.1
 */
#include <model_data/core_model_data.h>

#include <model_data/core_model_data.tpp>

DYCOREPLANET_OPEN_NAMESPACE

double
CoreModelData::get_reynolds_number(const double velocity,
                                   const double length,
                                   const double kinematic_viscosity)
{
  return (velocity * length) / kinematic_viscosity;
}


double
CoreModelData::get_peclet_number(const double velocity,
                                 const double length,
                                 const double thermal_diffusivity)
{
  return (velocity * length) / thermal_diffusivity;
}

double
CoreModelData::get_rossby_number(const double length,
                                 const double omega,
                                 const double velocity)
{
  return velocity / (length * omega);
}


double
CoreModelData::get_reference_accelleration(const double length,
                                           const double velocity)
{
  return velocity * velocity / length;
}


double
CoreModelData::get_grashoff_number(const int    dim,
                                   const double gravity_constant,
                                   const double expansion_coefficient,
                                   const double temperature_change,
                                   const double length,
                                   const double kinematic_viscosity)
{
  return (gravity_constant * expansion_coefficient * temperature_change *
          std::pow(length, dim) / kinematic_viscosity);
}


double
CoreModelData::get_prandtl_number(const double kinematic_viscosity,
                                  const double thermal_diffusivity)
{
  return kinematic_viscosity / thermal_diffusivity;
}


double
CoreModelData::get_rayleigh_number(const int    dim,
                                   const double gravity_constant,
                                   const double expansion_coefficient,
                                   const double temperature_change,
                                   const double length,
                                   const double kinematic_viscosity,
                                   const double thermal_diffusivity)
{
  return gravity_constant * expansion_coefficient * temperature_change *
         std::pow(length, dim) *
         get_prandtl_number(kinematic_viscosity, thermal_diffusivity);
}


double
CoreModelData::density(const double density,
                       const double expansion_coefficient,
                       const double temperature,
                       const double temperature_ref)
{
  return density *
         (1 - expansion_coefficient * (temperature - temperature_ref));
}


double
CoreModelData::density_scaling(const double expansion_coefficient,
                               const double temperature,
                               const double temperature_ref)
{
  return (1 - expansion_coefficient * (temperature - temperature_ref));
}

DYCOREPLANET_CLOSE_NAMESPACE
