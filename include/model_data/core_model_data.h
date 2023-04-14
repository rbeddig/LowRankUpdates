/**
 * @file core_model_data.h
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/physical_constants.h>
#include <model_data/reference_quantities.h>


DYCOREPLANET_OPEN_NAMESPACE

/*!
 * @namespace CoreModelData
 *
 * Namespace containing namespaces for different models such as Boussinesq
 * appromimation or the full primitive equations
 */
namespace CoreModelData
{
  template <int dim>
  class TangentialFunction : public TensorFunction<1, dim>
  {
  public:
    TangentialFunction(const double scale_factor);

    virtual Tensor<1, dim>
    value(const Point<dim> &p) const override;

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>> &  values) const override;

  private:
    const double scale_factor;

    /*!
     * Euler angles
     */
    const double alpha_, beta_, gamma_;

    Tensor<2, dim> rotation;
  };

  template <int dim>
  class RadialFunction : public TensorFunction<1, dim>
  {
  public:
    RadialFunction(const double scale_factor);

    virtual Tensor<1, dim>
    value(const Point<dim> &p) const override;

    virtual void
    value_list(const std::vector<Point<dim>> &points,
               std::vector<Tensor<1, dim>> &  values) const override;

  private:
    const double scale_factor;
  };

  /*!
   * Return the Reynolds number of the flow.
   */
  double
  get_reynolds_number(const double velocity,
                      const double length,
                      const double kinematic_viscosity);

  /*!
   * Return the Peclet number of the flow. This is the ratio af advective time
   * scale to the diffusive time scale.
   */
  double
  get_peclet_number(const double velocity,
                    const double length,
                    const double thermal_diffusivity);

  /*!
   * Return the Rossby number of the flow. This is the ratio of inertial to
   * coriolis forces.
   */
  double
  get_rossby_number(const double length,
                    const double omega,
                    const double velocity);

  /*!
   * Reference acceleration. The inverse enters as a scaling factor to the
   * right-hand side of the equation.
   */
  double
  get_reference_accelleration(const double length, const double velocity);

  /*!
   * Return the Grashoff number of the flow.
   */
  double
  get_grashoff_number(const int    dim,
                      const double gravity_constant,
                      const double expansion_coefficient,
                      const double temperature_change,
                      const double length,
                      const double kinematic_viscosity);

  /*!
   * Return the Prandtl number of the flow.
   */
  double
  get_prandtl_number(const double kinematic_viscosity,
                     const double thermal_diffusivity);

  /*!
   * Return the Rayleigh number of the flow.
   */
  double
  get_rayleigh_number(const int    dim,
                      const double gravity_constant,
                      const double expansion_coefficient,
                      const double temperature_change,
                      const double length,
                      const double kinematic_viscosity,
                      const double thermal_diffusivity);

  /*!
   * Density as function of temperature.
   *
   * @return desity
   */
  double
  density(const double density,
          const double expansion_coefficient,
          const double temperature,
          const double temperature_bottom);

  /*!
   * Density scaling as function of temperature. This is the density devided
   * by the reference density.
   *
   * @return desity
   */
  double
  density_scaling(const double expansion_coefficient,
                  const double temperature,
                  const double temperature_bottom);

  /*!
   * Compute vertical gravity vector at a given point.
   *
   * @return vertical gravity vector
   */
  template <int dim>
  Tensor<1, dim>
  vertical_gravity_vector(const Point<dim> &p, const double gravity_constant);

  /*!
   * Compute gravity vector at a given point.
   *
   * @return gravity vector
   */
  template <int dim>
  Tensor<1, dim>
  gravity_vector(const Point<dim> &p, const double gravity_constant);

  /*!
   * Compute coriolis vector at a given point.
   *
   * @param p
   * @return coriolis vector
   */
  template <int dim>
  Tensor<1, dim>
  coriolis_vector(const Point<dim> &p, const double omega);

} // namespace CoreModelData

/*
 * Extern template instantiations
 */
extern template class CoreModelData::TangentialFunction<2>;
extern template class CoreModelData::TangentialFunction<3>;

extern template class CoreModelData::RadialFunction<2>;
extern template class CoreModelData::RadialFunction<3>;

extern template Tensor<1, 2>
CoreModelData::vertical_gravity_vector<2>(const Point<2> &p,
                                          const double    gravity_constant);
extern template Tensor<1, 3>
CoreModelData::vertical_gravity_vector<3>(const Point<3> &p,
                                          const double    gravity_constant);

extern template Tensor<1, 2>
CoreModelData::gravity_vector<2>(const Point<2> &p,
                                 const double    gravity_constant);
extern template Tensor<1, 3>
CoreModelData::gravity_vector<3>(const Point<3> &p,
                                 const double    gravity_constant);

extern template Tensor<1, 2>
CoreModelData::coriolis_vector(const Point<2> & /*p*/, const double omega);

extern template Tensor<1, 3>
CoreModelData::coriolis_vector(const Point<3> & /*p*/, const double omega);

DYCOREPLANET_CLOSE_NAMESPACE
