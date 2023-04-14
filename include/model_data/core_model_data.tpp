/**
 * @file core_model_data.tpp
 * @author Konrad Simon
 * @version 0.1
 */

#include <model_data/core_model_data.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  template <int dim>
  TangentialFunction<dim>::TangentialFunction(const double scale_factor)
    : TensorFunction<1, dim>()
    , scale_factor(scale_factor)
    , alpha_(numbers::PI / 3)
    , beta_(numbers::PI / 6)
    , gamma_(numbers::PI / 4)
  {
    rotation[0][0] =
      cos(alpha_) * cos(gamma_) - sin(alpha_) * cos(beta_) * sin(gamma_);
    rotation[0][1] =
      -cos(alpha_) * sin(gamma_) - sin(alpha_) * cos(beta_) * cos(gamma_);
    rotation[0][2] = sin(alpha_) * sin(beta_);

    rotation[1][0] =
      sin(alpha_) * cos(gamma_) + cos(alpha_) * cos(beta_) * sin(gamma_);
    rotation[1][1] =
      -sin(alpha_) * sin(gamma_) + cos(alpha_) * cos(beta_) * cos(gamma_);
    rotation[1][2] = -cos(alpha_) * sin(beta_);

    rotation[2][0] = sin(beta_) * sin(gamma_);
    rotation[2][1] = sin(beta_) * cos(gamma_);
    rotation[2][2] = cos(beta_);
  }

  template <int dim>
  Tensor<1, dim>
  TangentialFunction<dim>::value(const Point<dim> &p) const
  {
    return scale_factor * (rotation * p);
  }

  template <int dim>
  void
  TangentialFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                      std::vector<Tensor<1, dim>> &values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p].clear();
        values[p] = value(points[p]);
      }
  }



  template <int dim>
  RadialFunction<dim>::RadialFunction(const double scale_factor)
    : TensorFunction<1, dim>()
    , scale_factor(scale_factor)
  {}

  template <int dim>
  Tensor<1, dim>
  RadialFunction<dim>::value(const Point<dim> &p) const
  {
    return scale_factor * p;
  }

  template <int dim>
  void
  RadialFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                  std::vector<Tensor<1, dim>> &  values) const
  {
    Assert(points.size() == values.size(),
           ExcDimensionMismatch(points.size(), values.size()));

    for (unsigned int p = 0; p < points.size(); ++p)
      {
        values[p].clear();
        values[p] = value(points[p]);
      }
  }

  template <int dim>
  Tensor<1, dim>
  vertical_gravity_vector(const Point<dim> & /* p */,
                          const double gravity_constant)
  {
    Tensor<1, dim> e_z;
    e_z[dim - 1] = 1;

    return -gravity_constant * e_z;
  }

  template <int dim>
  Tensor<1, dim>
  gravity_vector(const Point<dim> &p, const double gravity_constant)
  {
    const double r = p.norm();
    if (r > 1)
      return -gravity_constant * p / r;
    else
      return -gravity_constant * p / std::sqrt(r);
  }


  template <int dim>
  Tensor<1, dim>
  coriolis_vector(const Point<dim> & /*p*/, const double omega)
  {
    Tensor<1, dim> z;

    z[dim - 1] = omega;

    return z;
  }

} // namespace CoreModelData

DYCOREPLANET_CLOSE_NAMESPACE
