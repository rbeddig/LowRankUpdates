/**
 * @file boussinesq_model_data.tpp
 * @author Konrad Simon
 * @version 0.1
 */

#include <model_data/boussinesq_model_data.h>

DYCOREPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  namespace Boussinesq
  {
    //////////////////////////////////////////////////
    /// Initial temperature
    //////////////////////////////////////////////////

    template <>
    TemperatureInitialValues<2>::TemperatureInitialValues(const double R0,
                                                          const double R1)
      : Function<2>(1)
      , rotate(true)
      , alpha_(numbers::PI / 3)
      , beta_(0)
      , gamma_(0)
    {
      rotation = 0;

      if (rotate)
        {
          rotation[0][0] = cos(alpha_);
          rotation[0][1] = -sin(alpha_);
          rotation[1][0] = sin(alpha_);
          rotation[1][1] = cos(alpha_);
        }

      covariance_matrix = 0;

      for (unsigned int d = 0; d < 2; ++d)
        {
          covariance_matrix[d][d] = 20 / ((R1 - R0) / 2);
        }

      if (rotate)
        {
          Tensor<1, 2> center_tmp_1, center_tmp_2;

          center_tmp_1[0] = R0 + (R1 - R0) * 0.35;
          center_tmp_2[1] = R0 + (R1 - R0) * 0.65;

          /*
           * At this point centerX is still the origin
           */
          center1 += rotation * center_tmp_1 * transpose(rotation);
          center2 += rotation * center_tmp_2 * transpose(rotation);
        }
      else
        {
          center1(0) = R0 + (R1 - R0) * 0.35;
          center2(1) = R0 + (R1 - R0) * 0.65;
        }
    }

    template <>
    TemperatureInitialValues<3>::TemperatureInitialValues(const double R0,
                                                          const double R1)
      : Function<3>(1)
      , rotate(false)
      , alpha_(numbers::PI / 3)
      , beta_(numbers::PI / 6)
      , gamma_(numbers::PI / 4)
    {
      /*
       * Describe a 3D rotation with Euler angles
       */
      if (rotate)
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

      covariance_matrix = 0;

      for (unsigned int d = 0; d < 3; ++d)
        {
          covariance_matrix[d][d] = 20 / ((R1 - R0) / 2);
        }

      if (rotate)
        {
          Tensor<1, 3> center_tmp_1, center_tmp_2;

          center_tmp_1[0] = R0 + (R1 - R0) * 0.35;
          center_tmp_2[1] = R0 + (R1 - R0) * 0.65;

          /*
           * At this point centerX is still the origin
           */
          center1 += rotation * center_tmp_1 * transpose(rotation);
          center2 += rotation * center_tmp_2 * transpose(rotation);
        }
      else
        {
          center1(0) = R0 + (R1 - R0) * 0.35;
          center2(1) = R0 + (R1 - R0) * 0.65;
        }
    }


    template <int dim>
    double
    TemperatureInitialValues<dim>::value(const Point<dim> &p,
                                         const unsigned int) const
    {
      //   const double r = p.norm();
      //   const double h = R1 - R0;
      //   const double s = (r - R0) / h;
      //   const double q =
      //    (dim == 2) ? 1.0 : std::max(0.0, cos(numbers::PI * abs(p(2) / R1)));
      //   const double phi = std::atan2(p(0), p(1));
      //   const double tau = s + s * (1 - s) * sin(6 * phi) * q;
      //
      //   return reference_temperature_bottom * (1.0 - tau) +
      //         reference_temperature_top * tau;

      double temperature =
        sqrt(determinant(covariance_matrix)) *
          exp(-0.5 *
              scalar_product(p - center1, covariance_matrix * (p - center1))) /
          sqrt(std::pow(2 * numbers::PI, dim)) +
        sqrt(determinant(covariance_matrix)) *
          exp(-0.5 *
              scalar_product(p - center2, covariance_matrix * (p - center2))) /
          sqrt(std::pow(2 * numbers::PI, dim));

      return temperature;
    }



    template <int dim>
    void
    TemperatureInitialValues<dim>::value_list(
      const std::vector<Point<dim>> &points,
      std::vector<double> &          values,
      const unsigned int) const
    {
      Assert(points.size() == values.size(),
             ExcDimensionMismatch(points.size(), values.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        {
          values[p] = value(points[p]);
        }
    }


    template <int dim>
    TemperatureInitialValuesCuboid<dim>::TemperatureInitialValuesCuboid(
      const Point<dim> center,
      const double     diameter)
      : Function<dim>(1)
      , center(center)
    {
      covariance_matrix = 0;

      for (unsigned int d = 0; d < dim; ++d)
        {
          covariance_matrix[d][d] = 1 / (std::pow(diameter * 0.1, 2));
        }
    }


    template <int dim>
    double
    TemperatureInitialValuesCuboid<dim>::value(const Point<dim> &p,
                                               const unsigned int) const
    {
      double temperature =
        sqrt(determinant(covariance_matrix)) *
        exp(-0.5 *
            scalar_product(p - center, covariance_matrix * (p - center))) /
        (2 * sqrt(std::pow(2 * numbers::PI, 2)));

      return temperature;
    }


    template <int dim>
    void
    TemperatureInitialValuesCuboid<dim>::value_list(
      const std::vector<Point<dim>> &points,
      std::vector<double> &          values,
      const unsigned int) const
    {
      Assert(points.size() == values.size(),
             ExcDimensionMismatch(points.size(), values.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        {
          values[p] = value(points[p]);
        }
    }

    //////////////////////////////////////////////////
    /// Temperature RHS
    //////////////////////////////////////////////////

    template <int dim>
    double
    TemperatureRHS<dim>::value(const Point<dim> &p, const unsigned int) const
    {
      std::ignore = p;
      return 0;
    }


    template <int dim>
    void
    TemperatureRHS<dim>::value_list(const std::vector<Point<dim>> &points,
                                    std::vector<double> &          values,
                                    const unsigned int) const
    {
      Assert(points.size() == values.size(),
             ExcDimensionMismatch(points.size(), values.size()));

      for (unsigned int p = 0; p < points.size(); ++p)
        {
          values[p] = value(points[p]);
        }
    }


    //////////////////////////////////////////////////
    /// Initial velocity
    //////////////////////////////////////////////////

    template <int dim>
    Tensor<1, dim>
    VelocityInitialValues<dim>::value(const Point<dim> &) const
    {
      // This initializes to zero.
      Tensor<1, dim> value;

      return value;
    }



    template <int dim>
    void
    VelocityInitialValues<dim>::value_list(
      const std::vector<Point<dim>> &points,
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


  } // namespace Boussinesq

} // namespace CoreModelData

DYCOREPLANET_CLOSE_NAMESPACE
