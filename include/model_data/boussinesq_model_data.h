/**
 * @file boussinesq_model_data.h
 * @author Konrad Simon
 * @version 0.1
 */
#pragma once

// C++ STL
#include <cmath>
#include <string>

// Deal.ii
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/vector.h>

// AquaPlanet
#include <base/config.h>
#include <model_data/core_model_data.h>


DYCOREPLANET_OPEN_NAMESPACE

namespace CoreModelData
{
  /*!
   * @namespace Boussinesq
   *
   * Namespace containing constants and helper functions relevant to a transient
   * Boussinesq model.
   */
  namespace Boussinesq
  {
    //////////////////////////////////////////////////
    /// Initial velocity
    //////////////////////////////////////////////////

    /*!
     * Velocity initial values for rising warm bubble test.
     */
    template <int dim>
    class VelocityInitialValues : public TensorFunction<1, dim>
    {
    public:
      /*!
       * Constructor.
       */
      VelocityInitialValues()
        : TensorFunction<1, dim>()
      {}

      /*!
       * Return velocity value at a single point.
       *
       * @param p
       * @return
       */
      virtual Tensor<1, dim>
      value(const Point<dim> &p) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<1, dim>> &  values) const override;
    };

    //////////////////////////////////////////////////
    /// Initial temperature
    //////////////////////////////////////////////////

    /*!
     * Temerature initial values for rising warm bubble test.
     */
    template <int dim>
    class TemperatureInitialValues : public Function<dim>
    {
    public:
      /*!
       * Constructor.
       */
      TemperatureInitialValues(const double R0, const double R1);


      /*!
       * Return temperature value at a single point.
       *
       * @param p
       * @param component
       * @return
       */
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double> &          values,
                 const unsigned int             component = 0) const override;

    private:
      Point<dim>     center1, center2;
      Tensor<2, dim> covariance_matrix;

      const bool rotate;

      /*!
       * Euler angles
       */
      const double alpha_, beta_, gamma_;

      /*!
       * Rotation tensor.
       */
      Tensor<2, dim> rotation;
    };


    /*!
     * Temerature initial values for rising warm bubble test in cuboid geometry.
     */
    template <int dim>
    class TemperatureInitialValuesCuboid : public Function<dim>
    {
    public:
      /*!
       * Constructor.
       */
      TemperatureInitialValuesCuboid(const Point<dim> center,
                                     const double     diameter);


      /*!
       * Return temperature value at a single point.
       *
       * @param p
       * @param component
       * @return
       */
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double> &          values,
                 const unsigned int             component = 0) const override;

    private:
      Point<dim>     center;
      Tensor<2, dim> covariance_matrix;
    };

    //////////////////////////////////////////////////
    /// Temperature RHS
    //////////////////////////////////////////////////

    /*!
     * Temerature right-hand side for rising warm bubble test. This term
     * represents external heat sources.
     */
    template <int dim>
    class TemperatureRHS : public Function<dim>
    {
    public:
      /*!
       * Constructor.
       */
      TemperatureRHS()
        : Function<dim>(1)
      {}

      /*!
       * Return temperature value at a single point.
       *
       * @param p
       * @param component
       * @return
       */
      virtual double
      value(const Point<dim> & p,
            const unsigned int component = 0) const override;

      /*!
       * Return temperature value as a vector at a single point.
       *
       * @param points
       * @param values
       */
      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<double> &          value,
                 const unsigned int             component = 0) const override;
    };

  } // namespace Boussinesq

} // namespace CoreModelData

/*
 * Forward declarations (necessary due to subsequent specializations)
 */
template <>
CoreModelData::Boussinesq::TemperatureInitialValues<
  2>::TemperatureInitialValues(const double R0, const double R1);
template <>
CoreModelData::Boussinesq::TemperatureInitialValues<
  3>::TemperatureInitialValues(const double R0, const double R1);

/*
 * Extern template instantiations
 */
extern template class CoreModelData::Boussinesq::VelocityInitialValues<2>;
extern template class CoreModelData::Boussinesq::VelocityInitialValues<3>;

extern template class CoreModelData::Boussinesq::TemperatureInitialValues<2>;
extern template class CoreModelData::Boussinesq::TemperatureInitialValues<3>;

extern template class CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<
  2>;
extern template class CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<
  3>;

extern template class CoreModelData::Boussinesq::TemperatureRHS<2>;
extern template class CoreModelData::Boussinesq::TemperatureRHS<3>;

DYCOREPLANET_CLOSE_NAMESPACE
