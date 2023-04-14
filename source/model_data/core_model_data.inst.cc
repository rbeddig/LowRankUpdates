#include <model_data/core_model_data.h>

#include <model_data/core_model_data.tpp>

DYCOREPLANET_OPEN_NAMESPACE

template class CoreModelData::TangentialFunction<2>;
template class CoreModelData::TangentialFunction<3>;

template class CoreModelData::RadialFunction<2>;
template class CoreModelData::RadialFunction<3>;

template Tensor<1, 2>
CoreModelData::vertical_gravity_vector<2>(const Point<2> &p,
                                          const double    gravity_constant);
template Tensor<1, 3>
CoreModelData::vertical_gravity_vector<3>(const Point<3> &p,
                                          const double    gravity_constant);

template Tensor<1, 2>
CoreModelData::gravity_vector<2>(const Point<2> &p,
                                 const double    gravity_constant);
template Tensor<1, 3>
CoreModelData::gravity_vector<3>(const Point<3> &p,
                                 const double    gravity_constant);

template Tensor<1, 2>
CoreModelData::coriolis_vector(const Point<2> & /*p*/, const double omega);

template Tensor<1, 3>
CoreModelData::coriolis_vector(const Point<3> & /*p*/, const double omega);

DYCOREPLANET_CLOSE_NAMESPACE
