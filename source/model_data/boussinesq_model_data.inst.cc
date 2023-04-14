#include <model_data/boussinesq_model_data.h>

#include <model_data/boussinesq_model_data.tpp>

DYCOREPLANET_OPEN_NAMESPACE

// Extern template instantiations
template class CoreModelData::Boussinesq::VelocityInitialValues<2>;
template class CoreModelData::Boussinesq::VelocityInitialValues<3>;

template class CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<2>;
template class CoreModelData::Boussinesq::TemperatureInitialValuesCuboid<3>;

template class CoreModelData::Boussinesq::TemperatureInitialValues<2>;
template class CoreModelData::Boussinesq::TemperatureInitialValues<3>;

template class CoreModelData::Boussinesq::TemperatureRHS<2>;
template class CoreModelData::Boussinesq::TemperatureRHS<3>;

DYCOREPLANET_CLOSE_NAMESPACE
