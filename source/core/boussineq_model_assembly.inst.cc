#include <core/boussineq_model_assembly.h>

#include <core/boussineq_model_assembly.tpp>

DYCOREPLANET_OPEN_NAMESPACE

// template instantiations
template class Standard::Assembly::Scratch::NSEPreconditioner<2>;
template class Standard::Assembly::Scratch::NSESystem<2>;
template class Standard::Assembly::Scratch::TemperatureMatrix<2>;
template class Standard::Assembly::Scratch::TemperatureRHS<2>;
template class Standard::Assembly::Scratch::VelocityMass<2>;

template class Standard::Assembly::CopyData::NSEPreconditioner<2>;
template class Standard::Assembly::CopyData::NSESystem<2>;
template class Standard::Assembly::CopyData::TemperatureMatrix<2>;
template class Standard::Assembly::CopyData::TemperatureRHS<2>;
template class Standard::Assembly::CopyData::VelocityMass<2>;

template class Standard::Assembly::Scratch::NSEPreconditioner<3>;
template class Standard::Assembly::Scratch::NSESystem<3>;
template class Standard::Assembly::Scratch::TemperatureMatrix<3>;
template class Standard::Assembly::Scratch::TemperatureRHS<3>;
template class Standard::Assembly::Scratch::VelocityMass<3>;

template class Standard::Assembly::CopyData::NSEPreconditioner<3>;
template class Standard::Assembly::CopyData::NSESystem<3>;
template class Standard::Assembly::CopyData::TemperatureMatrix<3>;
template class Standard::Assembly::CopyData::TemperatureRHS<3>;
template class Standard::Assembly::CopyData::VelocityMass<3>;

DYCOREPLANET_CLOSE_NAMESPACE
