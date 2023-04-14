/**
 * @file boussinesq_model_parameters.cc
 * @author Konrad Simon, Rebekka Beddig
 * @version 0.1
 */
#include <model_data/boussinesq_model_parameters.h>

DYCOREPLANET_OPEN_NAMESPACE


CoreModelData::Parameters::Parameters(const std::string &parameter_filename)
  : space_dimension(2)
  , reference_quantities(parameter_filename)
  , physical_constants(parameter_filename)
  , final_time(1.0)
  , time_step(0.1)
  , adapt_time_step(false)
  , initial_global_refinement(2)
  , cuboid_geometry(false)
  , nse_theta(0.5)
  , nse_velocity_degree(2)
  , correct_pressure_to_zero_mean(false)
  , correct_pressure_rhs(false)
  , remove_nullspace(false)
  , only_dirichlet(false)
  , use_picard(false)
  , block00_preconditioner_name("amg")
  , AMG_aggregation_threshold(0.024)
  , use_Mu(false)
  , schur_preconditioner_name("bfbt")
  , bfbt_aggregation_threshold(0.02)
  , only_amg(false)
  , use_ic(false)
  , tolerance(1e-6)
  , precondition_poisson(false)
  , bfbt_scale(false)
  , scale_with_velocity_mass(true)
  , damping_coefficient(0.01)
  , weight(1.0)
  , svd_solver_type("random")
  , rank(15)
  , number(2)
  , use_locally_conservative_discretization(true)
  , solver_diagnostics_print_level(1)
  , use_schur_complement_solver(true)
  , use_direct_solver(false)
  , NSE_solver_interval(1)
  , temperature_theta(0.5)
  , temperature_degree(2)
  , hello_from_cluster(false)
{
  ParameterHandler prm;
  CoreModelData::Parameters::declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename);
  if (!parameter_file)
    {
      parameter_file.close();
      std::ofstream parameter_out(parameter_filename);
      prm.print_parameters(parameter_out, ParameterHandler::Text);
      AssertThrow(false,
                  ExcMessage(
                    "Input parameter file <" + parameter_filename +
                    "> not found. Creating a template file of the same name."));
    }
  prm.parse_input(parameter_file,
                  /* filename = */ "generated_parameter.in",
                  /* last_line = */ "",
                  /* skip_undefined = */ true);
  parse_parameters(prm);
}



void
CoreModelData::Parameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Mesh parameters");
    {
      prm.declare_entry("initial global refinement",
                        "3",
                        Patterns::Integer(0),
                        "The number of global refinement steps performed on "
                        "the initial coarse mesh, before the problem is first "
                        "solved there.");

      prm.declare_entry(
        "cuboid geometry",
        "false",
        Patterns::Bool(),
        "Sets the domain geometry to cuboid. All directions are periodic apart "
        "from the z-direction. This is useful for debugging and later to restrict "
        "global simulations to a full 3D column.");
    }
    prm.leave_subsection();

    prm.declare_entry("space dimension",
                      "2",
                      Patterns::Integer(2, 3),
                      "Spatial dimension of the problem.");

    prm.declare_entry("final time",
                      "1.0",
                      Patterns::Double(0),
                      "The end time of the simulation in seconds.");

    prm.declare_entry("time step",
                      "0.1",
                      Patterns::Double(0),
                      "Time step size.");


    prm.declare_entry("adapt time step",
                      "false",
                      Patterns::Bool(),
                      "Flag to adapt time step by recomputing the CFL number");

    prm.declare_entry("nse theta",
                      "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Theta value for theta method.");

    prm.declare_entry("nse velocity degree",
                      "2",
                      Patterns::Integer(1),
                      "The polynomial degree to use for the velocity variables "
                      "in the NSE system.");

    prm.declare_entry("correct pressure to zero mean",
                      "false",
                      Patterns::Bool(),
                      "Use pressure correction for certain types of BCs.");

    prm.declare_entry("remove nullspace",
                      "false",
                      Patterns::Bool(),
                      "Remove pressure nullspace by fixing a DoF.");

    prm.declare_entry(
      "only dirichlet boundary",
      "false",
      Patterns::Bool(),
      "Use only homogenous Dirichlet boundaries for the velocity.");

    prm.declare_entry(
      "use picard",
      "false",
      Patterns::Bool(),
      "Linearize the advection in the NSE with the Picard iteration.");

    prm.declare_entry("correct pressure rhs",
                      "false",
                      Patterns::Bool(),
                      "Correct pressure rhs (subtract null space).");

    // prm.declare_entry("use AMG",
    //                   "true",
    //                   Patterns::Bool(),
    //                   "Use AMG preconditioner for the upper left block.");
    prm.declare_entry(
      "block00 preconditioner",
      "amg",
      Patterns::Selection("amg|ilu|jacobi"),
      "Choose the preconditioner for the upper left block (amg|jacobi).");

    prm.declare_entry(
      "AMG aggregation threshold",
      "0.02",
      Patterns::Double(),
      "Aggregation threshold for the AMG preconditioner of the upper left block.");

    prm.declare_entry(
      "use Mu",
      "false",
      Patterns::Bool(),
      "Precondition the upper left block with the velocity mass matrix.");

    prm.declare_entry("schur complement preconditioner",
                      "bfbt",
                      Patterns::Selection("bfbt|orig"),
                      "Choose Schur complement preconditioner (orig|bfbt).");

    prm.declare_entry("bfbt aggregation threshold",
                      "0.02",
                      Patterns::Double(1e-4, 0.04),
                      "Aggregation threshold for the AMG in the BFBt method.");

    prm.declare_entry(
      "only amg",
      "false",
      Patterns::Bool(),
      "Use only AMG for Poisson solves in the Schur complement preconditioner.");

    prm.declare_entry(
      "use ic",
      "false",
      Patterns::Bool(),
      "Use IC(0) for Poisson solves in the Schur complement preconditioner.");

    prm.declare_entry(
      "tolerance",
      "1e-6",
      Patterns::Double(1e-12, 1),
      "Relative solver tolerance for Poisson solves in the Schur complement preconditioner.");

    prm.declare_entry(
      "precondition poisson",
      "false",
      Patterns::Bool(),
      "Precondition Poisson-type solver in the Schur complement preconditioner.");

    prm.declare_entry("bfbt_scale",
                      "false",
                      Patterns::Bool(),
                      "Scale BFBt preconditioner.");

    prm.declare_entry("scale_with_velocity_mass",
                      "true",
                      Patterns::Bool(),
                      "Scale BFBt with velocity mass.");

    prm.declare_entry("damping_coefficient",
                      "0.01",
                      Patterns::Double(),
                      "Damp Dirichlet boundary dofs");

    prm.declare_entry("weight",
                      "1.0",
                      Patterns::Double(0.0, 2.0),
                      "Weight the error matrix");

    prm.declare_entry("svd solver",
                      "random",
                      Patterns::Selection("arpack|anasazi|random|arnoldi"),
                      "SVD solver type");

    prm.declare_entry("rank",
                      "15",
                      Patterns::Integer(),
                      "Rank of low-rank update");

    prm.declare_entry("number",
                      "2",
                      Patterns::Integer(),
                      "Number of low-rank updates");

    prm.declare_entry(
      "use locally conservative discretization",
      "true",
      Patterns::Bool(),
      "Whether to use a Navier-Stokes discretization that is locally "
      "conservative at the expense of a larger number of degrees "
      "of freedom, or to go with a cheaper discretization "
      "that does not locally conserve mass (although it is "
      "globally conservative.");

    prm.declare_entry("solver diagnostics level",
                      "1",
                      Patterns::Integer(0),
                      "Output level for solver for debug purposes.");

    prm.declare_entry(
      "use schur complement solver",
      "false",
      Patterns::Bool(),
      "Choose whether to use a preconditioned Schur complement solver or an iterative block system solver with block preconditioner.");

    prm.declare_entry(
      "use direct solver",
      "false",
      Patterns::Bool(),
      "Choose whether to use a direct solver for the Navier-Stokes part.");

    prm.declare_entry("NSE solver interval",
                      "1",
                      Patterns::Integer(1),
                      "Apply the NSE solver only every n-th time step.");

    prm.declare_entry("temperature theta",
                      "0.5",
                      Patterns::Double(0.0, 1.0),
                      "Theta value for theta method.");

    prm.declare_entry(
      "temperature degree",
      "2",
      Patterns::Integer(1),
      "The polynomial degree to use for the temperature variable.");

    prm.declare_entry("filename output",
                      "dycore",
                      Patterns::FileName(),
                      "Base filename for output.");

    prm.declare_entry("dirname output",
                      "data-output",
                      Patterns::FileName(),
                      "Name of output directory.");

    prm.declare_entry(
      "hello from cluster",
      "false",
      Patterns::Bool(),
      "Output some (node) information of each MPI process (rank, node name, number of threads).");
  }
  prm.leave_subsection();
}



void
CoreModelData::Parameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Boussinesq Model");
  {
    prm.enter_subsection("Mesh parameters");
    {
      initial_global_refinement = prm.get_integer("initial global refinement");

      cuboid_geometry = prm.get_bool("cuboid geometry");
    }
    prm.leave_subsection();

    space_dimension = prm.get_integer("space dimension");

    final_time = prm.get_double("final time");
    time_step  = prm.get_double("time step");

    nse_theta           = prm.get_double("nse theta");
    nse_velocity_degree = prm.get_integer("nse velocity degree");

    correct_pressure_to_zero_mean =
      prm.get_bool("correct pressure to zero mean");

    correct_pressure_rhs = prm.get_bool("correct pressure rhs");

    remove_nullspace = prm.get_bool("remove nullspace");

    only_dirichlet = prm.get_bool("only dirichlet boundary");

    use_picard = prm.get_bool("use picard");


    // preconditioners
    block00_preconditioner_name = prm.get("block00 preconditioner");
    AMG_aggregation_threshold   = prm.get_double("AMG aggregation threshold");
    use_Mu                      = prm.get_bool("use Mu");
    schur_preconditioner_name   = prm.get("schur complement preconditioner");
    bfbt_aggregation_threshold  = prm.get_double("bfbt aggregation threshold");
    only_amg                    = prm.get_bool("only amg");
    use_ic                      = prm.get_bool("use ic");
    tolerance                   = prm.get_double("tolerance");
    precondition_poisson        = prm.get_bool("precondition poisson");
    bfbt_scale                  = prm.get_bool("bfbt_scale");
    scale_with_velocity_mass    = prm.get_bool("scale_with_velocity_mass");
    damping_coefficient         = prm.get_double("damping_coefficient");

    weight          = prm.get_double("weight");
    rank            = prm.get_integer("rank");
    number          = prm.get_integer("number");
    svd_solver_type = prm.get("svd solver");

    use_locally_conservative_discretization =
      prm.get_bool("use locally conservative discretization");

    solver_diagnostics_print_level =
      prm.get_integer("solver diagnostics level");
    use_schur_complement_solver = prm.get_bool("use schur complement solver");
    use_direct_solver           = prm.get_bool("use direct solver");

    NSE_solver_interval = prm.get_integer("NSE solver interval");

    temperature_theta  = prm.get_double("temperature theta");
    temperature_degree = prm.get_integer("temperature degree");

    filename_output = prm.get("filename output");
    dirname_output  = prm.get("dirname output");

    hello_from_cluster = prm.get_bool("hello from cluster");
  }
  prm.leave_subsection();
}


DYCOREPLANET_CLOSE_NAMESPACE
