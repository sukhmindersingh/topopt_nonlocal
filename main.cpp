#include <filesystem>
#include <string>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>

#include "Teuchos_StandardCatchMacros.hpp"

#include "nonlocal.hpp"

int main(int argc, char **argv) {

  dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);

  bool success = true;

  try {

    auto params = Teuchos::getParametersFromYamlFile(argv[1]);

    std::string output_dir = params->get<std::string>("output directory name");

    std::filesystem::path input_file(argv[1]);

    if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      if (!std::filesystem::is_directory(output_dir) ||
          !std::filesystem::exists(output_dir)) {
        std::filesystem::create_directory(output_dir);
      }
      std::filesystem::copy_file(
          input_file, output_dir + "/input.yaml",
          std::filesystem::copy_options::overwrite_existing);
    }

    Nonlocal nonlocal_problem(params);
    nonlocal_problem.run();
  }

  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

  return 0;
}
