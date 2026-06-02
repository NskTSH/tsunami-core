//#include "SingleLineGridSimulationSubgridCUDA.h" // Include the new simulation class header
//#include "ASCParser.h"                  // Include your specific parser implementation
//#include "IBaseBathymetryParser.h"      // Include the base parser interface
//#include <iostream>
//#include <string>
//#include <chrono>       // For timing
//#include <memory>       // For std::unique_ptr (safer resource management)
//#include <stdexcept>    // For exception handling
//
//int main(int argc, char* argv[]) {
//
//    // --- Configuration ---
//    // Determine which scenario to run (e.g., Test1 or from parser)
//    // You could use command-line arguments, config files, or hardcode it.
//    bool use_parser_scenario = true; // Example: Set to true to use parser data
//
//    // Parameters needed if using the parser scenario
//    std::string input_bathymetry_file = "../Data/JapanSea.asc"; // Example: path relative to repo (see $NSKTSH_DATA)
//    double gravity = 9.81;
//    double time_step_duration = 0.5; // ** IMPORTANT: Adjust based on stability (CFL condition)! **
//    int total_timesteps = 18000;      // Example number of steps
//    double min_depth_for_calc = 5; // Minimum depth to perform calculations
//
//    // Parameters applicable to both scenarios
//    int number_of_subgrids = 1; // ** Choose based on grid size and GPU memory **
//    int results_save_interval = 10; // Save results every 50 timesteps
//
//
//    // --- Parser Setup (if using parser) ---
//    std::unique_ptr<IBaseBathymetryParser> bathymetry_parser = nullptr;
//    if (use_parser_scenario) {
//        try {
//            // Use a smart pointer for automatic memory management
//            bathymetry_parser = std::make_unique<ASCParser>();
//            std::cout << "Reading bathymetry data from: " << input_bathymetry_file << std::endl;
//            bathymetry_parser->Read(input_bathymetry_file);
//            std::cout << "Bathymetry loaded ("
//                << bathymetry_parser->GetCols() << "x" << bathymetry_parser->GetRows()
//                << ")" << std::endl;
//        }
//        catch (const std::exception& e) {
//            std::cerr << "Error reading bathymetry file: " << e.what() << std::endl;
//            return 1; // Exit if parser fails
//        }
//    }
//
//    // --- Create Simulation Object ---
//    // Uses RAII (destructor will handle cleanup of subgrids and shared CUDA memory)
//    SingleLineGridSimulationSubgridCUDA simulation;
//
//    try {
//        std::cout << "Initializing Simulation..." << std::endl;
//        // --- Initialize Simulation ---
//        if (use_parser_scenario && bathymetry_parser) {
//            // Initialize using the data read from the parser
//            simulation.InitializeFromParser(
//                *bathymetry_parser,      // Pass the parser object (dereferenced pointer)
//                gravity,                 // Gravitational acceleration
//                time_step_duration,      // Time step duration (tau)
//                total_timesteps,         // Total number of timesteps to run
//                number_of_subgrids,      // How many chunks to divide the grid into
//                results_save_interval,   // Frequency of saving results
//                min_depth_for_calc       // Minimum depth threshold
//            );
//            std::cout << "Simulation initialized using parser data." << std::endl;
//        }
//        else {
//            // Initialize using the built-in Test1 scenario
//            // Test1 defines its own grid size, parameters, and initial conditions internally
//            simulation.SetParamsTest1(
//                number_of_subgrids       // Specify how many subgrids for Test1
//            );
//            // Optional: Override save interval or timesteps if needed for Test1
//            // total_timesteps = simulation.GetTimesteps(); // Get timesteps set by Test1
//            // results_save_interval = 50; // Set custom save interval
//            std::cout << "Simulation initialized using Test1 parameters." << std::endl;
//        }
//
//
//        // --- Run Simulation and Measure Time ---
//        std::cout << "\nStarting simulation run..." << std::endl;
//        auto start_time = std::chrono::high_resolution_clock::now();
//
//        simulation.Run(); // Execute the main simulation loop
//
//        auto end_time = std::chrono::high_resolution_clock::now();
//        std::cout << "\nSimulation run finished." << std::endl;
//
//        // --- Calculate and Print Elapsed Time ---
//        std::chrono::duration<double> elapsed = end_time - start_time;
//        std::cout << std::endl
//            << "SingleLineGridSimulationSubgridCUDA (" << number_of_subgrids << " subgrids) completed in "
//            << elapsed.count() << " seconds." << std::endl;
//
//    }
//    catch (const std::exception& e) {
//        // Catch standard exceptions (including runtime_error from CUDA checks)
//        std::cerr << "\n--- Simulation Error ---" << std::endl;
//        std::cerr << e.what() << std::endl;
//        // Consider adding cudaDeviceReset() here if errors might leave GPU in bad state
//        // #include <cuda_runtime.h> // Need this header for cudaDeviceReset
//        // cudaDeviceReset();
//        return 1; // Indicate failure
//    }
//
//    return 0; // Indicate success
//}