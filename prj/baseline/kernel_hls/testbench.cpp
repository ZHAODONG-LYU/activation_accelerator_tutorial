#include "activation_accelerator.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <cstring>
#include <cmath> // For std::abs
#include <chrono>
#include <unistd.h>
#include <string>

// Data loading function
bool load_binary_data(const std::string& filename, uint16* data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    // 直接读取uint16数据
    file.read(reinterpret_cast<char*>(data), DATA_SIZE * sizeof(uint16_t));
    
    file.close();
    return true;
}

// Save binary data
bool save_binary_data(const std::string& filename, uint16* data) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    // 直接保存uint16数据
    file.write(reinterpret_cast<char*>(data), DATA_SIZE * sizeof(uint16_t));
    file.close();
    return true;
}

// Print data statistics
void print_data_stats(uint16* data, const std::string& name) {
    uint16 min_val = data[0];
    uint16 max_val = data[0];
    double sum = 0;
    
    for (int i = 0; i < DATA_SIZE; i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        sum += data[i]; // uint16直接相加
    }
    
    std::cout << name << " statistics: min=" << min_val
              << ", max=" << max_val
              << ", mean=" << sum/DATA_SIZE << std::endl;
}

// Compare results
int compare_results(uint16* result, uint16* golden, uint16* in0, uint16* in1, int32 config) {
    int errors = 0;
    uint16_t max_diff = 0;
    for (int i = 0; i < DATA_SIZE; ++i) {
        if (result[i] != golden[i]) {
            uint16_t diff = std::abs(result[i] - golden[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            errors++;
        }
    }

    
    // 输出最大误差统计
    std::cout << "Max Difference: " << max_diff << std::endl;
    return errors;
}

// A wrapper function to run a single test configuration
bool run_test(int config, uint16* in0, uint16* in1, uint16* out, uint16* golden_data_ptr, const std::string& data_path, uint16* mask_data) {
    std::cout << "\n--- Testing Config " << config << " ---" << std::endl;

    // Select input1 based on config (only config 2 uses mask)
    uint16* current_in1 = (config == 2) ? mask_data : in1;

    // Stage 1: Compute
    int32 current_stage = STAGE_COMPUTE;
    activation_accelerator(in0, current_in1, out, current_stage, config);

    // Stage 2: Store
    current_stage = STAGE_STORE;
    activation_accelerator(in0, current_in1, out, current_stage, config);

    // Compare results
    int errors = compare_results(out, golden_data_ptr, in0, current_in1, config);

    // Save HLS output
    std::string output_filename = data_path + "hls_output_config_" + std::to_string(config) + ".bin";
    save_binary_data(output_filename, out);
    std::cout << "Output results saved to: " << output_filename << std::endl;

    return errors == 0;
}

std::string get_data_path() {
    std::string rel_path = "../../data/";
    std::string test_file = rel_path + "in0_bf16.bin";
    std::ifstream f(test_file.c_str());
    if (f.good()) {
        std::cout << "Using relative data path: " << rel_path << std::endl;
        return rel_path;
    } else {
        std::cerr << "ERROR: Cannot find data file at relative path: " << test_file << std::endl;
        exit(1);
    }
}

int main() {
    // Allocate memory
    uint16* in0 = new uint16[DATA_SIZE];
    uint16* in1 = new uint16[DATA_SIZE];
    uint16* out = new uint16[DATA_SIZE];
    uint16* mask = new uint16[DATA_SIZE];
    uint16* golden_data[7];
    for (int i = 0; i < 7; ++i) golden_data[i] = new uint16[DATA_SIZE];

    // Load test data
    std::cout << "Loading test data..." << std::endl;
    std::string data_path = get_data_path();
    if (!load_binary_data(data_path + "in0_bf16.bin", in0) ||
        !load_binary_data(data_path + "in1_bf16.bin", in1) ||
        !load_binary_data(data_path + "mask_bf16.bin", mask)) {
        std::cerr << "Unable to load bf16 input data, using random data" << std::endl;
        srand(time(NULL));
        for(int i = 0; i < DATA_SIZE; i++) {
            in0[i] = rand() % 1000 - 500;
            in1[i] = rand() % 1000 - 500;
            mask[i] = rand() % 2;
        }
    } else {
        std::cout << "BF16 data loaded successfully" << std::endl;
    }
    print_data_stats(in0, "in0");
    print_data_stats(in1, "in1");
    print_data_stats(mask, "mask");
    std::cout << "=== HLS Activation Accelerator Testbench ===" << std::endl;

    // Load all golden results
    for (int i = 0; i < 7; ++i) {
        std::string golden_file = data_path + "golden_out_config_" + std::to_string(i) + "_bf16.bin";
        std::cout << "Trying to load: " << golden_file << std::endl;
        if (!load_binary_data(golden_file, golden_data[i])) {
            std::cerr << "Unable to load Config " << i << " golden data" << std::endl;
            for(int j = 0; j < DATA_SIZE; j++) golden_data[i][j] = rand() % 1000 - 500;
            std::cout << "Using random data as golden reference for Config " << i << std::endl;
        } else {
            std::cout << "Loaded Config " << i << " golden data from bf16" << std::endl;
        }
    }

    // Stage 0: Data transfer
    int32 stage = STAGE_LOAD;
    std::cout << "Set activation_accelerator.stage = 0" << std::endl;
    activation_accelerator(in0, in1, out, stage, 0);
    std::cout << "Data transfer complete" << std::endl;

    // Main loop for all configs
    double total_time = 0.0;
    for (int config = 0; config < 7; ++config) {
        // Stage 1: Compute
        stage = STAGE_COMPUTE;
        std::cout << "\n--- Testing Config " << config << " ---" << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        activation_accelerator(in0, in1, out, stage, config);
        auto t2 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_time += elapsed;
        std::cout << "Config " << config << " compute time: " << elapsed << " ms" << std::endl;

        // Stage 2: Store
        stage = STAGE_STORE;
        activation_accelerator(in0, in1, out, stage, config);
        std::cout << "Output results saved to: " << data_path << "hls_output_config_" << config << ".bin" << std::endl;
        save_binary_data(data_path + "hls_output_config_" + std::to_string(config) + ".bin", out);

        // Compare results
        int errors = compare_results(out, golden_data[config], in0, in1, config);
        if (errors == 0) {
            std::cout << "Config " << config << " test passed!" << std::endl;
        } else {
            std::cout << "Config " << config << " test failed with " << errors << " errors" << std::endl;
        }
    }
    std::cout << "\nTotal compute time for all configs: " << total_time << " ms" << std::endl;

    // Clean up
    for (int i = 0; i < 7; ++i) delete[] golden_data[i];
    delete[] in0;
    delete[] in1;
    delete[] out;
    delete[] mask;
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}