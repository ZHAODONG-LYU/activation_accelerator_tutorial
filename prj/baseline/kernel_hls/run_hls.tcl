#Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# Before you run this script, please activate vitis_hls environment

set project_name "activation_accelerator"
set opt_method "baseline"

# Create a project
open_project -reset ${project_name}

# Add design files
add_files ${project_name}.cpp
add_files ${project_name}.h
# Add test bench & files
add_files -tb testbench.cpp

# Set the top-level function
set_top activation_accelerator

# ########################################################
# Create a solution
open_solution -reset ${opt_method} -flow_target vivado

# Define technology and clock rate
set_part  {xck26-sfvc784-2LV-c}
create_clock -period 10

# Set variable to select which steps to execute
set hls_exec 3


csim_design
# Set any optimization directives
# End of directives

if {$hls_exec >= 1} {
	# Run Synthesis
   csynth_design
}
if {$hls_exec >= 2} {
	# Run Synthesis, RTL Simulation
   cosim_design
}
if {$hls_exec >= 3} { 
	# Run Synthesis, RTL Simulation, RTL implementation
   #export_design -format ip_catalog -version "1.00a" -library "hls" -vendor "xilinx.com" -description "A memory mapped IP created by Vitis HLS" -evaluate verilog
   export_design -format ip_catalog -evaluate verilog
}

exit
