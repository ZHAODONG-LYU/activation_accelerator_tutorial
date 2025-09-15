#******************************************************************************
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#******************************************************************************
################################################################


################################################################
# This is a generated script based on design: activation_accelerator
#
# Though there are limitations about the generated script,
# the main purpose of this utility is to make learning
# IP Integrator Tcl commands easier.
################################################################


set project_name "activation_accelerator"
set opt_method "baseline"
set design_name "design_1"

#Clean up
file delete {*}[glob *.log]
file delete {*}[glob *.jou]
file delete -force ./${project_name}
file delete -force .crashReporter
file delete -force .Xil

create_project ${project_name} ./${project_name} -part xck26-sfvc784-2LV-c -force
set_property board_part xilinx.com:kv260_som:part0:1.4 [current_project]

# You should revise the path for your own export
set_property  ip_repo_paths  ../kernel_hls/${project_name}/${opt_method}/impl/ip [current_project]

update_ip_catalog

create_bd_design ${design_name}

# Create Zynq UltraScale+ PS
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.4 zynq_ultra_ps_e_0
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]

# 启用PL到PS中断
set_property -dict [list CONFIG.PSU__ENABLE__IRQ0 {1}] [get_bd_cells zynq_ultra_ps_e_0]
# Create reset controller
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0

# Create activation accelerator IP
create_bd_cell -type ip -vlnv xilinx.com:hls:activation_accelerator:1.0 activation_accelerat_0

# Create SmartConnect for control interface
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_0

# Create SmartConnect for memory interfaces
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smartconnect_1

# Configure PS after all IPs are created (based on GUI export)
set_property -dict [list \
  CONFIG.PSU__USE__IRQ0 {1} \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {0} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
] [get_bd_cells zynq_ultra_ps_e_0]

# Configure SmartConnect
set_property CONFIG.NUM_SI {1} [get_bd_cells smartconnect_0]
set_property CONFIG.NUM_SI {3} [get_bd_cells smartconnect_1]

# Connect memory interfaces to PS
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/M00_AXI] [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]

# Connect control interface (use LPD for low power)
connect_bd_intf_net [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM0_FPD] [get_bd_intf_pins smartconnect_0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins activation_accelerat_0/s_axi_control] [get_bd_intf_pins smartconnect_0/M00_AXI]

# Connect memory interfaces to SmartConnect
connect_bd_intf_net [get_bd_intf_pins activation_accelerat_0/m_axi_gmem0] [get_bd_intf_pins smartconnect_1/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smartconnect_1/S01_AXI] [get_bd_intf_pins activation_accelerat_0/m_axi_gmem1]
connect_bd_intf_net [get_bd_intf_pins activation_accelerat_0/m_axi_gmem2] [get_bd_intf_pins smartconnect_1/S02_AXI]

# Connect clocks
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_0/aclk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins activation_accelerat_0/ap_clk]
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins smartconnect_1/aclk]

# Apply automation for reset
apply_bd_automation -rule xilinx.com:bd_rule:board -config { Manual_Source {Auto}}  [get_bd_pins proc_sys_reset_0/ext_reset_in]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins proc_sys_reset_0/slowest_sync_clk]
apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config { Clk {/zynq_ultra_ps_e_0/pl_clk0 (99 MHz)} Freq {100} Ref_Clk0 {} Ref_Clk1 {} Ref_Clk2 {}}  [get_bd_pins zynq_ultra_ps_e_0/maxihpm0_fpd_aclk]
# Connect HP0 clock
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_clk0] [get_bd_pins zynq_ultra_ps_e_0/saxihp0_fpd_aclk]

# Connect interrupt (skip for now - pin may not exist)
connect_bd_net [get_bd_pins zynq_ultra_ps_e_0/pl_ps_irq0] [get_bd_pins activation_accelerat_0/interrupt]

# Connect resets
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_0/aresetn]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins activation_accelerat_0/ap_rst_n]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_pins smartconnect_1/aresetn]

regenerate_bd_layout

# Assign address spaces (based on GUI export)
assign_bd_address -offset 0x000800000000 -range 0x000800000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem0] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_HIGH] -force
assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem0] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force
assign_bd_address -offset 0xC0000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem0] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_QSPI] -force
assign_bd_address -offset 0x000800000000 -range 0x000800000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem1] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_HIGH] -force
assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem1] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force
assign_bd_address -offset 0xC0000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem1] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_QSPI] -force
assign_bd_address -offset 0x000800000000 -range 0x000800000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem2] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_HIGH] -force
assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem2] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_DDR_LOW] -force
assign_bd_address -offset 0xC0000000 -range 0x20000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem2] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_QSPI] -force
assign_bd_address -offset 0xA0000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces zynq_ultra_ps_e_0/Data] [get_bd_addr_segs activation_accelerat_0/s_axi_control/Reg] -force

# Exclude Address Segments
exclude_bd_addr_seg -offset 0xFF000000 -range 0x01000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem0] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_LPS_OCM]
exclude_bd_addr_seg -offset 0xFF000000 -range 0x01000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem1] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_LPS_OCM]
exclude_bd_addr_seg -offset 0xFF000000 -range 0x01000000 -target_address_space [get_bd_addr_spaces activation_accelerat_0/Data_m_axi_gmem2] [get_bd_addr_segs zynq_ultra_ps_e_0/SAXIGP2/HP0_LPS_OCM]

validate_bd_design

generate_target all [get_files  ./${project_name}/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd]
catch { config_ip_cache -export [get_ips -all ${design_name}_zynq_ultra_ps_e_0_0] }
catch { config_ip_cache -export [get_ips -all ${design_name}_activation_accelerat_0_0] }
catch { config_ip_cache -export [get_ips -all ${design_name}_auto_pc_0] }
catch { config_ip_cache -export [get_ips -all ${design_name}_rst_ps8_0_100M_0] }
catch { config_ip_cache -export [get_ips -all ${design_name}_auto_pc_1] }
catch { config_ip_cache -export [get_ips -all ${design_name}_auto_us_0] }
export_ip_user_files -of_objects [get_files ./${project_name}/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -no_script -sync -force -quiet
create_ip_run [get_files -of_objects [get_fileset sources_1] ./${project_name}/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd]
launch_runs ${design_name}_zynq_ultra_ps_e_0_0_synth_1 ${design_name}_activation_accelerat_0_0_synth_1 ${design_name}_auto_pc_0_synth_1 ${design_name}_rst_ps8_0_100M_0_synth_1 ${design_name}_auto_pc_1_synth_1 ${design_name}_auto_us_0_synth_1 -jobs 8
export_simulation -of_objects [get_files ./${project_name}/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -directory ./${project_name}/${project_name}.ip_user_files/sim_scripts -ip_user_files_dir ./${project_name}/${project_name}.ip_user_files -ipstatic_source_dir ./${project_name}/${project_name}.ip_user_files/ipstatic -lib_map_path [list {modelsim=./${project_name}/${project_name}.cache/compile_simlib/modelsim} {questa=./${project_name}/${project_name}.cache/compile_simlib/questa} {riviera=./${project_name}/${project_name}.cache/compile_simlib/riviera} {activehdl=./${project_name}/${project_name}.cache/compile_simlib/activehdl}] -use_ip_compiled_libs -force -quiet

make_wrapper -files [get_files ./${project_name}/${project_name}.srcs/sources_1/bd/${design_name}/${design_name}.bd] -top
add_files -norecurse ./${project_name}/${project_name}.gen/sources_1/bd/${design_name}/hdl/${design_name}_wrapper.v

launch_runs synth_1 -jobs 8
wait_on_run synth_1

launch_runs impl_1 -jobs 8
wait_on_run impl_1

launch_runs impl_1 -to_step write_bitstream -jobs 8 
wait_on_run impl_1

#move and rename bitstream to final location
file copy -force ./${project_name}/${project_name}.runs/impl_1/${design_name}_wrapper.bit ${project_name}.bit

file copy -force ./${project_name}/${project_name}.gen/sources_1/bd/${design_name}/hw_handoff/${design_name}.hwh ${project_name}.hwh