# Tổng hợp thiết kế cho AMD Spartan-7 XC7S25CSGA225-1 và xuất báo cáo bắt buộc.
#
# Cách chạy (từ thư mục gốc repo, Vivado 2025.2):
#   vivado -mode batch -source vivado/scripts/synth_xc7s25.tcl -tclargs <top> [rtl_dir] [xdc_file]
#
#   <top>      tên module top (bắt buộc)
#   [rtl_dir]  thư mục chứa .v/.sv, mặc định: rtl   (tìm cả thư mục con)
#   [xdc_file] file ràng buộc, mặc định: vivado/constraints/top.xdc (bỏ qua nếu không có)
#
# Kết quả: reports/<top>_utilization.rpt, reports/<top>_timing_summary.rpt,
#          reports/<top>_utilization_hier.rpt, vivado/build/<top>_synth.dcp

set part xc7s25csga225-1

if {[llength $argv] < 1} {
    puts "ERROR: missing top module name. Example: -tclargs cnn_top"
    exit 1
}
set top      [lindex $argv 0]
set rtl_dir  [expr {[llength $argv] > 1 ? [lindex $argv 1] : "rtl"}]
set xdc_file [expr {[llength $argv] > 2 ? [lindex $argv 2] : "vivado/constraints/top.xdc"}]

proc find_files {dir patterns} {
    set result {}
    foreach pat $patterns {
        foreach f [glob -nocomplain -directory $dir -type f $pat] {
            lappend result $f
        }
    }
    foreach sub [glob -nocomplain -directory $dir -type d *] {
        set result [concat $result [find_files $sub $patterns]]
    }
    return $result
}

set v_files  [find_files $rtl_dir {*.v}]
set sv_files [find_files $rtl_dir {*.sv}]
set mem_files [find_files $rtl_dir {*.mem *.hex}]

if {[llength $v_files] + [llength $sv_files] == 0} {
    puts "ERROR: no .v/.sv files found in $rtl_dir"
    exit 1
}

file mkdir reports
file mkdir vivado/build

if {[llength $v_files]}  { read_verilog $v_files }
if {[llength $sv_files]} { read_verilog -sv $sv_files }
foreach m $mem_files { add_files -norecurse $m }

if {[file exists $xdc_file]} {
    read_xdc $xdc_file
    puts "INFO: using constraints $xdc_file"
} else {
    puts "WARNING: $xdc_file not found, timing report will have no clock constraint"
}

synth_design -top $top -part $part

report_utilization -file reports/${top}_utilization.rpt
report_utilization -hierarchical -file reports/${top}_utilization_hier.rpt
report_timing_summary -file reports/${top}_timing_summary.rpt
write_checkpoint -force vivado/build/${top}_synth.dcp

puts "INFO: done. Reports are in reports/"
