#!/bin/bash

files="../out/sq_qsi/run_CQBRW_sw1024_L8_p0.050_Jzz1.000_Jxx0.000_Jyy0.000_merge8.davg.h5 \
../out/sq_qsi/run_CQBRW_sw1024_L8_p0.050_Jzz1.000_Jxx-0.050_Jyy-0.050_merge8.davg.h5 \
../out/sq_qsi/run_CQBRW_sw1024_L8_p0.050_Jzz1.000_Jxx0.050_Jyy0.050_merge8.davg.h5 \
../out/sq_qsi/run_CQBRW_sw1024_L8_p0.050_Jzz1.000_Jxx0.100_Jyy0.100_merge8.davg.h5 \
../out/sq_qsi/run_CQBRW_sw1024_L8_p0.050_Jzz1.000_Jxx0.200_Jyy0.200_merge8.davg.h5"

python3 scripts/plot_heat_capacity.py $files --y_logscale --plot C &
python3 scripts/plot_heat_capacity.py $files --plot S &
