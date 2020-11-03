#!/usr/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: ./run_all_inj_rate.sh [OUTPUT LOG FILE]"
    exit 1
fi

scons build/NULL/gem5.opt PROTOCOL=Garnet_standalone -j 16
rm -rf ${1}
# for inj_rate in 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
for inj_rate in 0.12 0.14 0.16 0.18
do
  # for i in 1 2 3 4 5 6 7 8 9 10
  # do
    echo ${inj_rate}
    ./build/NULL/gem5.opt configs/example/garnet_synth_traffic.py --num-cpus=64 \
    --num-dirs=64 --network=garnet --topology=Mesh_XY --mesh-rows=8 \
    --sim-cycles=1000000000 --synthetic=uniform_random --injectionrate=${inj_rate} \
    --vcs-per-vnet=1 >> ${1} 2>&1
  # done
done
