#!/usr/bin/bash

scons build/NULL/gem5.opt PROTOCOL=Garnet_standalone -j 16
rm -rf log
for i in 1 2 3 4 5 6 7 8 9 10
do
  ./build/NULL/gem5.opt configs/example/garnet_synth_traffic.py --num-cpus=16 \
  --num-dirs=16 --network=garnet --topology=Mesh_XY --mesh-rows=4 \
  --sim-cycles=500000 --synthetic=uniform_random --injectionrate=0.38 \
  --vcs-per-vnet=1 >> log
done
grep -i "avg" ./log
