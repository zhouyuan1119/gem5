#!/usr/bin/bash

scons build/NULL/gem5.opt PROTOCOL=Garnet_standalone -j 16
for i in 1 2 3 4 5
do
  ./build/NULL/gem5.opt --outdir=./m5out_${i} configs/example/garnet_synth_traffic.py --num-cpus=16 \
  --num-dirs=16 --network=garnet --topology=Mesh_XY --mesh-rows=4 \
  --sim-cycles=1500000000 --synthetic=uniform_random --injectionrate=0.32 \
  --vcs-per-vnet=1 2>&1 | grep 'Avg packet latency'
done
