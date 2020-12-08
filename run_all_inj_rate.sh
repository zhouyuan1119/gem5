#!/usr/bin/bash
scons build/NULL/gem5.opt PROTOCOL=Garnet_standalone -j 16
for inj_rate in 0.24 0.26 0.28 0.3 0.32 0.34 0.36
do
  echo ${inj_rate}
  ./build/NULL/gem5.opt --outdir=./m5out_${inj_rate} configs/example/garnet_synth_traffic.py --num-cpus=16 \
  --num-dirs=16 --network=garnet --topology=Mesh_XY --mesh-rows=4 \
  --sim-cycles=1500000000 --synthetic=uniform_random --injectionrate=${inj_rate} \
  --vcs-per-vnet=1 2>&1 &
  # | grep 'Avg packet latency'
done
