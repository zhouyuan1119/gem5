import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_path', type=str)

flags, unparsed = parser.parse_known_args()

policies = ['fifo', 'logic', 'ga', 'rr']
# inj_rates = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
inj_rates = [0.12, 0.14, 0.16, 0.18]
num_runs = 1

all_data = np.zeros((len(inj_rates), num_runs, len(policies)*3), 
                    dtype = np.float32)
for idx, p in enumerate(policies):
  log_file = os.path.join(flags.input_dir, p + '.log')
  run_cnter = 0
  with open(log_file, 'r') as f:
    for line in f:
      inj_rate_idx = run_cnter // num_runs
      run_idx = run_cnter % num_runs
      if line.startswith('Avg packet latency'):
        all_data[inj_rate_idx, run_idx, idx*3] = float(line.split(':')[1])
      elif line.startswith('Avg queueing latency'):
        all_data[inj_rate_idx, run_idx, idx*3+1] = float(line.split(':')[1])
      elif line.startswith('Avg network latency'):
        all_data[inj_rate_idx, run_idx, idx*3+2] = float(line.split(':')[1])
        run_cnter += 1

mean_and_std = np.zeros((len(inj_rates), 2, len(policies)*3), 
                        dtype = np.float32)
mean_and_std[:, 0, :] = np.mean(all_data, axis=1)
mean_and_std[:, 1, :] = np.std(all_data, axis=1)
mean_and_std = mean_and_std.reshape(len(inj_rates), 6*len(policies))
mean_and_std = np.hstack((np.array(inj_rates).reshape(-1, 1), mean_and_std))
np.savetxt(flags.output_path, mean_and_std, delimiter=',')
