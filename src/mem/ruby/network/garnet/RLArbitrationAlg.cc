/*
 *               RL-based arbitration algorithm implementation
 *                         Yuan Zhou 10/14/2020
 *
 */

#include <vector>
#include <cassert>
#include <iostream>

#include "mem/ruby/network/garnet/SwitchAllocator.hh"
#include "mem/ruby/network/garnet/RLArbitrationAlg.hh"

// Define two basic operators: matrix-times-vector and relu
// They operate on "vectors"

// relu
void relu_inplace(std::vector<float>& input) {
  // Just use loops
  for (int i = 0; i < input.size(); i ++ ) {
    input[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
  }
}

// dense layer
std::vector<float> linear(
  const std::vector<float>& input, 
  const std::vector<float>& wgt,
  const std::vector<float>& bias) {
  // Check that the sizes match
  int input_size = input.size();
  int wgt_array_size = wgt.size();
  assert(wgt_array_size % input_size == 0);

  // Allocate space
  int output_size = wgt_array_size / input_size;
  assert(bias.size() == output_size);
  std::vector<float> output(output_size, 0.0f);
  
  // Start computing
  for (int out_idx = 0; out_idx < output_size; out_idx ++ ) {
    int wgt_start_idx = out_idx * input_size;
    for (int in_idx = 0; in_idx < input_size; in_idx ++ ) {
      output[out_idx] += input[in_idx] * wgt[wgt_start_idx + in_idx];
    }
    output[out_idx] += bias[out_idx];
  }
  return output;
}

// the normalization function, normalizes the features to the range of (0, 1)
std::vector<float> SwitchAllocator::normalize_features(
    const std::vector<int>& tmp_local_age, 
    const std::vector<int>& tmp_payload,
    const std::vector<int>& tmp_hop_count,
    const std::vector<int>& tmp_distance) {

  // First, combine the features
  std::vector<float> normalized;
  for (int i = 0; i < tmp_local_age.size(); i ++ ) {
    normalized.push_back(tmp_local_age[i] / local_age_norm_factor);
    normalized.push_back(tmp_payload[i] / payload_size_norm_factor);
    normalized.push_back(tmp_hop_count[i] / max_topological_distance);
    normalized.push_back(tmp_distance[i] / max_distance);
  }
  return normalized;
}

// the prediction function, assuming a neural net
// if we use a tree representation during evaluation, just replace this 
// completely with another function
std::vector<float> SwitchAllocator::ml_predict(
    const std::vector<int>& tmp_local_age, 
    const std::vector<int>& tmp_payload,
    const std::vector<int>& tmp_hop_count,
    const std::vector<int>& tmp_distance) {
  std::vector<float> inputs = normalize_features(
    tmp_local_age, tmp_payload, tmp_hop_count, tmp_distance);
  std::vector<float>& curr_inputs = inputs;
  for (int layer = 0; layer < n_layers; layer ++ ) {
    std::vector<float> res = linear(curr_inputs, wgts[layer], biases[layer]);
    if (layer != n_layers - 1) 
      relu_inplace(res);
    curr_inputs = res;
  }

  return curr_inputs;
}

