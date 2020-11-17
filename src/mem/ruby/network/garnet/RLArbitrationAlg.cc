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
#include "mem/ruby/network/garnet/dt.hh"

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

// concatenates all feature vectors
std::vector<int> concatenate_features(
    const std::vector<int>& tmp_local_age, 
    const std::vector<int>& tmp_payload,
    const std::vector<int>& tmp_hop_count,
    const std::vector<int>& tmp_distance) {

  // First, combine the features
  std::vector<int> res;
  for (int i = 0; i < tmp_local_age.size(); i ++ ) {
    res.push_back(tmp_local_age[i]);
    res.push_back(tmp_payload[i]);
    res.push_back(tmp_hop_count[i]);
    res.push_back(tmp_distance[i]);
  }
  return res;
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
std::vector<float> SwitchAllocator::rl_predict(
    const std::vector<int>& tmp_local_age, 
    const std::vector<int>& tmp_payload,
    const std::vector<int>& tmp_hop_count,
    const std::vector<int>& tmp_distance) {
  std::vector<float> inputs = normalize_features(
    tmp_local_age, tmp_payload, tmp_hop_count, tmp_distance);
  
  // Since we want to support both unified and splitted prediction
  // do a trick here
  int feature_size = layer_sizes[0];
  if (inputs.size() == feature_size) {
    // unified prediction
    std::vector<float>& curr_inputs = inputs;
    for (int layer = 0; layer < n_layers; layer ++ ) {
      std::vector<float> res = linear(curr_inputs, wgts[layer], biases[layer]);
      if (layer != n_layers - 1) 
        relu_inplace(res);
      curr_inputs = res;
    }
    return curr_inputs;
  } else {
    // splitted prediction
    assert(inputs.size() % feature_size == 0);
    std::vector<float> res;
    res.clear();
    int total_num_vcs = inputs.size() / feature_size;
    for (int vc = 0; vc < total_num_vcs; vc ++ ) {
      // prepare a new input vector
      std::vector<float>::const_iterator first = 
          inputs.begin() + vc * feature_size;
      std::vector<float>::const_iterator last = 
          inputs.begin() + vc * feature_size + feature_size;
      std::vector<float> actual_input(first, last);
      // predict the score
      std::vector<float>& curr_inputs = actual_input;
      for (int layer = 0; layer < n_layers; layer ++ ) {
        std::vector<float> res = linear(curr_inputs, wgts[layer], biases[layer]);
        if (layer != n_layers - 1) 
          relu_inplace(res);
        curr_inputs = res;
      }
      // append the result
      assert(curr_inputs.size() == 1);
      res.push_back(curr_inputs[0]);
    }
    return res;
  }
}

// use the distilled decision tree for prediction
std::vector<float> SwitchAllocator::tree_predict(
    const std::vector<int>& tmp_local_age, 
    const std::vector<int>& tmp_payload,
    const std::vector<int>& tmp_hop_count,
    const std::vector<int>& tmp_distance) {

  std::vector<int> inputs = concatenate_features(
    tmp_local_age, tmp_payload, tmp_hop_count, tmp_distance);
  // Since we want to support both unified and splitted prediction
  // do a trick here
  int feature_size = layer_sizes[0];
  if (inputs.size() == feature_size) {
    // unified prediction
    int predicted_class = dt_predict(inputs);
    // must convert into scores so that we can still use the same error
    // recovery logic
    std::vector<float> res(tmp_local_age.size(), 0.0f);
    res[predicted_class] = 1.0f;
    return res;
  } else {
    // splitted prediction
    assert(inputs.size() % feature_size == 0);
    std::vector<float> res;
    res.clear();
    int total_num_vcs = inputs.size() / feature_size;
    for (int vc = 0; vc < total_num_vcs; vc ++ ) {
      // prepare a new input vector
      std::vector<int>::const_iterator first = 
          inputs.begin() + vc * feature_size;
      std::vector<int>::const_iterator last = 
          inputs.begin() + vc * feature_size + feature_size;
      std::vector<int> actual_input(first, last);
      // predict the score
      int predicted_score = dt_predict(actual_input);
      // append the result
      res.push_back(predicted_score);
    }
    return res;
  }
}