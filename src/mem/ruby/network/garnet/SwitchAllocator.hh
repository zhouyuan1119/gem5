/*
 * Copyright (c) 2020 Inria
 * Copyright (c) 2016 Georgia Institute of Technology
 * Copyright (c) 2008 Princeton University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#ifndef __MEM_RUBY_NETWORK_GARNET_0_SWITCHALLOCATOR_HH__
#define __MEM_RUBY_NETWORK_GARNET_0_SWITCHALLOCATOR_HH__

#include <iostream>
#include <vector>

#include "mem/ruby/common/Consumer.hh"
#include "mem/ruby/network/garnet/CommonTypes.hh"
#include "mem/ruby/network/garnet/RLArbitrationAlg.hh"

// RL training related variables
typedef enum {
  ROUND_ROBIN = 0,
  RL,
  GLOBAL_AGE,
  LOGIC,
  LOCAL_AGE,
  TREE
} ArbitrationAlg;
const int invalid_choice = -1;
const float global_age_norm_factor = 500.0f;
const float payload_size_norm_factor = 72.0f;
const float local_age_norm_factor = 31.0f;

class Router;
class InputUnit;
class OutputUnit;

class SwitchAllocator : public Consumer
{
  public:
    SwitchAllocator(Router *router);
    void wakeup();
    void init();
    void clear_request_vector();
    void check_for_wakeup();
    int get_vnet (int invc);
    void print(std::ostream& out) const {};
    void arbitrate_inports();
    void arbitrate_outports();
    bool send_allowed(int inport, int invc, int outport, int outvc);
    int vc_allocate(int outport, int inport, int invc);

    // Utilities for RL-based arbitration
    
    // dump data
    void dump_data();
    // load weights
    void load_weights();
    // Top-level function of one-stage arbitration
    // This will replace both arbitrate_inports() and arbitrate_outputs()
    void unified_arbitrate();
    // ML-prediction functions, defined in a separate file
    std::vector<float> normalize_features(
      const std::vector<int>& tmp_local_age, 
      const std::vector<int>& tmp_payload,
      const std::vector<int>& tmp_hop_count,
      const std::vector<int>& tmp_distance);
    std::vector<float> rl_predict(
      const std::vector<int>& tmp_local_age, 
      const std::vector<int>& tmp_payload,
      const std::vector<int>& tmp_hop_count,
      const std::vector<int>& tmp_distance);
    std::vector<float> tree_predict(
      const std::vector<int>& tmp_local_age, 
      const std::vector<int>& tmp_payload,
      const std::vector<int>& tmp_hop_count,
      const std::vector<int>& tmp_distance);
    // Choose the best legal result
    int choose_best_result(
      const std::vector<float>& scores,
      const std::vector<bool>& useful);
    // Compute the reward
    int compute_reward(
      const int winner, 
      const std::vector<float>& global_age,
      const std::vector<bool>& useful);

    // send flits to output ports based on the arbitration result
    void arbitrate_outports_with_winners(
      const std::vector<int>& output_port_winners);

    // Loop through all inport ports, all input VCs, and get the features
    void populate_features(
      std::vector<bool>& useful_for_this_port,
      std::vector<int>& tmp_local_age,
      std::vector<int>& tmp_payload_size,
      std::vector<int>& tmp_hop_count,
      std::vector<float>& tmp_global_age,
      std::vector<int>& tmp_distance,
      const int outport);

    // Check for trivial cases to reduce the number of useless samples
    bool check_trivial_cases(
      const std::vector<bool>& useful_for_this_port,
      int& winner);

    // Dump RL data into a file
    void dump_rl_data(
      const std::vector<bool>& useful_for_this_port,
      const std::vector<int>& tmp_local_age,
      const std::vector<int>& tmp_payload_size,
      const std::vector<int>& tmp_hop_count,
      const std::vector<float>& tmp_global_age,
      const std::vector<int>& tmp_distance,
      const int outport);

    // We also implement several baselines just to evaluate whether the problem
    // is meaningful or not

    // Predict using global age (oracle)
    std::vector<float> global_age_predict(
      const std::vector<float>& tmp_global_age);

    // Predict using the logic representation from the HPCA paper
    std::vector<float> logic_predict(
      const std::vector<int>& tmp_local_age,
      const std::vector<int>& tmp_hop_count);

    // Predict using local age (FIFO)
    std::vector<float> local_age_predict(
      const std::vector<int>& tmp_local_age);

    // Some static variables shared across all routers

    // Neural net weights used for online prediction
    static std::vector<std::vector<float>> wgts;
    static std::vector<std::vector<float>> biases;
    // Probability that we will choose a random action
    static float rand_ratio;

    // A giant vector to store all the data collected in the current period
    static std::vector<int> simulation_data;

    // A boolean to track whether it is time to dump data or not
    static bool refresh_requested;

    // RL-related stuff ends here

    inline double
    get_input_arbiter_activity()
    {
        return m_input_arbiter_activity;
    }
    inline double
    get_output_arbiter_activity()
    {
        return m_output_arbiter_activity;
    }

    void resetStats();


  private:
    int m_num_inports, m_num_outports;
    int m_num_vcs, m_vc_per_vnet;

    double m_input_arbiter_activity, m_output_arbiter_activity;

    Router *m_router;
    std::vector<int> m_round_robin_invc;
    std::vector<int> m_round_robin_inport;
    std::vector<std::vector<bool>> m_port_requests;
    std::vector<std::vector<int>> m_vc_winners; // a list for each outport

    // Some private variables for RL routing policy

    // for RL, we allocate space for 6-port routers
    int max_num_inports, max_num_outports;  
    // Choose between RL-based routing algorithm, various baselines,
    // and round-robin
    ArbitrationAlg alg;
    // Also we need to store some normalization factors
    float max_distance;
    float max_topological_distance;

    // We need to store the "previous state" at each port
    // so we can use the target network to predict the future reward
    std::vector<std::vector<float>> previous_state;
    // Also the previous reward
    std::vector<int> previous_reward;
    // Also the previous action
    std::vector<int> previous_winner;
    // Also whether the previous choice is an easy choice
    std::vector<bool> previous_is_easy_choice;
};

#endif // __MEM_RUBY_NETWORK_GARNET_0_SWITCHALLOCATOR_HH__
