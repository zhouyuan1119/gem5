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

#include <iostream>
#include <fstream>
#include <ctime>
#include <cstdlib>
#include <algorithm>

#include "mem/ruby/network/garnet/SwitchAllocator.hh"

#include "debug/RubyNetwork.hh"
#include "mem/ruby/network/garnet/GarnetNetwork.hh"
#include "mem/ruby/network/garnet/InputUnit.hh"
#include "mem/ruby/network/garnet/OutputUnit.hh"
#include "mem/ruby/network/garnet/Router.hh"
#include "mem/ruby/network/Topology.hh"

// #define RL_DEBUGGING

// Must initialize the static variable in a source file
bool SwitchAllocator::refresh_requested = true;
float SwitchAllocator::rand_ratio = 0.0f;
std::vector<std::vector<float>> SwitchAllocator::wgts;
std::vector<std::vector<float>> SwitchAllocator::biases;
std::vector<int> SwitchAllocator::simulation_data;

SwitchAllocator::SwitchAllocator(Router *router)
    : Consumer(router)
{
    m_router = router;
    m_num_vcs = m_router->get_num_vcs();
    m_vc_per_vnet = m_router->get_vc_per_vnet();

    m_input_arbiter_activity = 0;
    m_output_arbiter_activity = 0;

    // Change this when we want to run customized arbitration algorithms
    // alg = ROUND_ROBIN;
    alg = TREE;
    // alg = RL;

    // Give initial values for the RL-related variables
    rand_ratio = 0.0f;
    max_distance = 0.0f;
    max_topological_distance = 0.0f;
}

void
SwitchAllocator::init()
{
  m_num_inports = m_router->get_num_inports();
  m_num_outports = m_router->get_num_outports();
  max_num_inports = max_num_outports = 6;
  m_round_robin_inport.resize(m_num_outports);
  m_round_robin_invc.resize(m_num_inports);
  m_port_requests.resize(m_num_outports);
  m_vc_winners.resize(m_num_outports);
#ifdef RL_DEBUGGING
  std::cout << "Router " << m_router->get_id()
            << ": " << m_num_inports << " input ports, "
            << m_num_outports << " output ports, "
            << m_num_vcs << " virtual channels" << std::endl;
#endif

  for (int i = 0; i < m_num_inports; i++) {
    m_round_robin_invc[i] = 0;
  }

  for (int i = 0; i < m_num_outports; i++) {
    m_port_requests[i].resize(m_num_inports);
    m_vc_winners[i].resize(m_num_inports);

    m_round_robin_inport[i] = 0;

    for (int j = 0; j < m_num_inports; j++) {
      m_port_requests[i][j] = false; // [outport][inport]
    }
  }

  if (alg == RL) {
    // allocate space for previous state
    previous_state.resize(max_num_outports);

    for (int i = 0; i < max_num_outports; i ++ ) {
      previous_state[i] = 
        std::vector<float>(max_num_inports * m_num_vcs * 4, 0.0f);
    }
    // Previous reward is 0
    // We give a positive reward if the arbitor makes the decision we want
    // otherwise 0 reward
    previous_reward.resize(max_num_outports);
    // Previous action (winner) is initialized to be invalid
    previous_winner.resize(max_num_outports);
    // previous is easy choice is initialized to be true
    previous_is_easy_choice.resize(max_num_outports);
    for (int i = 0; i < max_num_outports; i ++ ) {
      previous_reward[i] = 0;
      previous_winner[i] = invalid_choice;
      previous_is_easy_choice[i] = true;
    }
    // Allocate space for weights and bias
    if (wgts.size() == 0) {
      for (int i = 0; i < n_layers; i ++ ) {
        wgts.push_back(std::vector<float>());
        biases.push_back(std::vector<float>());
      }
    }
    // initialize two feature normalization factors
    // here we just use the numbers of vnet 0
    Topology* topology_ptr = m_router->get_net_ptr()->get_topology();
    max_distance = float(topology_ptr->max_distance[0]);
    max_topological_distance = 
      float(topology_ptr->max_topological_distance[0]);
    // load weights during initialization
    // the 0-th router is in charge of loading weights here, seems fine for now
    if (m_router->get_id() == 0) {
      std::cout << "[GEM5] Load weights in init()... " << std::endl;
      load_weights();
    }
    // We need to set the random seed
    srand(time(NULL));
  }
}

void SwitchAllocator::load_weights() {
  // Load the weights, assuming the space for the weights have been allocated
  std::ifstream infile;
  infile.open(wgt_pipe_name, std::fstream::in | std::fstream::binary);
  if (infile.is_open()) {
    std::cout << "[GEM5] Loading weights... " << std::endl;
    // The file includes a number for the RL exploration strategy
    infile.read((char*)&rand_ratio, sizeof(float));
    // Read weights and biases, assume they are stored in proper order
    for (int layer = 0; layer < n_layers; layer ++ ) {
      int read_cnt = layer_sizes[layer] * layer_sizes[layer + 1];
      // Read weights directly into the vector data storage
      // We play safe here, although it will be slower
      wgts[layer].clear();
      biases[layer].clear();
      for (int i = 0; i < read_cnt; i ++ ) {
        float buf;
        infile.read((char*)&buf, sizeof(float));
        wgts[layer].push_back(buf);
      }
      if (load_bias) {
        for (int i = 0; i < layer_sizes[layer+1]; i ++ ) {
          float buf;
          infile.read((char*)&buf, sizeof(float));
          biases[layer].push_back(buf);
        }
      } else {
        for (int i = 0; i < layer_sizes[layer+1]; i ++ ) {
          biases[layer].push_back(0.0f);
        }
      }
    }
    infile.close();
    std::cout << "[GEM5] Weights loaded!" << std::endl;
  } else {
    assert(false && "The weight pipe cannot be opened properly!");
  }
}

void SwitchAllocator::dump_data() {
  // Dump all the data currently recorded into the named pipe
  // Then clear the data buffer
  
  std::ofstream outfile;
  outfile.open(data_pipe_name, std::fstream::out | std::fstream::binary);
  if (outfile.is_open()) {
    std::cout << "[GEM5] Dumping data... " << std::endl;
    int state_size = max_num_inports * m_num_vcs * 4;
    // current state, action, next state, reward
    int sample_size = 2 * state_size + 2;
    assert(simulation_data.size() % sample_size == 0);
    int num_samples = simulation_data.size() / sample_size;
    // First write the number of samples
    outfile.write((char*)&num_samples, sizeof(int));
    // Then write out all data, all at once, if there is any
    if (num_samples > 0) {
      outfile.write((char*)simulation_data.data(), 
                    sizeof(int) * simulation_data.size());
      std::cout << "[GEM5] Write " << num_samples << " samples to pipe!"
                << std::endl;
    }
    // Also we would like to write out some stats
    float avg_packet_lat = 
        (m_router->get_net_ptr()->m_avg_packet_latency).total();
    outfile.write((char*)&avg_packet_lat, sizeof(float));
    outfile.close();
    // Clear everything in the simulation data array
    simulation_data.clear();
  } else {
    assert(false && "The data pipe cannot be opened properly!");
  }
  
}

/*
 * The wakeup function of the SwitchAllocator performs a 2-stage
 * seperable switch allocation. At the end of the 2nd stage, a free
 * output VC is assigned to the winning flits of each output port.
 * There is no separate VCAllocator stage like the one in garnet1.0.
 * At the end of this function, the router is rescheduled to wakeup
 * next cycle for peforming SA for any flits ready next cycle.
 */

void
SwitchAllocator::wakeup()
{
  if (alg == ROUND_ROBIN) {
    arbitrate_inports(); // First stage of allocation
    arbitrate_outports(); // Second stage of allocation
  } else {
    // Do the data dump and weight update here
    if (alg == RL) {
      auto curr_cycle = m_router->curCycle();
      if (curr_cycle > warmup_time) {
        if ((curr_cycle - warmup_time) % agent_update_freq == 
            (agent_update_freq - 1)) {
          // Check the flag, dump and load if necessary
          // We used to let the 0-th router to be in charge of this, but it
          // turns out that the 0-th router is not waken up at every cycle...
          // So in this new logic, the first router that is waken up dumps
          // data and loads the weight 
          if (refresh_requested) {
            int dump_cnt = (curr_cycle - warmup_time) / agent_update_freq;
            std::cout << "Dump for the " << dump_cnt << " time... "
                      << std::endl;
            // dump data
            dump_data();
            // load weights
            load_weights();
            // reset the flag
            refresh_requested = false;
          }
        } else if ((curr_cycle - warmup_time) % agent_update_freq == 0) {
          // set the flag again in the next cycle!
          refresh_requested = true;
        }
      }
    }
    unified_arbitrate();
  }

  clear_request_vector();
  check_for_wakeup();
}

/*
 * SA-I (or SA-i) loops through all input VCs at every input port,
 * and selects one in a round robin manner.
 *    - For HEAD/HEAD_TAIL flits only selects an input VC whose output port
 *     has at least one free output VC.
 *    - For BODY/TAIL flits, only selects an input VC that has credits
 *      in its output VC.
 * Places a request for the output port from this input VC.
 */

void SwitchAllocator::arbitrate_inports()
{
  // Select a VC from each input in a round robin manner
  // Independent arbiter at each input port
  for (int inport = 0; inport < m_num_inports; inport++)
  {
    int invc = m_round_robin_invc[inport];

    for (int invc_iter = 0; invc_iter < m_num_vcs; invc_iter++)
    {
      auto input_unit = m_router->getInputUnit(inport);

      if (input_unit->need_stage(invc, SA_, curTick()))
      {
        // This flit is in SA stage

        int outport = input_unit->get_outport(invc);
        int outvc = input_unit->get_outvc(invc);

        // check if the flit in this InputVC is allowed to be sent
        // send_allowed conditions described in that function.
        bool make_request =
            send_allowed(inport, invc, outport, outvc);

        if (make_request)
        {
          m_input_arbiter_activity++;
          m_port_requests[outport][inport] = true;
          m_vc_winners[outport][inport] = invc;

          break; // got one vc winner for this port
        }
      }

      invc++;
      if (invc >= m_num_vcs)
        invc = 0;
    }
  }
}

/*
 * SA-II (or SA-o) loops through all output ports,
 * and selects one input VC (that placed a request during SA-I)
 * as the winner for this output port in a round robin manner.
 *      - For HEAD/HEAD_TAIL flits, performs simplified outvc allocation.
 *        (i.e., select a free VC from the output port).
 *      - For BODY/TAIL flits, decrement a credit in the output vc.
 * The winning flit is read out from the input VC and sent to the
 * CrossbarSwitch.
 * An increment_credit signal is sent from the InputUnit
 * to the upstream router. For HEAD_TAIL/TAIL flits, is_free_signal in the
 * credit is set to true.
 */

void SwitchAllocator::arbitrate_outports()
{
  // Now there are a set of input vc requests for output vcs.
  // Again do round robin arbitration on these requests
  // Independent arbiter at each output port
  for (int outport = 0; outport < m_num_outports; outport++)
  {
    int inport = m_round_robin_inport[outport];

    for (int inport_iter = 0; inport_iter < m_num_inports;
         inport_iter++)
    {

      // inport has a request this cycle for outport
      if (m_port_requests[outport][inport])
      {
        auto output_unit = m_router->getOutputUnit(outport);
        auto input_unit = m_router->getInputUnit(inport);

        // grant this outport to this inport
        int invc = m_vc_winners[outport][inport];

        int outvc = input_unit->get_outvc(invc);
        if (outvc == -1)
        {
          // VC Allocation - select any free VC from outport
          outvc = vc_allocate(outport, inport, invc);
        }

        // remove flit from Input VC
        flit *t_flit = input_unit->getTopFlit(invc);

        DPRINTF(RubyNetwork, "SwitchAllocator at Router %d "
                             "granted outvc %d at outport %d "
                             "to invc %d at inport %d to flit %s at "
                             "cycle: %lld\n",
                m_router->get_id(), outvc,
                m_router->getPortDirectionName(
                    output_unit->get_direction()),
                invc,
                m_router->getPortDirectionName(
                    input_unit->get_direction()),
                *t_flit,
                m_router->curCycle());

        // Update outport field in the flit since this is
        // used by CrossbarSwitch code to send it out of
        // correct outport.
        // Note: post route compute in InputUnit,
        // outport is updated in VC, but not in flit
        t_flit->set_outport(outport);

        // set outvc (i.e., invc for next hop) in flit
        // (This was updated in VC by vc_allocate, but not in flit)
        t_flit->set_vc(outvc);

        // decrement credit in outvc
        output_unit->decrement_credit(outvc);

        // flit ready for Switch Traversal
        t_flit->advance_stage(ST_, curTick());
        m_router->grant_switch(inport, t_flit);
        m_output_arbiter_activity++;

        if ((t_flit->get_type() == TAIL_) ||
            t_flit->get_type() == HEAD_TAIL_)
        {

          // This Input VC should now be empty
          assert(!(input_unit->isReady(invc, curTick())));

          // Free this VC
          input_unit->set_vc_idle(invc, curTick());

          // Send a credit back
          // along with the information that this VC is now idle
          input_unit->increment_credit(invc, true, curTick());
        }
        else
        {
          // Send a credit back
          // but do not indicate that the VC is idle
          input_unit->increment_credit(invc, false, curTick());
        }

        // remove this request
        m_port_requests[outport][inport] = false;

        // Update Round Robin pointer
        m_round_robin_inport[outport] = inport + 1;
        if (m_round_robin_inport[outport] >= m_num_inports)
          m_round_robin_inport[outport] = 0;

        // Update Round Robin pointer to the next VC
        // We do it here to keep it fair.
        // Only the VC which got switch traversal
        // is updated.
        m_round_robin_invc[inport] = invc + 1;
        if (m_round_robin_invc[inport] >= m_num_vcs)
          m_round_robin_invc[inport] = 0;

        break; // got a input winner for this outport
      }

      inport++;
      if (inport >= m_num_inports)
        inport = 0;
    }
  }
}

/* 
 * One-stage arbitration algorithm, including multiple versions
 *
 * This algorithm uses the distilled form of an RL agent to make decisions. It
 * handles both VC selection at input ports and input selection at output ports.
 *
 * At each output port, the RL agent gets queried. Features from the heads of
 * all virtual channels are sent to the agent. 
 */
void SwitchAllocator::unified_arbitrate() {
  
  // For each output port, collect information from all input ports and 
  // VCs, send the information into the model and get predictions

  // Pre-compute this so we don't have to recompute every time
  int num_inputs = max_num_inports * m_num_vcs;

  // The final decisions
  std::vector<int> output_port_winners(max_num_outports, invalid_choice);

#ifdef RL_DEBUGGING
  // Dump some topology-dependent information
  Topology* topology_ptr = m_router->get_net_ptr()->get_topology();
  for (int vnet = 0; vnet < 3; vnet ++ ) {
    std::cout << "Max distance of vnet " << vnet << ": " 
              << topology_ptr->max_distance[vnet] << std::endl;
    std::cout << "Max hop count of vnet " << vnet << ": " 
              << topology_ptr->max_topological_distance[vnet] << std::endl;
  }
#endif

  // TODO: this part definitely has room for improvement (at cost of more space)
  //       low priority for now, do it later
  
  // Loop through all output ports
  for (int outport = 0; outport < m_num_outports; outport ++ ) {
    // Allocate space for prediction and logging
    std::vector<bool> useful_for_this_port(num_inputs, false);
    std::vector<int> tmp_local_age(num_inputs, 0);
    std::vector<int> tmp_payload_size(num_inputs, 0);
    std::vector<int> tmp_hop_count(num_inputs, 0);
    std::vector<float> tmp_global_age(num_inputs, 0.0f);
    std::vector<int> tmp_distance(num_inputs, 0);

    // Loop through all inport ports, all input VCs, and get the features
    populate_features(useful_for_this_port,
                      tmp_local_age,
                      tmp_payload_size,
                      tmp_hop_count,
                      tmp_global_age,
                      tmp_distance,
                      outport);

    // Now we should have all the features from each VC
    // with the ones unrelated to the current outport masked

    // If our agent has more than one choice, then refer to the model
    // Otherwise, we can tell right now
    int winner = invalid_choice;
    int tmp_reward = 0;
    bool easy_choice = check_trivial_cases(useful_for_this_port, winner);
    if (!easy_choice) {
      std::vector<float> scores;
      if (alg == RL) {
        // Use our model to predict
        scores = rl_predict(
            tmp_local_age, tmp_payload_size, tmp_hop_count, tmp_distance);
      } else if (alg == GLOBAL_AGE) {
        scores = global_age_predict(tmp_global_age);
      } else if (alg == LOGIC) {
        scores = logic_predict(tmp_local_age, tmp_hop_count);
      } else if (alg == LOCAL_AGE) {
        scores = local_age_predict(tmp_local_age);
      } else if (alg == TREE) {
        scores = tree_predict(
          tmp_local_age, tmp_payload_size, tmp_hop_count, tmp_distance);
      }
      // Choose the best legal result
      winner = choose_best_result(scores, useful_for_this_port);
      if (alg == RL) {
        // Compute the reward for RL
        // Same with Jieming's HPCA'20 paper, we give a constant reward for 
        // choosing the globally oldest message, and 0 otherwise
        tmp_reward = 
          compute_reward(winner, tmp_global_age, useful_for_this_port);
      }
    }
    // Store the winner
    output_port_winners[outport] = winner;


    // Log by writing to the giant vector we are maintaining
    // Only log when the previous cycle was not an easy choice
    // Also, not logging if we are during warmup
    // Then refresh the previous state, also not refreshing during warmup
    if (alg == RL) {
      auto curr_cycle = m_router->curCycle();
      if (curr_cycle > warmup_time) {
        if (!previous_is_easy_choice[outport]) {
          dump_rl_data(useful_for_this_port,
                      tmp_local_age,
                      tmp_payload_size,
                      tmp_hop_count,
                      tmp_global_age,
                      tmp_distance,
                      outport);
        }
        // Refresh the previous state
        for (int i = 0; i < num_inputs; i ++ ) {
          previous_state[outport][4*i] = tmp_local_age[i];
          previous_state[outport][4*i+1] = tmp_payload_size[i];
          previous_state[outport][4*i+2] = tmp_hop_count[i];
          previous_state[outport][4*i+3] = tmp_distance[i];
        }
        previous_reward[outport] = tmp_reward;
        previous_winner[outport] = winner;
        previous_is_easy_choice[outport] = easy_choice;
      }
    }
  }
  
  // Actually do the arbitration
  arbitrate_outports_with_winners(output_port_winners);

}

/* 
 * Choose the best legal result from the ML predictor's scores
 */
int SwitchAllocator::choose_best_result(
    const std::vector<float>& scores, 
    const std::vector<bool>& useful) {
  // Count the number of actual useful inputs
  int useful_cnt = 0;
  for (int i = 0; i < useful.size(); i ++ )
    useful_cnt += useful[i];
  // If nothing is useful, return -1 (invalid choice)
  if (useful_cnt == 0)
    return invalid_choice;

  float best_score = -100000.0f; // should be small enough
  int best_choice = -1;

  // Choose between random choice and selecting the best
  int rand_num = rand() % 10000;
  if (rand_num < int(10000 * rand_ratio)) {
    // Make a random choice
    int chosen_idx = rand() % useful_cnt;
    int cnter = 0;
    for (int i = 0; i < scores.size(); i ++ ) {
      if (useful[i]) {
        if (cnter == chosen_idx) {
          best_choice = i;
          break;
        } else {
          cnter += 1;
        }
      }
    }
  } else {
    // Not messing with the iterators and stuff
    for (int i = 0; i < scores.size(); i ++ ) {
      if ((scores[i] > best_score) && (useful[i])) {
        best_score = scores[i];
        best_choice = i;
      }
    }
  }
  assert(best_choice != invalid_choice);
  return best_choice;
}

/*
 * Compute the reward of the model's prediction
 */
int SwitchAllocator::compute_reward(
    const int winner,
    const std::vector<float>& global_age,
    const std::vector<bool>& useful) {
  float max_age = -100000.0f; // should be small enough
  int best_choice = invalid_choice;

  // Not messing with the iterators and stuff
  for (int i = 0; i < global_age.size(); i ++ ) {
    if ((global_age[i] > max_age) && (useful[i])) {
      max_age = global_age[i];
      best_choice = i;
    }
  }
  // Notice that if the best choice is the invalid choice
  // or the winner is invalid choice
  // there is no reward
  return int((winner != invalid_choice) && (best_choice != invalid_choice)
             && (best_choice == winner));
}

// send flits to output ports based on the arbitration result 
void SwitchAllocator::arbitrate_outports_with_winners(
    const std::vector<int>& output_port_winners) {
  // We are done with arbitration, get the winners' flits
  // Now there are a set of input vc requests for output vcs.
  // Again do round robin arbitration on these requests
  // Independent arbiter at each output port
  for (int outport = 0; outport < m_num_outports; outport++) {
    // Get the winner
    int winner = output_port_winners[outport];
    // Skip if we don't have any accesses targeting this port at all
    if (winner != invalid_choice) {
      // Compute the actual inport and in-vc of the winner
      int inport = winner / m_num_vcs;
      int invc = winner % m_num_vcs;

      // The rest should be the same with arbitrate_outports()...

      // Get the input and output units
      auto output_unit = m_router->getOutputUnit(outport);
      auto input_unit = m_router->getInputUnit(inport);
      // grant this outport to this inport
      int outvc = input_unit->get_outvc(invc);
      if (outvc == -1) {
        // VC Allocation - select any free VC from outport
        outvc = vc_allocate(outport, inport, invc);
      }

      // remove flit from Input VC
      flit *t_flit = input_unit->getTopFlit(invc);

      DPRINTF(RubyNetwork, "[RLArbitor] SwitchAllocator at Router %d "
        "granted outvc %d at outport %d "
        "to invc %d at inport %d to flit %s at "
        "cycle: %lld\n",
        m_router->get_id(), outvc,
        m_router->getPortDirectionName(
          output_unit->get_direction()),
        invc,
        m_router->getPortDirectionName(
          input_unit->get_direction()),
        *t_flit,
        m_router->curCycle());


      t_flit->set_outport(outport);
      t_flit->set_vc(outvc);
      output_unit->decrement_credit(outvc);
      t_flit->advance_stage(ST_, curTick());
      m_router->grant_switch(inport, t_flit);
      m_output_arbiter_activity++;

      if ((t_flit->get_type() == TAIL_) || (t_flit->get_type() == HEAD_TAIL_)) {
        // This Input VC should now be empty
        assert(!(input_unit->isReady(invc, curTick())));
        // Free this VC
        input_unit->set_vc_idle(invc, curTick());
        // Send a credit back
        // along with the information that this VC is now idle
        input_unit->increment_credit(invc, true, curTick());
      } else {
        // Send a credit back
        // but do not indicate that the VC is idle
        input_unit->increment_credit(invc, false, curTick());
      }

      // remove this request
      m_port_requests[outport][inport] = false;

      // Not updating any of the round-robin stuff because we are not using it 
    }
  }
}


// Loop through all inport ports, all input VCs, and get the features
void SwitchAllocator::populate_features(
    std::vector<bool>& useful_for_this_port,
    std::vector<int>& tmp_local_age,
    std::vector<int>& tmp_payload_size,
    std::vector<int>& tmp_hop_count,
    std::vector<float>& tmp_global_age,
    std::vector<int>& tmp_distance,
    int outport) {
  // Now go through all inport ports and gather the features
  for (int inport = 0; inport < m_num_inports; inport++) {
    // Get the input unit
    auto input_unit = m_router->getInputUnit(inport);

    // Go through all virtual channels
    for (int invc = 0; invc < m_num_vcs; invc++) {
      // Compute the index
      int idx = inport * m_num_vcs + invc;

      // If this channel needs arbitration, go on
      if (input_unit->need_stage(invc, SA_, curTick())) {
        // check if the flit in this InputVC is allowed to be sent
        // send_allowed conditions described in that function.
        int target_outport = input_unit->get_outport(invc);
        int target_outvc = input_unit->get_outvc(invc);
#ifdef RL_DEBUGGING
        std::cout << "[RLArbitor] Cycle " << m_router->curCycle()
          << ": Flit at input port " << inport 
          << ", invc " << invc 
          << " targets outport " << target_outport 
          << ", outvc " << target_outvc << std::endl;
#endif
        // Only proceed if this flit is going to the target outport
        if (target_outport == outport) {
          bool make_request = send_allowed(inport, invc, 
              target_outport, target_outvc);
#ifdef RL_DEBUGGING
          std::cout << "make_request = " << make_request << std::endl;
#endif

          if (make_request) {
            // If it is allowed, add information to the queues
            flit *t_flit = input_unit->peekTopFlit(invc);
#ifdef RL_DEBUGGING
            std::cout << "Current tick: " << curTick() 
              << ", local enqueue time: " << t_flit->get_local_enqueue_time()
              << ", creation time: " << t_flit->get_enqueue_time()
              << std::endl;
#endif
            // Local age
            Tick clk_period = m_router->clockPeriod();
            tmp_local_age[idx] = std::min(32UL, 
                (curTick() - t_flit->get_local_enqueue_time()) / clk_period);
            // Payload size
            // notice that the two types of sizes are 8 and 72... 
            tmp_payload_size[idx] = std::min(72, t_flit->msgSize);
            // Get the rest of the information from the route
            RouteInfo route = t_flit->get_route();

            // distance from current router to destination
            // get from the routing table
            Topology* topology_ptr = m_router->get_net_ptr()->get_topology();
            int vnet = route.vnet;
            int curr_id = m_router->get_id() + 2 * topology_ptr->m_nodes;
            int dest_id = route.dest_router + 2 * topology_ptr->m_nodes;
            tmp_distance[idx] = 
              topology_ptr->shortest_path_table[vnet][curr_id][dest_id];

            // global age, in cycles
            tmp_global_age[idx] = 
              ((curTick() - t_flit->get_enqueue_time()) / float(clk_period)) / 
              global_age_norm_factor; 

            // number of hops already traversed
            tmp_hop_count[idx] = route.hops_traversed;

            // This VC is useful for the current port
            useful_for_this_port[idx] = true;
          }
          // We don't need to do anything otherwise, because the invalid tokens
          // are already pre-set
        }
        // Same here
      }
      // Same here
    }
  }
}


// Check for trivial cases to reduce the number of useless samples
bool SwitchAllocator::check_trivial_cases(
    const std::vector<bool>& useful_for_this_port,
    int& winner) {
  int useful_cnt = 0;
  for (int i = 0; i < useful_for_this_port.size(); i ++ ) {
    if (useful_for_this_port[i]) {
      useful_cnt += 1;
      winner = i;
    }
  }
  if (useful_cnt <= 1) {
    return true;
  } else {
    winner = invalid_choice;
    return false;
  }
}


// Predict using global age (oracle)
std::vector<float> SwitchAllocator::global_age_predict(
  const std::vector<float>& tmp_global_age) {
  // Don't really need to do anything, directly return this
  return tmp_global_age;
}

// Predict using local age (FIFO)
std::vector<float> SwitchAllocator::local_age_predict(
  const std::vector<int>& tmp_local_age) {
  // Don't really need to do anything, directly return this
  std::vector<float> scores(tmp_local_age.size(), 0.0f);
  for (int i = 0; i < tmp_local_age.size(); i ++ ) {
    scores[i] = float(tmp_local_age[i]);
  }
  return scores;
}

// Predict using the logic representation from the HPCA paper
std::vector<float> SwitchAllocator::logic_predict(
  const std::vector<int>& tmp_local_age,
  const std::vector<int>& tmp_hop_count) {
  std::vector<float> scores(tmp_local_age.size(), 0.0f);
  for (int i = 0; i < tmp_local_age.size(); i ++ ) {
    // 4x4 mesh formula
    scores[i] = tmp_local_age[i] * 2.0f + tmp_hop_count[i] * 0.5f;
    // 8x8 mesh formula
    // scores[i] = tmp_local_age[i] + tmp_hop_count[i] * 2.0f;
  }
  return scores;
}

// Dump RL data into a file
void SwitchAllocator::dump_rl_data(
  const std::vector<bool>& useful_for_this_port,
  const std::vector<int>& tmp_local_age,
  const std::vector<int>& tmp_payload_size,
  const std::vector<int>& tmp_hop_count,
  const std::vector<float>& tmp_global_age,
  const std::vector<int>& tmp_distance,
  const int outport) {
  int num_inputs = max_num_inports * m_num_vcs;
#ifdef RL_DEBUGGING
  // Print all results onto the screen
  std::cout << "[RLArbitor] RL features at cycle " << m_router->curCycle()
            << ", router " << m_router->get_id()
            << ", outport " << outport << ": " << std::endl;
  std::cout << "-----------Previous state----------" << std::endl;
  std::cout << "Local age: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << previous_state[outport][4*i] << " ";
  std::cout << std::endl;
  std::cout << "Payload size: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << previous_state[outport][4*i+1] << " ";
  std::cout << std::endl;
  std::cout << "Hop count: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << previous_state[outport][4*i+2] << " ";
  std::cout << std::endl;
  std::cout << "Distance: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << previous_state[outport][4*i+3] << " ";
  std::cout << std::endl;
  std::cout << "-----------Current state----------" << std::endl;
  std::cout << "Local age: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << tmp_local_age[i] << " ";
  std::cout << std::endl;
  std::cout << "Payload size: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << tmp_payload_size[i] << " ";
  std::cout << std::endl;
  std::cout << "Hop count: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << tmp_hop_count[i] << " ";
  std::cout << std::endl;
  std::cout << "Distance: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << tmp_distance[i] << " ";
  std::cout << std::endl;
  std::cout << "Global age: ";
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << tmp_global_age[i] << " ";
  std::cout << std::endl;
  std::cout << "-----------Scores----------" << std::endl;
  for (int i = 0; i < num_inputs; i ++ ) 
    std::cout << scores[i] << " ";
  std::cout << std::endl;
  std::cout << "----------Useful-----------" << std::endl;
  for (int i = 0; i < num_inputs; i ++ )
    std::cout << useful_for_this_port[i] << " ";
  std::cout << std::endl;
  std::cout << "Select input " << winner 
            << ", reward = " << tmp_reward << std::endl;
#else
  // Add the collected data into the whole data vector
  // Only dump the cases where some input is selected
  if (previous_winner[outport] != invalid_choice) {
    // Dump the "current state"
    // actually stored as a previous state in our class
    for (int i = 0; i < num_inputs; i ++ ) {
      // previous local age
      simulation_data.push_back(previous_state[outport][4*i]);
      // previous payload size
      simulation_data.push_back(previous_state[outport][4*i+1]);
      // previous hop count
      simulation_data.push_back(previous_state[outport][4*i+2]);
      // previous distance
      simulation_data.push_back(previous_state[outport][4*i+3]);
    }
    // action
    simulation_data.push_back(previous_winner[outport]);
    // Dump the "next state" (actually the current state)
    for (int i = 0; i < num_inputs; i ++ ) {
      simulation_data.push_back(tmp_local_age[i]);
      simulation_data.push_back(tmp_payload_size[i]);
      simulation_data.push_back(tmp_hop_count[i]);
      simulation_data.push_back(tmp_distance[i]);
    }
    // Finally, dump the reward
    simulation_data.push_back(previous_reward[outport]);
  }
#endif
}

/*
 * A flit can be sent only if
 * (1) there is at least one free output VC at the
 *     output port (for HEAD/HEAD_TAIL),
 *  or
 * (2) if there is at least one credit (i.e., buffer slot)
 *     within the VC for BODY/TAIL flits of multi-flit packets.
 * and
 * (3) pt-to-pt ordering is not violated in ordered vnets, i.e.,
 *     there should be no other flit in this input port
 *     within an ordered vnet
 *     that arrived before this flit and is requesting the same output port.
 */

bool SwitchAllocator::send_allowed(int inport, int invc, int outport, int outvc)
{
  // Check if outvc needed
  // Check if credit needed (for multi-flit packet)
  // Check if ordering violated (in ordered vnet)

  int vnet = get_vnet(invc);
  bool has_outvc = (outvc != -1);
  bool has_credit = false;

  auto output_unit = m_router->getOutputUnit(outport);
  if (!has_outvc)
  {

    // needs outvc
    // this is only true for HEAD and HEAD_TAIL flits.

    if (output_unit->has_free_vc(vnet))
    {

      has_outvc = true;

      // each VC has at least one buffer,
      // so no need for additional credit check
      has_credit = true;
    }
  }
  else
  {
    has_credit = output_unit->has_credit(outvc);
  }

  // cannot send if no outvc or no credit.
#ifdef RL_DEBUGGING
  if (!has_outvc)
    std::cout << "No free outvc!" << std::endl;
  if (!has_credit)
    std::cout << "No credit left!" << std::endl;
#endif
  if (!has_outvc || !has_credit)
    return false;

  // protocol ordering check
  if ((m_router->get_net_ptr())->isVNetOrdered(vnet))
  {
    auto input_unit = m_router->getInputUnit(inport);

    // enqueue time of this flit
    Tick t_enqueue_time = input_unit->get_enqueue_time(invc);

    // check if any other flit is ready for SA and for same output port
    // and was enqueued before this flit
    int vc_base = vnet * m_vc_per_vnet;
    for (int vc_offset = 0; vc_offset < m_vc_per_vnet; vc_offset++)
    {
      int temp_vc = vc_base + vc_offset;
      if (input_unit->need_stage(temp_vc, SA_, curTick()) &&
          (input_unit->get_outport(temp_vc) == outport) &&
          (input_unit->get_enqueue_time(temp_vc) < t_enqueue_time))
      {
#ifdef RL_DEBUGGING
        std::cout << "Ordering violated!" << std::endl;
#endif
        return false;
      }
    }
  }

  return true;
}

// Assign a free VC to the winner of the output port.
int SwitchAllocator::vc_allocate(int outport, int inport, int invc)
{
  // Select a free VC from the output port
  int outvc =
      m_router->getOutputUnit(outport)->select_free_vc(get_vnet(invc));

  // has to get a valid VC since it checked before performing SA
  assert(outvc != -1);
  m_router->getInputUnit(inport)->grant_outvc(invc, outvc);
  return outvc;
}

// Wakeup the router next cycle to perform SA again
// if there are flits ready.
void SwitchAllocator::check_for_wakeup()
{
  Tick nextCycle = m_router->clockEdge(Cycles(1));

  if (m_router->alreadyScheduled(nextCycle))
  {
    return;
  }

  for (int i = 0; i < m_num_inports; i++)
  {
    for (int j = 0; j < m_num_vcs; j++)
    {
      if (m_router->getInputUnit(i)->need_stage(j, SA_, nextCycle))
      {
        m_router->schedule_wakeup(Cycles(1));
        return;
      }
    }
  }
}

int SwitchAllocator::get_vnet(int invc)
{
  int vnet = invc / m_vc_per_vnet;
  assert(vnet < m_router->get_num_vnets());
  return vnet;
}

// Clear the request vector within the allocator at end of SA-II.
// Was populated by SA-I.
void SwitchAllocator::clear_request_vector()
{
  for (int i = 0; i < m_num_outports; i++)
  {
    for (int j = 0; j < m_num_inports; j++)
    {
      m_port_requests[i][j] = false;
    }
  }
}

void SwitchAllocator::resetStats()
{
  m_input_arbiter_activity = 0;
  m_output_arbiter_activity = 0;
}
