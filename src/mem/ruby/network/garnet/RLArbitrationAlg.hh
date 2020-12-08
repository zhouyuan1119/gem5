#ifndef __MEM_RUBY_NETWORK_GARNET_0_RLARBITRATIONALG_HH__
#define __MEM_RUBY_NETWORK_GARNET_0_RLARBITRATIONALG_HH__
#include <string>
const int n_layers = 2;
const int layer_sizes[n_layers+1] = {4, 16, 1};
const std::string wgt_pipe_name = "/work/shared/users/phd/yz882/comparc_learn/test_0.36/wgt_pipe";
const std::string data_pipe_name = "/work/shared/users/phd/yz882/comparc_learn/test_0.36/data_pipe";
const long long int agent_update_freq = 5000LL;
const long long int warmup_time = 2000000LL;
const bool load_bias = false;
#endif
