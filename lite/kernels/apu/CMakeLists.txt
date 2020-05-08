add_subdirectory(bridges)

add_kernel(subgraph_compute_apu APU basic SRCS subgraph_compute.cc DEPS ${lite_kernel_deps} device_apu neuron_adapter subgraph_bridge_engine ${apu_subgraph_bridges})
