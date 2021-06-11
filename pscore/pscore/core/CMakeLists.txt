add_subdirectory(pb)

cc_library(pscore_core
        SRCS
            status.cc
            servable_state.cc
            servable_id.cc
            loader.cc
            resource.cc
            resource_manager.cc
            any_ptr.cc
        DEPS
            error_codes_proto
            absl_base
            glog
        )

cc_test(test_pscore_resource SRCS resource_test.cc DEPS pscore_core)
