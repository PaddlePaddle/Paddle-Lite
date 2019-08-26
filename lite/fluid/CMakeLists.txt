if (LITE_WITH_X86)
lite_cc_library(fluid_data_type SRCS data_type.cc DEPS framework_proto eigen3)
# lite_cc_library(selected_rows SRCS selected_rows.cc)
endif()
