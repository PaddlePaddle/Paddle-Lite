lite_cc_library(nnadapter_hal_rockchip_npu SRCS nnadapter_hal.cc)

set(nnadapter_drivers ${nnadapter_drivers} nnadapter_hal_rockchip_npu CACHE INTERNAL "nnadapter_drivers")
