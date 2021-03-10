lite_cc_library(nnadapter_hal_mediatek_apu SRCS nnadapter_hal.cc)

set(nnadapter_drivers ${nnadapter_drivers} nnadapter_hal_mediatek_apu CACHE INTERNAL "nnadapter_drivers")
