if(NOT LITE_WITH_NNADAPTER)
  return()
endif()

set(nnadapter_drivers CACHE INTERNAL "nnadapter_drivers")

if(NNADAPTER_WITH_MEDIATEK_APU)
  add_subdirectory(mediatek_apu)
endif()

if(NNADAPTER_WITH_MEDIATEK_APU)
  add_subdirectory(rockchip_npu)
endif()
