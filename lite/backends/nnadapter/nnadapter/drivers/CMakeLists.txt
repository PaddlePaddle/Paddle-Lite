set(nnadapter_drivers CACHE INTERNAL "")

if(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  add_subdirectory(huawei_kirin_npu)
endif()

if(NNADAPTER_WITH_MEDIATEK_APU)
  add_subdirectory(mediatek_apu)
endif()

if(NNADAPTER_WITH_ROCKCHIP_NPU)
  add_subdirectory(rockchip_npu)
endif()

if(NNADAPTER_WITH_IMAGINATION_NNA)
  add_subdirectory(imagination_nna)
endif()
