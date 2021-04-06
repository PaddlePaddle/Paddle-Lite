# Enable throwing exception when check failed in nnadapter_logging.cc
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fexceptions -fasynchronous-unwind-tables -funwind-tables")

set(NNADAPTER_DRIVER_PREFIX nnadapter_driver)
add_definitions(-DNNADAPTER_DRIVER_PREFIX=${NNADAPTER_DRIVER_PREFIX})

add_library(nnadapter_logging STATIC nnadapter_logging.cc)
add_library(nnadapter_driver STATIC nnadapter_driver.cc)
add_library(nnadapter SHARED nnadapter_api.cc nnadapter_runtime.cc)
target_link_libraries(nnadapter nnadapter_logging nnadapter_driver)

add_subdirectory(drivers)
