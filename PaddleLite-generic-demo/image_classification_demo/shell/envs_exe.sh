#!/bin/bash
# export DISPLAY=:0.0
export OCL_ICD_DEBUG=0
export MESA_LOADER_DRIVER_OVERRIDE=radeonsi
export OCL_ICD_VENDORS=/home/lixianrui/opencl/mesa_18/tools/etc/OpenCL/vendors
export LD_LIBRARY_PATH=/home/lixianrui/opencl/mesa_18/tools/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/home/lixianrui/opencl/opencv-4.x/install/lib:${LD_LIBRARY_PATH}
export LIBGAL_DRIVERS_PATH=/home/lixianrui/opencl/mesa_18/tools/lib/dri
export CMAKE_PREFIX_PATH=/home/lixianrui/opencl/opencv-4.x/install/
export LD_LIBRARY_PATH=/home/lixianrui/opencl/opencv-4.x/install/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/lixianrui/opencl/opencl-icd/OpenCL-ICD-Loader/install/lib/:$LD_LIBRARY_PATH
