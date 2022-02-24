# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Golbal compilation configuration for NNAdapter and its driver HALs
include(driver/configuration.cmake)

add_subdirectory(core)
add_subdirectory(operation)
add_subdirectory(utility)
add_subdirectory(optimizer)
add_subdirectory(runtime)
add_subdirectory(driver)
add_subdirectory(api)
