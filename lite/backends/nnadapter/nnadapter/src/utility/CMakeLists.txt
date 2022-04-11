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

add_library(nnadapter_utility_logging STATIC logging.cc)
add_library(nnadapter_utility_string STATIC string.cc)
add_library(nnadapter_utility_debug STATIC debug.cc)
add_library(nnadapter_utility_utility STATIC utility.cc)
add_library(nnadapter_utility_modeling STATIC modeling.cc)
add_library(nnadapter_utility_cache STATIC cache.cc)
add_library(nnadapter_utility_hints STATIC hints.cc)
add_library(nnadapter_utility_any STATIC any.cc)
add_library(nnadapter_utility_pattern_matcher STATIC pattern_matcher.cc)
add_library(nnadapter_utility_graph STATIC graph.cc)
add_library(nnadapter_utility_node STATIC node.cc)
add_library(nnadapter_utility_fuse_base STATIC fuse_base.cc)

set(NNADAPTER_UTILITIES nnadapter_utility_fuse_base nnadapter_utility_node nnadapter_utility_graph nnadapter_utility_pattern_matcher nnadapter_utility_logging nnadapter_utility_string nnadapter_utility_debug nnadapter_utility_utility nnadapter_utility_cache nnadapter_utility_modeling nnadapter_utility_hints nnadapter_utility_any CACHE INTERNAL "")
