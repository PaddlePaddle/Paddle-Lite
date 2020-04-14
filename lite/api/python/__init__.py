# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os 
import sys

if os.name =='nt':
    current_path = os.path.abspath(os.path.dirname(__file__))
    third_lib_path = current_path + os.sep + 'libs'
    os.environ['path'] =  third_lib_path+ ';' + os.environ['path']
    sys.path.insert(0, third_lib_path)
