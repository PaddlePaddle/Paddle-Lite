# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from auto_scan_base import IgnoreReasonsBase
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--target")
args = parser.parse_args()

if args.target == "ARM" or args.target == "OpenCL":
    from auto_scan_test_rpc import AutoScanTest
    from auto_scan_test_rpc import FusePassAutoScanTest
else:
    from auto_scan_test_no_rpc import AutoScanTest
    from auto_scan_test_no_rpc import FusePassAutoScanTest

IgnoreReasons = IgnoreReasonsBase
AutoScanTest = AutoScanTest
FusePassAutoScanTest = FusePassAutoScanTest
