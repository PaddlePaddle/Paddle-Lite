# -*- coding: utf-8 -*

import os
import sys
import math
import subprocess
import numpy as np
import paddle.fluid as fluid

def sh(command):
    pipe = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return pipe.stdout.read().decode("utf-8")

for fold in range(100, 1001, 100):
    print("checking fold : {}".format(fold))
    max_entropy = sh("./quantify 1 model params {}".format(fold))
    print("max entropy :", max_entropy, end="")
    sh("rm -rf scripts/model")
    sh("rm -rf scripts/quantification_model")
    sh("cp -r model scripts/model")
    sh("cp -r model scripts/quantification_model")
    sh("mv params scripts/quantification_model")
    diff = sh("cd scripts && python run.py {}".format(fold))
    print("output diff :", diff, end="")
