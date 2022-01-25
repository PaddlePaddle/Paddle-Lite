#!/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2014 Baidu.com, Inc. All Rights Reserved
#
################################################################################

import configparser
import os
import sys
import io
import math
import numpy as np


def get_model_list(cp):
    """
    get the model list from config.ini
    """
    #models= cp.options('models')
    models = cp.sections()
    return models


def get_inputshape_list(cp):
    """
    get the model inputshape list from config.ini
    """
    models = get_model_list(cp)
    shape_list = []
    for model_name in models:
        input_shape = cp.get(model_name, 'input_shape')
        shape_list.append(input_shape)
    return shape_list


def get_exclude_list(cp):
    """
    get the model exclude list from config.ini
    """
    models = get_model_list(cp)
    exclude_list = []
    for model_name in models:
        exclude = cp.get(model_name, 'exclude')
        exclude_list.append(exclude)
    return exclude_list


def vector_length(arr):
    """compute a np array vector size"""
    return math.sqrt(sum(np.square(arr)))


def ratio_vector(target, base):
    """compute ratio of 2 vector's length"""
    base_length = vector_length(base)
    if base_length != 0:
        return vector_length(target - base) / base_length
    else:
        return 0


def abs_diff(arrA, arrB):
    """compute absolute vector length difference"""
    abs_arr = np.absolute(np.absolute(arrA) - np.absolute(arrB))
    return abs_arr


def ratio_diff(arrA, arrB):
    """compute relative vector length difference"""
    rel_arr = np.zeros(arrA.shape)
    idxNoneZeros = np.where(arrB != 0)
    idxZeros = np.where(arrB == 0)
    rel_arr[idxNoneZeros] = np.absolute(
        np.absolute(arrA[idxNoneZeros]) / np.absolute(arrB[idxNoneZeros]))
    return rel_arr


def compare_output():
    """
    compare result with fluid 
    """
    lite_output_path = "/".join(os.getcwd().split('/')[:-2])
    arm_abi = ["armv8, armv7"]
    accuracy = ["fp32", "fp16", "int8"]
    rerr = 1e-5
    aerr = 1e-5
    for arm in arm_abi:
        for acc in accuracy:
            if acc == "fp32":
                rerr = 1e-5
                aerr = 1e-5
            elif acc == "fp16":
                rerr = 5e-2
                aerr = 5e-2
            elif acc == "int8":
                rerr = 1e-2
                aerr = 1
            lite_output_path_arm_acc = lite_output_path + "/output/" + arm + "/" + acc
            for root, dirs, files in os.walk(lite_output_path_arm_acc):
                for output_txt in files:
                    lite_txt = lite_output_path_arm_acc + "/" + output_txt
                    r = open(lite_txt, 'r')
                    lines = r.readlines()
                    r.close()
                    w = open(lite_txt, 'w')
                    w.truncate()
                    for line in lines:
                        if "out[" in line:
                            w.write(((line.split(":"))[1]))
                    w.close()
                    paddle_output_path = lite_output_path + "/airank_fluid_output" + "/fluid_output_fp32_all1"
                    paddle_txt = paddle_output_path + "/" + (
                        (output_txt.split("."))[0]) + "/" + output_txt
                    lite_results = list()
                    lite_results_tmp = list()
                    with open(lite_txt) as flite_result:
                        for data in flite_result.readlines():
                            if data.strip() == "nan" or data.strip() == "inf":
                                print(arm, acc, ((output_txt.split("."))[0]),
                                      " Output nan of inf")
                                return True
                            lite_results_tmp.append(float(data.strip()))
                    lite_results = np.array(lite_results_tmp)
                    paddle_results = list()
                    paddle_results_tmp = list()
                    with open(paddle_txt) as fpaddle_result:
                        for data in fpaddle_result.readlines():
                            paddle_results_tmp.append(float(data.strip()))
                    paddle_results = np.array(paddle_results_tmp)
                    if len(lite_results) != len(paddle_results):
                        print(arm, acc, ((output_txt.split("."))[0]),
                              " Outshape Diff")
                        return True
                    a = abs_diff(lite_results, paddle_results)
                    r = ratio_diff(lite_results - paddle_results,
                                   paddle_results)
                    for i in range(len(a)):
                        if (a[i] > aerr and r[i] > rerr) or (a[i] > aerr and
                                                             r[i] == 0):
                            print(arm, acc, ((output_txt.split("."))[0]),
                                  " Output Data Diff , Diff element:", i,
                                  " lite result:", lite_results[i],
                                  " paddle result:", paddle_results[i],
                                  " abolute error:", a[i], " relative error:",
                                  r[i])
                            return True
    return False


def main():
    """
    the start of this program
    """
    type = sys.argv[1]

    file_name = os.path.join(os.getcwd(), 'config.ini')
    cp = configparser.ConfigParser()
    cp.read(file_name)
    if (type == "modellist"):
        model_list = get_model_list(cp)
        model_str = ".".join(model_list)
        print(model_str)
    elif (type == "shapelist"):
        shape_list = get_inputshape_list(cp)
        shape_str = ".".join(shape_list)
        print(shape_str)
    elif (type == "excludelist"):
        exclude_list = get_exclude_list(cp)
        exclude_str = ".".join(exclude_list)
        print(exclude_str)
    elif (type == "cmp_diff"):
        out_diff = compare_output()
        assert out_diff != True, "model output data diff or output shape diff, please fix the precision error!"


if __name__ == "__main__":
    main()
