#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
A separate Fluid test file for feeding specific data.
'''

import argparse
import sys
import numpy as np
import os
import paddle.fluid as fluid
from paddle.fluid import debugger
from paddle.fluid import core
import subprocess

GLB_model_path = ''
GLB_out_txt = ''
GLB_arg_name = ''
GLB_batch_size = 1
GLB_txt_path = ''

def load_inference_model(model_path, exe):
    '''
    '''
    model_abs_path = os.path.join(model_path, 'model')
    param_abs_path = os.path.join(model_path, 'params')
    if os.path.exists(model_abs_path) and os.path.exists(param_abs_path):
        return fluid.io.load_inference_model(model_path, exe, 'model', 'params')
    else:
        return fluid.io.load_inference_model(model_path, exe)

def feed_ones(block, feed_target_names, batch_size=1):
    """ 
    """ 
    feed_dict = dict()
    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        return shape
    def fill_ones(var_name, batch_size):
        var = block.var(var_name)
        np_shape = set_batch_size(list(var.shape), 1)
        print('shape: ', np_shape)
        var_np = {
            core.VarDesc.VarType.BOOL: np.bool_,
            core.VarDesc.VarType.INT32: np.int32,
            core.VarDesc.VarType.INT64: np.int64,
            core.VarDesc.VarType.FP16: np.float16,
            core.VarDesc.VarType.FP32: np.float32,
            core.VarDesc.VarType.FP64: np.float64,
        }
        np_dtype = var_np[var.dtype]
        return np.ones(np_shape, dtype=np_dtype)
    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_ones(feed_target_name, batch_size)

    return feed_dict


def feed_randn(block, feed_target_names, batch_size=1, need_save=True):
    """ 
    """ 
    feed_dict = dict()
    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        return shape
    def fill_randn(var_name, batch_size, need_save):
        var = block.var(var_name)
        np_shape = set_batch_size(list(var.shape), 1)
        var_np = {
            core.VarDesc.VarType.BOOL: np.bool_,
            core.VarDesc.VarType.INT32: np.int32,
            core.VarDesc.VarType.INT64: np.int64,
            core.VarDesc.VarType.FP16: np.float16,
            core.VarDesc.VarType.FP32: np.float32,
            core.VarDesc.VarType.FP64: np.float64,
        }
        np_dtype = var_np[var.dtype]
        numpy_array = np.random.random(np_shape).astype(np.float32)
        if need_save is True:
            numpy_to_txt(numpy_array, 'feed_' + var_name + '.txt', True)
        return numpy_array
    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_randn(feed_target_name, batch_size, need_save)
    return feed_dict

def feed_txt(block, feed_target_names, batch_size=1, need_save=True):
    """
    """
    feed_dict = dict()
    def set_batch_size(shape, batch_size):
        if shape[0] == -1:
            shape[0] = batch_size
        return shape
    def fill_txt(var_name, batch_size, need_save):
        var = block.var(var_name)
        np_shape = set_batch_size(list(var.shape), 1)
        # print('shape: ', np_shape)
	var_np = {
            core.VarDesc.VarType.BOOL: np.bool_,
            core.VarDesc.VarType.INT32: np.int32,
            core.VarDesc.VarType.INT64: np.int64,
            core.VarDesc.VarType.FP16: np.float16,
            core.VarDesc.VarType.FP32: np.float32,
            core.VarDesc.VarType.FP64: np.float64,
        }
        np_dtype = var_np[var.dtype]
        fp_r = open(GLB_txt_path, 'r')
        data = fp_r.readline()
        arry = []
        while data:
            #print('data: ', data.strip())
            arry.append(float(data.strip()))
            data = fp_r.readline()
        fp_r.close()
        numpy_array = np.array(arry).reshape(np_shape).astype(np.float32) #np.random.random(np_shape).astype(np.float32)
        if need_save is True:
            numpy_to_txt(numpy_array, 'feed_' + var_name + '.txt', True)
        return numpy_array
    for feed_target_name in feed_target_names:
        feed_dict[feed_target_name] = fill_txt(feed_target_name, batch_size, need_save)
    return feed_dict

def draw(block, filename='debug'):
    """
    """
    dot_path = './' + filename + '.dot'
    pdf_path = './' + filename + '.pdf'
    debugger.draw_block_graphviz(block, path=dot_path)
    cmd = ["dot", "-Tpdf", dot_path, "-o", pdf_path]
    subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def fetch_tmp_vars(block, fetch_targets, var_names_list=None):
    """
    """
    def var_names_of_fetch(fetch_targets):
        var_names_list = []
        for var in fetch_targets:
            var_names_list.append(var.name)
        return var_names_list

    fetch_var = block.var('fetch')
    old_fetch_names = var_names_of_fetch(fetch_targets)
    new_fetch_vars = []
    for var_name in old_fetch_names:
        var = block.var(var_name)
        new_fetch_vars.append(var)
    i = len(new_fetch_vars)
    if var_names_list is None:
        var_names_list = block.vars.keys()
    for var_name in var_names_list:
        if var_name != '' and var_name not in old_fetch_names:
            var = block.var(var_name)
            new_fetch_vars.append(var)
            block.append_op(
                type='fetch',
                inputs={'X': [var_name]},
                outputs={'Out': [fetch_var]},
                attrs={'col': i})
            i = i + 1
    return new_fetch_vars


def numpy_var(scope, var_name):
    """
    get numpy data by the name of var.
    """
    if hasattr(fluid.executor, '_fetch_var'):
        numpy_array = fluid.executor._fetch_var(var_name, scope, True)
    elif hasattr(fluid.executor, 'fetch_var'):
        numpy_array = fluid.executor.fetch_var(var_name, scope, True)
    else:
        raise NameError('ERROR: Unknown Fluid version.')
    return numpy_array


def var_dtype(block, var_name):
    """
    get dtype of fluid var.
    """
    var = block.var(var_name)
    return var.dtype


def print_ops_type(block):
    """
    """
    def ops_type(block):
        ops = list(block.ops)
        cache = []
        for op in ops:
            if op.type not in cache:
                cache.append(op.type)
        return cache
    type_cache = ops_type(block)
    print 'type: '
    for op_type in type_cache:
        print op_type


def print_results(results, fetch_targets, need_save=False):
    """
    """
    for result in results:
        idx = results.index(result)
        print fetch_targets[idx]
        A = np.array(result)
        print A, 'mean={}'.format(A.flatten().mean())
	print 'sum={}'.format(A.flatten().sum())
        if need_save is True:
            numpy_to_txt(result, 'result_' + fetch_targets[idx].name.replace('/', '_'), True)

def print_results_last(results, fetch_targets, need_save=False):
    """
    """
    for result in results:
        idx = results.index(result)
        print fetch_targets[idx]
        A = np.array(result)
        print A, 'mean={}'.format(A.flatten().mean())
        if need_save is True:
            numpy_to_txt(result, GLB_out_txt, True)
            break


def numpy_to_txt(numpy_array, save_name, print_shape=True):
    """
    transform numpy to txt.
    """
    np_array = np.array(numpy_array)
    fluid_fetch_list = list(np_array.flatten())
    fetch_txt_fp = open(save_name, 'w')
    for num in fluid_fetch_list:
        fetch_txt_fp.write(str(num) + '\n')
    if print_shape is True:
        fetch_txt_fp.write('Shape: (')
        for val in np_array.shape:
            fetch_txt_fp.write(str(val) + ', ')
        fetch_txt_fp.write(')\n')
    fetch_txt_fp.close()


def fluid_inference_test(model_path):
    """
    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        [net_program, 
        feed_target_names, 
        fetch_targets] = load_inference_model(model_path, exe)
        global_block = net_program.global_block()
        draw(global_block)
        #feed_list = feed_ones(global_block, feed_target_names, 1)
	print('GLB_txt_path: ', GLB_txt_path)
	if GLB_txt_path is None or GLB_txt_path == "":
		feed_list = feed_ones(global_block, feed_target_names, 1)
	else:
        	feed_list = feed_txt(global_block, feed_target_names, 1)
	#feed_list = feed_randn(global_block, feed_target_names, 1, need_save=True)
        fetch_targets = fetch_tmp_vars(global_block, fetch_targets, [GLB_arg_name])
        results = exe.run(program=net_program,
                          feed=feed_list,
                          fetch_list=fetch_targets,
                          return_numpy=False)
        # print_results(results, fetch_targets, need_save=True)
        print_results_last(results, fetch_targets, need_save=True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser('fluid feed_ones')

    arg_parser.add_argument('--model_path', type=str, required=True)
    arg_parser.add_argument('--out_txt', type=str) #, required=True
    arg_parser.add_argument('--arg_name', type=str)
    arg_parser.add_argument('--batch_size', type=int)
    arg_parser.add_argument('--txt_path', type=str)
    args = arg_parser.parse_args()

    GLB_model_path = args.model_path
    GLB_out_txt = args.out_txt
    if args.arg_name is not None:
        GLB_arg_name = args.arg_name
    if args.batch_size is not None:
        GLB_batch_size = args.batch_size
    if args.txt_path is not None:
	    GLB_txt_path = args.txt_path
    # print('GLB_model_path: ', GLB_model_path)
    # print('GLB_out_txt: ', GLB_out_txt)
    fluid_inference_test(GLB_model_path)
