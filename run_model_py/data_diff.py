import sys
import os
import math
import numpy as np
def diff_value(lite_path, fluid_path):
    fp_lite = open(lite_path, 'r')
    fp_fluid = open(fluid_path, 'r')
    if (not fp_lite):
        print('open fp_lite file failed!', lite_path)
        return 
    
    if (not fp_fluid):
        print('open fp_fluid file failed!', fluid_path)
        return 

    data_lite = fp_lite.readline()
    lite_arr = []
    fluid_arr = []
    while data_lite:
        line = data_lite.strip().split('  ')
        # print('line: ', line)
        for val in line:
            lite_arr.append(float(val))
        
        data_lite = fp_lite.readline()
    n = len(lite_arr)   
    # print('n: ', n) 
    data_fluid = fp_fluid.readline()
    for i in range(n):
        # print('i: ', i, data_fluid)
        fluid_arr.append(float(data_fluid.strip()))
        data_fluid = fp_fluid.readline()
    

	fp_lite.close()

    res = ratio_vector(np.array(lite_arr), np.array(fluid_arr))
    print('res: ', res)
    fp_fluid.close()
    return res

def vector_length(arr):
    """compute a np array vector size"""
    return math.sqrt(sum(np.square(arr)))

def ratio_vector(target, base):
    """compute ratio of 2 vector's length"""
    base_length = vector_length(base)
    if base_length != 0:
        return vector_length(target - base)/base_length
    else:
        return 0

if __name__ == '__main__':
    diff_value(sys.argv[1], sys.argv[2])