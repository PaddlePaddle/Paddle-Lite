import sys
import os
import math
import time
import numpy as np
def profile_test(device_id, model_lists, input_shapes, threads, arm_abi):
   i = 0
   res_root = '/home/chenjiao04/vis_shoubai/Paddle-Lite/run_model_py'
   fail_path = '/home/chenjiao04/vis_shoubai/Paddle-Lite/run_model_py/failed.txt'
   for name in model_lists:
       print('model: ', name)
       #model_dir = name + '/opt2'
       model_dir = name
       lite_res = os.system('sh ./run.sh %s %s %s %s %d %d %d %d' % (device_id, model_dir, 'b', input_shapes[i], 10, 50, threads, arm_abi))
       if (lite_res != 0):
	   print('lite run error')
	   return
       i = i + 1
       time.sleep(10)

def mnn_test(device_id, model_lists, input_shape, threads, arm_abi):
    i = 0
    for name in model_lists:
        print('model: ', name)
        model_dir = name
        lite_res = os.system('sh ./run_mnn.sh %s %s %d %d %d %s' % (device_id, model_dir, 50, threads, arm_abi, input_shape[i]))
        if (lite_res != 0):
		print('lite run error')
		return
        i = i + 1
        time.sleep(10)
device_id = ['17c3cc34', '7f1446bd'] 
#device_id = ['17c3cc34']
model_lists = ['mobilenetv1', 'mobilenetv2', 'mnasnet']
#model_lists = ['tf2mnn_mnasnet-a1.pb.mnn', 'tf2mnn_mobilenet_v2_1.4_224_frozen.pb.mnn', 'tf2mnn_mobilenet_v1_1.0_224_frozen.pb.mnn']
input_shapes = ['1,3,224,224', '1,3,224,224', '1,3,224,224']
#input_shapes = ['', '1x3x224x224', '']
if __name__ == '__main__':
	os.system('rm time_*.txt')
	for de in device_id:
		print('init: ', de)
		res = os.system('sh ./push.sh ' + de)
		if (res != 0):
			print('run push.sh error')
		for arm_abi in [0, 1]:
			os.system('adb -s {} shell rm /data/local/tmp/lite/profile/time.txt'.format(de))
			arm = 'v7'
			if arm_abi == 1:
				arm = 'v8'
			for num in [1, 2, 4]:
				profile_test(de, model_lists, input_shapes, num, arm_abi)
				#mnn_test(de, model_lists, input_shapes, num, arm_abi)
				os.system('adb -s {} pull /data/local/tmp/lite/profile/time.txt ./'.format(de))
				time_de = "time_" + de + '_' + arm + ".txt"
				os.system('mv ./time.txt {}'.format(time_de))
		


