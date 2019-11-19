import sys
import os
import math
import numpy as np
def diff_value(lite_path, fluid_path):
	""""""
	fp_lite = open(lite_path, 'r')
	fp_fluid = open(fluid_path, 'r')
	if (not fp_lite):
		print('open fp_lite file failed!', lite_path)
		return 
	
	if (not fp_fluid):
		print('open fp_fluid file failed!', fluid_path)
		return 

	data_lite = fp_lite.readline()
	data_fluid = fp_fluid.readline()
	lite_arr = []
	fluid_arr = []
	while data_lite:
		lite_arr.append(float(data_lite.strip()))
		fluid_arr.append(float(data_fluid.strip()))
		data_lite = fp_lite.readline()
		data_fluid = fp_fluid.readline()
	fp_lite.close()
	fp_fluid.close()

	#print
	# print("data_lite: ", lite_arr);
	# print("data_fluid: ", fluid_arr);
	#compare
	res = ratio_vector(np.array(lite_arr), np.array(fluid_arr))
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


def diff_fluid(device_id, model_lists, input_shapes, threads, arm_abi):
	model_root = '/home/chenjiao04/vis_shoubai/'
	i = 0
	res_root = '/home/chenjiao04/vis_shoubai/Paddle-Lite/run_model_py'
	fail_path = '/home/chenjiao04/vis_shoubai/Paddle-Lite/run_model_py/failed.txt'
	faile_wr = open(fail_path, 'a+')
	for name in model_lists:
		print('model name: ', name)
		if name == 'int8':
			i = i + 1
			continue
		if name == 'ar_cup_detection_int8':
			i = i + 2
			continue
			model_path = model_root + '/' + name + '/detection'
			fluid_path = res_root + '/fluid_' + name + '_detection' + '.txt'
			lite_path = 'lite_' + name + '_detection' + '.txt'
			model_name = name + '/detection'
			print('lite run')
			lite_res = os.system('sh ./run_model.sh %s %s %s %s %d %d %d %d' % (device_id, model_name, lite_path, input_shapes[i], 10, 50, threads, arm_abi))
			if (lite_res != 0):
				print('lite run error')
				return

			print('fluid run')
			fluid_res = os.system('python ./feed_ones.py --model_path={} --out_txt={}'.format(model_path, fluid_path))
			if (fluid_res != 0):
				print('fluid run error')
				return

			print('diff: ')
			res = diff_value(lite_path, fluid_path)
			if (res > 1e-5):
				faile_wr.write('model name: {}, arm_abi: {}, threads: {} res: {} \n'.format(model_name, arm_abi, threads, res))
				print('++++COMPUTE ERROR: model_name = ', model_name)
				print('res: ', res)
			else:
				print('++++COMPUTE SUCCESS')

			i = i + 1

			model_path = model_root + '/' + name + '/track'
			fluid_path = res_root + '/fluid_' + name + '_track' + '.txt'
			lite_path = 'lite_' + name + '_track' + '.txt'
			name = name + '/detection'
			
		elif(name == 'handkeypoints'):
			model_path = model_root + '/' + name + '/kpt_model_detection'
			fluid_path = res_root + '/fluid_' + name + '_kpt_model_detection' + '.txt'
			lite_path = 'lite_' + name + '_kpt_model_detection' + '.txt'
			model_name = name + '/kpt_model_detection'
			print('lite run')
			lite_res = os.system('sh ./run_model.sh %s %s %s %s %d %d %d %d' % (device_id, model_name, lite_path, input_shapes[i], 10, 50, threads, arm_abi))
			if (lite_res != 0):
				print('lite run error')
				return

			print('fluid run')
			fluid_res = os.system('python ./feed_ones.py --model_path={} --out_txt={}'.format(model_path, fluid_path))
			if (fluid_res != 0):
				print('fluid run error')
				return

			print('diff: ')
			res = diff_value(lite_path, fluid_path)
			if (res > 1e-5):
				faile_wr.write('model name: {}, arm_abi: {}, threads: {} res: {} \n'.format(model_name, arm_abi, threads, res))
				print('++++COMPUTE ERROR: model_name = ', model_name)
				print('res: ', res)
			else:
				print('++++COMPUTE SUCCESS')

			i = i + 1

			model_path = model_root + '/' + name + '/kpt_model_keypoints'
			fluid_path = res_root + '/fluid_' + name + '_kpt_model_keypoints' + '.txt'
			lite_path = 'lite_' + name + '_kpt_model_keypoints' + '.txt'
			name = name + '/kpt_model_keypoints'
		
		else:
			model_path = model_root + '/' + name
			fluid_path = res_root + '/fluid_' + name + '.txt'
			lite_path = 'lite_' + name + '.txt'
		print('lite run')
		# print('name: ', name)
		# print('lite_path: ', lite_path)
		# print('input_shapes: ', input_shapes[i])
		# print('model_path: ', model_path)
		# print('fluid_path: ', fluid_path)
		lite_res = os.system('sh ./run_model.sh %s %s %s %s %d %d %d %d' % (device_id, name, lite_path, input_shapes[i], 10, 50, threads, arm_abi))
		if (lite_res != 0):
			print('lite run error')
			return

		print('fluid run')
		fluid_res = os.system('python ./feed_ones.py --model_path={} --out_txt={}'.format(model_path, fluid_path))
		if (fluid_res != 0):
			print('fluid run error')
			return

		print('diff: ')
		res = diff_value(lite_path, fluid_path)
		if (res > 1e-5):
			faile_wr.write('model name: {},  arm_abi: {}, threads: {}, res: {} \n'.format(name,  arm_abi, threads, res))
			print('++++COMPUTE ERROR: model_name = ', name)
			print('res: ', res)
			# return
		else:
			print('++++COMPUTE SUCCESS')
		
		i = i + 1
	faile_wr.close()
		

device_id = ['17c3cc34', '5380268d', '7f1446bd']
model_lists = ['automl_mv3_5ms_64_s_ftdongxiao_shoubai', 'eye_mv1s_infer', 
				'handkeypoints', 'int8', 'models_0158', 'mouth_mv6_epoch320_shoubai', 'mv3_gp_shoubai',
				'mv8_angle_shoubai', 'skyseg_shufflenet_0520_160', 'ar_cup_detection_int8']
input_shapes = ['1,3,64,64', '1,3,24,24', '1,3,224,224', '1,3,192,192',
				'1,3,512,512', '1,3,224,128', '1,3,48,48', '1,3,128,128', '1,3,64,64', '1,3,160,160',
				'1,3,192,192', '1,3,128,128']

if __name__ == '__main__':
	# res = os.system('sh ./run_init.sh ')
	# if (res != 0):
	# 	print('run_init.sh error')
	# 	return
	# clean data
	os.system('rm fluid_*.txt')
	os.system('rm lite_*.txt')
	os.system('rm time_*.txt')
	for de in device_id:
		print('init: ', de)
		res = os.system('sh ./run_push.sh ' + de)
		if (res != 0):
			print('run_push.sh error')
			break
		for arm_abi in [0, 1]:
			# rm time
			os.system('adb -s {} shell rm /data/local/tmp/lite/vis_shoubai/time.txt'.format(de))
			arm = 'v7'
			if arm_abi == 1:
				arm = 'v8'
			for num in [1, 2, 4]:
				diff_fluid(de, model_lists, input_shapes, num, arm_abi)
				os.system('adb -s {} pull /data/local/tmp/lite/vis_shoubai/time.txt ./'.format(de))
				time_de = "time_" + de + '_' + arm + ".txt"
				os.system('mv ./time.txt {}'.format(time_de))
		

