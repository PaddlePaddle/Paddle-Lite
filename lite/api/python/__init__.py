import os
from ctypes import *
from lite_core.so import *
lib_path = os.path.join(os.path.dirname(__file__), 'lite.so')
lib = CDLL(lib_path)
