import numpy as np
import paddle.fluid as fluid

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

data = np.array([5.0])
x = fluid.layers.data(name="x", shape=[1], dtype="float32")
y = fluid.layers.relu6(x, threshold=4.0)

prog = fluid.default_main_program()
outputs = exe.run(prog, feed={"x": data}, fetch_list=[y])
print(outputs)
