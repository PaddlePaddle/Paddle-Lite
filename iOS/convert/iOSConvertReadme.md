this readme specify how to convert the caffemodel to iOS gpu format

iOS GPU in mdl need the json from caffe2mdl, and need the parameter from the conert.py in   
subpath of .../baidu/mms/mobile-deep-learning/iOS/convert
for example, need to convert the squeeze net:

这个 readme 是用来说明怎样获取 mdl iOS GPU 库 所需的参数
mdl iOS GPU库 所需的json文件需要通过 caffe2mdl 获取, 所需的参数文件则需要 .../baidu/mms/mobile-deep-learning/iOS/convert 目录下 子目录中的 convert.py 脚本,
例如需要转换squeeze net:

```
#get the json form caffe2mdl

./build.sh mac
cd ./build/release/x86/tools/build

# copy your model.prototxt and model.caffemodel to this path
# also need the input data

./caffe2mdl model.prototxt model.caffemodel data

#after this you can get the model.json
# now let's get the parameter

cd iOS/convert/SqueezeNet

python convert.py

cd Parameters/

# now you get the parameters
# copy the json and the parameters to your project

```


if you want to get the json and parameters of the MobileNet which has batch normal and scale layer, there is some difference you need to do

如果你想转换 含有 batch normal 和 scale 层的 mobile net, 你需要做的会有一些不同:

```
#get the json from caffe2mdl

./build.sh mac
cd ./build/release/x86/tools/build

# copy your model.prototxt and model.caffemodel to this path
# also need the input data
./caffe2mdl model.prototxt model.caffemodel data -ios_gpu_mobilenet_classify

# if the model is not for classification, you need to do:
# ./caffe2mdl model.prototxt model.caffemodel data -ios_gpu_mobilenet instead the above command

# after this you can get the model.json
# now let's get the parameter

cd iOS/convert/MobileNet

python convert.py

cd Parameters/

# now you get the parameters
# copy the json and the parameters to your project

```



