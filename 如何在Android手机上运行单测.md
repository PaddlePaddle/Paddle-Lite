
运行run_on_android.sh
1. 到tools目录下执行编译脚本 选择对应的平台目标和网络类型(可不选)  
```
cd tools
sh build.sh android googlenet
```
 
2. 到scripts目录下，执行run_on_android.sh，将模型文件及运行单测所需要的二进制文件和paddle-mobile动态库push到手机中，可执行文件在data/local/tmp/bin目录下  
```
cd android-debug-script
sh run_on_android.sh (npm) 参数npm选择是否传输模型文件到手机上，第二次可以加上npm参数
```
    
出现:  
...     
test-softmax      
test-squeezenet      
test-transpose-op      
test-yolo     
**** choose OP or NET to test ****     
which to test :  
输入想要test的op或net       
which to test : test-googlenet

3.显示返回结果

注意:如果需要删除可进入adb shell 手动删除/data/local/tmp下所有文件夹，重新run_on_android.sh