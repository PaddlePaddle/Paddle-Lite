## OS X Installation
We highly recommend using [HomeBrew](https://brew.sh/) as the package manager,and [CMake](https://cmake.org/) as the build tool, in the following, we assume you have installed HomeBrew and CMake already.

### General dependencies
**ProtoBuffer**  
Install ProtoBuffer via Homebrew:  
 
```
brew install protobuffer 
```

the caffe.pb.cc & caffe.pb.h in tools, are gernerated by proto3.4.0, if the version is incompatible with the protoc you just installed, you can regenerate the source code and replcace the caff.pb.cc and caffe.pb.h
Generate source code with protoc:  
 
```bash
cd tools

protoc --proto_path=. --cpp_out=. caffe.proto
```

**NDK**  

To install the Android NDK, simply expand the archive in the folder where you want to install it.

After installing the NDK, define an environment variable identifying the path to the NDK. For example,
on OS X, you might add the following to your .bash_profile:

```bash 
export NDK_ROOT=//path to your NDK
```

Make sure the environment variable name is 'NDK_ROOT', or the build.sh can't find NDK tools while compiling for Android.

### Model Converting
* To get the caffe2mdl binary file:  
build the tools dir alone:  

```bash
cd tools
cmake .   
make
```   

or You can build the whole project:  

```bash
sh build.sh mac
```  

the executable file will be created in the build tree directory corresponding to the source tree directory.
* To convert caffe model to MDL

`./caffe2mdl deploy.prototxt full.caffemodel` 

the third para is optional, if you want to test the model produced by this script, provide color value array of an image as the third parameter ,like this:

```bash
./caffe2mdl model.prototxt model.caffemodel data
```

* How to generate the data file 
 The data file is an plain text file, numbers are seperated with a space, the numbers is organized in the order of RGB,like this:

 ```
 RRRRRRRR…GGGGGGGGGG……BBBBBBBBBBB……
 ```  

 It should be noted that the color value has been preprocessed according to the model, take googlenet in our directory for example, each value of RGB array has been substracted by 148 (the mean value of the model).
 
### Test on Mac
 ```bash
 ./build.sh mac

  cd build/release/x86/build

 ./mdlTest
 ```

* For obect detection task, the result array indicates coordinate of the rect  
* For classification task, the result array indicates the probability of classification
 









  


