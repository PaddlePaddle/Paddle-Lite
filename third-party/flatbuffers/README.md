## How can we update flatbuffer head files in Paddle-Lite?
### Step1: update fbs file
``` shell
lite/model_parser/flatbuffers/framework.fbs
lite/model_parser/flatbuffers/param.fbs
lite/backends/opencl/utils/cache.fbs
lite/backends/opencl/utils/tune_cache.fbs
```
### Step2. update flatbuffer pre-build results
```shell
# Scripts below can help update flatbuffers head files according to 
# framework.fbs and param.fbs automatically
third-party/flatbuffer/update_fbs.sh
```
### Step2. push your update to the remote github repo
```shell
git add -f third-party/flatbuffers/pre-build
git commit -m"your commit message"
git push
```
