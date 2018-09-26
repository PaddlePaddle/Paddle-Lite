# 模型量化脚本

#### 量化脚本使用指南
1. 在PaddleMobile项目目录下（如 ~/PaddleProject/paddle-mobile）

2. cd到  tools/quantification/ 目录

3. cmake编译

    ``` sh
    cmake .
    make
    ```

4. 运行量化脚本
    ```sh
    ./quantify (0:seperated. 1:combined ) (输入路径) (输出路径)
    # quantify googlenet seperated   from  /Users/xiebaiyuan/PaddleProject/quali/models/googlenet to ./googlenet_min
    ./quantify 0 /Users/xiebaiyuan/PaddleProject/quali/models/googlenet ./googlenet_min 

    ```

*注:*
*量化工具中*
*1.seperated模型model文件默认命名为 "__model__";*
*2.combined模型的model文件默认命名为 "model",参数文件默认命名为"params";*

    
##### 整体如下:
以googlenet非combined为例：

```sh
cd tools/quantification/
cmake .
make
./quantify 0 /Users/xiebaiyuan/PaddleProject/quali/models/googlenet ./googlenet_min
```


