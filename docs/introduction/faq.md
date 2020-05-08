# FAQ 常见问题

问题或建议可以发Issue，为加快问题解决效率，可先检索是否有类似问题，我们也会及时解答！
欢迎加入Paddle-Lite百度官方QQ群：696965088

1. 在Host端采用交叉编译方式编译PaddleLite，将编译后的libpaddle_light_api_shared.so和可执行程序放到板卡上运行，出现了如下图所示的错误，怎么解决？ 
![host_target_compiling_env_miss_matched](https://user-images.githubusercontent.com/9973393/75761527-31b8b700-5d74-11ea-8a9a-0bc0253ee003.png)
- 原因是Host端的交叉编译环境与Target端板卡的运行环境不一致，导致libpaddle_light_api_shared.so链接的GLIBC库高于板卡环境的GLIBC库。目前有四种解决办法（为了保证编译环境与官方一致，推荐第一种方式）：1）在Host端，参考[源码编译](../user_guides/source_compile)中的Docker方式重新编译libpaddle_light_api_shared.so；2）在Host端，使用与Target端版本一致的ARM GCC和GLIBC库重新编译libpaddle_light_api_shared.so；3）在Target端板卡上，参考[源码编译](../user_guides/source_compile)中的ARM Linux本地编译方式重新编译libpaddle_light_api_shared.so；4）在Target端板卡上，将GLIBC库升级到和Host端一致的版本，即GLIBC2.27。
