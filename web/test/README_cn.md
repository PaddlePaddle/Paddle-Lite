# PaddleJS 单元测试

百度 PaddleJS 的单元和功能测试可以在本部分实现。

## 基本用法

执行 npm run testunits 可以指定目标算子执行，根据输入和计算输出的测试用例判断算子执行正确性。

```bash
cd web                        # 进入根目录
npm i                         # 安装依赖
mkdir dist                    # 创建资源目录
cd dist                       # 进入资源目录
git clone testnuits           # 获取模型
mv testnuits dist             # 移动单元测试数据移动到指定地点
cd ..                         # 返回根目录
npm run testunits             # 启动 testunits 单元测试

```


## 浏览器覆盖面

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser
