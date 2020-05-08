# 开发基础须知

可以参考 [Paddle 开发者文档](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.5/advanced_usage/development/contribute_to_paddle/local_dev_guide.html)。

## 提交PR

需要在 commit message 里加上 `test=develop` 才能触发 CI

## 版本发布检查清单

1. 所有 feature 梳理，确认状态
2. 所有 QA 测试结果梳理，确认版本可靠
3. Release note 确认 review 通过
4. 确认需要 release 的 binary 编译完毕
