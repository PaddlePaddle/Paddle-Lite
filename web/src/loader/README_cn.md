# PaddleJS Examples

百度 PaddleJS 的使用这个加载器进行模型获取到浏览器。模型加载器可以加载浏览器友好的json文件类型和二进制文件类型，支持单文件加载和文件分片加载，极大的利用浏览器并行请求的特性加载推理模型。

## 使用方法

创建Paddle对象，指定加模型地址，添加配置参数，通过load方法加载模型。

## 参数说明


| 表格      | 参数    | 描述     |
| ------------- | ------------- | ------------- |
| MODEL_ADDRESS   |  dir    | 存放模型的文件夹 |
| MODEL_ADDRESS    | main     | 主文件     |
| options    | multipart     | 是否分片获取 |
| options    | dataType    | binary/json   |
| options    | fileCount     | 分片数量     |
| options    | ietest     | 是否开启测试输出 |



```bash
	const MODEL_CONFIG = {
	    dir: `/${path}/`, // 存放模型的文件夹
	    main: 'model.json', // 主文件
	};

	const paddle = new Paddle({
		urlConf: MODEL_CONFIG,
		options: {
		    multipart: true,
		    dataType: 'binary',
		    options: {
		        fileCount: n, // 切成了n文件
		        getFileName(i) { // 获取第i个文件的名称
		            return 'chunk_0.dat';
		        }
		    }
		}
	});

	model = await paddle.load();

```



## 浏览器覆盖面

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser