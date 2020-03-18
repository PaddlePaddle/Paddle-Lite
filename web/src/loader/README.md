[中文版](./README_cn.md)
# PaddleJS odel loader

Baidu paddlejs uses this loader  to get the model to the browser. The model loader can load browser friendly JSON file types and binary file types, supports single file loading and file fragment loading, and greatly uses the characteristics of browser parallel request to load reasoning model.

## Demonstration

Create the paddy object, specify the add model address, add configuration parameters, and load the model through the load method.

## Parameter description


| 表格      | 参数    | 描述     |
| ------------- | ------------- | ------------- |
| MODEL_ADDRESS   |  dir    | 存放模型的文件夹 |
| MODEL_ADDRESS    | main     | 主文件     |
| options    | multipart     | 是否分片获取 |
| options    | dataType    | binary/json   |
| options    | fileCount     | 分片数量     |
| options    | ietest     | 是否开启测试输出 |




```bash

	const MODEL_ADDRESS = {
	    dir: `/${path}/`, // 存放模型的文件夹
	    main: 'model.json', // 主文件
	};

	const paddle = new Paddle({
		urlConf: MODEL_ADDRESS,
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





## Browser coverage

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser


