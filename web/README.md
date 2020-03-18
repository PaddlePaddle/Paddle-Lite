[中文版](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/web/README_cn.md)

# Web

Paddle.js is an Web project for Baidu Paddle, which is an an open source deep learning framework designed to work on web browser. Load a pretrained paddle.js SavedModel or Paddle Hub module into the browser and run inference through Paddle.js. It could run on nearly every browser with WebGL support.

## Key Features

### Modular

Web project is built on Atom system which is a versatile framework to support GPGPU operation on WebGL. It is quite modular and could be used to make computation tasks faster by utilizing WebGL.

### High Performance

Web project could run TinyYolo model in less than 30ms on chrome. This is fast enough to run deep learning models in many realtime scenarios.

### Browser Coverage

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser

### Supported operations

Currently Paddle.js only supports a limited set of Paddle Ops. See the full list. If your model uses unsupported ops, the Paddle.js script will fail and produce a list of the unsupported ops in your model. Please file issues to let us know what ops you need support with.

[Supported operations Pages](./src/factory/fshader/README.md)


## Loading and running in the browser

If the original model was a SavedModel, use paddle.load(). 

```bash

	import * as tf from 'paddlejs';


	let feed = io.process({
        input: document.getElementById('image'),
        params: {
            gapFillWith: '#000', // What to use to fill the square part after zooming
            targetSize: {
                height: fw,
                width: fh
            },
            targetShape: [1, 3, fh, fw], // Target shape changed its name to be compatible with previous logic
            // shape: [3, 608, 608], // Preset sensor shape
            mean: [117.001, 114.697, 97.404], // Preset mean
            // std: [0.229, 0.224, 0.225]  // Preset std
        }
    });

	const MODEL_CONFIG = {
        dir: `/${path}/`, // model URL
        main: 'model.json', // main graph
    };
	
	const paddle = new Paddle({
        urlConf: MODEL_CONFIG,
        options: {
            multipart: true,
            dataType: 'binary',
            options: {
                fileCount: 1, // How many model have been cut
                getFileName(i) { 
                    return 'chunk_' + i + '.dat';
                }
            }
        }
    });

    model = await paddle.load();

    // 
	let inst = model.execute({
        input: feed
    });

    // There should be a fetch execution call or a fetch output
    let result = await inst.read();


```

Please see feed documentation for details.

Please see fetch documentation for details.


## Run the converter script provided by the pip package:

The converter expects a Paddlejs SavedModel, Paddle Hub module, Tpaddle.js JSON format for input.


## Web-friendly format

The conversion script above produces 2 types of files:

 - model.json (the dataflow graph and weight manifest file)
 - group1-shard\*of\* (collection of binary weight files)


## Preview Demo

Paddle.js has some pre-converted models to Paddle.js format .There are some demos in the following URL, open a browser page with the demo.

[Supported Demo Pages](./examples/README.md)


## Feedback and Community Support

- Questions, reports, and suggestions are welcome through Github Issues!
- Forum: Opinions and questions are welcome at our [PaddlePaddle Forum](https://ai.baidu.com/forum/topic/list/168)！
- QQ group chat: 696965088
