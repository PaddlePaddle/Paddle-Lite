[中文版](https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/web/README_cn.md)

# Web

Web project is an open source deep learning framework designed to work on web browser. It could run on nearly every browser with WebGL support.

## Key Features

### Modular

Web project is built on Atom system which is a versatile framework to support GPGPU operation on WebGL. It is quite modular and could be used to make computation tasks faster by utilizing WebGL.

### High Performance

Web project could run TinyYolo model in less than 30ms on chrome. This is fast enough to run deep learning models in many realtime scenarios.

### Browser Coverage

* PC: Chrome
* Mac: Chrome
* Android: Baidu App and QQ Browser

## How To Build & Deploy Demo

```bash
cd web                        # enter root directory for web project
npm i                         # install dependencies for npm
mkdir dist                    # create deployment directory
cd dist                       # enter deployment directory
git clone https://github.com/DerekYangMing/Paddle-Web-Models.git # get models
mv Paddle-Web-Models/separablemodel .                            # move models to specific directory
cd ..                         # return to root directory for web project
npm run testVideoDemo         # start demo
```

## How To Preview Demo

1. Open chrome with url: https://localhost:8123/
2. Start demo by pressing the 【start detection】 button.
3. Ensure at least one face is recorded by the camera. The face detection rectangle should be displayed if everything goes fine.

## Feedback and Community Support

- Questions, reports, and suggestions are welcome through Github Issues!
- Forum: Opinions and questions are welcome at our [PaddlePaddle Forum](https://ai.baidu.com/forum/topic/list/168)！
- QQ group chat: 696965088
