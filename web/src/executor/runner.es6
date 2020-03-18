/**
 * @file Runner 整个流程封装一下
 * @author hantian(hantianjiao@baidu.com)
 * 使用方法：
 * const runner = new Runner({
 *      modelName: 'separate' // '608' | '320' | '320fused' | 'separate'
 *  });
 *  runner.preheat().then(r => {
 *      r.run(document.getElementById('test'));
 *  });
 */
import IO from '../feed/ImageFeed';
import DataFeed from '../feed/dataFeed';
import Graph from './loader';
import PostProcess from './postProcess';
import models from '../utils/models';
import Logger from '../../tools/logger';
window.log = new Logger();

export default class Runner {
    // 加载模型&预热
    constructor(options) {
        this.modelConfig = models[options.modelName];
        this.flags = {
            isRunning: false,
            isPreheating: false,
            runVideoPaused: false
        };
        this.buffer = new Float32Array();
        this.io = new IO();
        this.postProcess = new PostProcess(options);
    }

    // 预热 用用空数据跑一遍
    async preheat() {
        this.flags.isPreheating = true;
        let {fh, fw} = this.modelConfig.feedShape;
        let path = this.modelConfig.modelPath;
        let feed = [{
            data: new Float32Array(3 * fh * fw),
            name: 'image',
            shape: [1, 3, fh, fw]
        }];
        const MODEL_URL = `/${path}/model.json`;
        let dir = `https://mms-graph.cdn.bcebos.com/activity/facegame/paddle/${path}/`;
        if (location.href.indexOf('test=1') > -1) {
            dir = `/src/view/common/lib/paddle/${path}/`;
        }
        const MODEL_CONFIG = {
            dir: dir,
            main: 'model.json' // 主文件
        };
        const graphModel = new Graph();
        this.model = await graphModel.loadGraphModel(MODEL_CONFIG, {
            multipart: true,
            dataType: 'binary',
            binaryOption: {
                fileCount: 1, // 切成了多少文件
                getFileName(i) { // 获取第i个文件的名称
                    return 'chunk_0.dat';
                }
            },
            feed
        });
        this.model.execute({
            input: feed
        });
        this.flags.isPreheating = false;
        return this;
    }

    // 跑一遍
    async run(input, callback) {
        this.flags.isRunning = true;
        let {fh, fw} = this.modelConfig.feedShape;
        let path = this.modelConfig.modelPath;
        if (!this.model) {
            console.warn('It\'s better to preheat the model before running.');
            await this.preheat();
        }
        // log.start('总耗时'); // eslint-disable-line
        // log.start('预处理'); // eslint-disable-line
        let feed;
        if (typeof input === 'string') {
            const dfIO = new DataFeed();
            feed = await dfIO.process({
                input: `/${path}/${input}`,
                shape: [1, 3, fh, fw]
            });
        }
        else {
            feed = this.io.process({
                input: input,
                params: {
                    gapFillWith: '#000', // 缩放后用什么填充不足方形部分
                    targetSize: {
                        height: fw,
                        width: fh
                    },
                    targetShape: [1, 3, fh, fw], // 目标形状 为了兼容之前的逻辑所以改个名
                    // shape: [3, 608, 608], // 预设tensor形状
                    mean: [117.001, 114.697, 97.404] // 预设期望
                    // std: [0.229, 0.224, 0.225]  // 预设方差
                }
            });
        }
        // log.end('预处理'); // eslint-disable-line
        // log.start('运行耗时'); // eslint-disable-line
        let inst = this.model.execute({
            input: feed
        });
        let result = await inst.read();
        // log.end('后处理-读取数据'); // eslint-disable-line
        const newData = [];
        let newIndex = -1;
        const [w, h, c, b] = this.modelConfig.outputShapes.from;
        // c channel
        for (let i = 0; i < c; i++) {
            // height channel
            for (let j = 0; j < h; j++) {
                // width channel
                for (let k = 0; k < w; k++) {
                    // position: (0, 0, 0, 0)
                    const index = j * (c * h) + k * c + i;
                    // const index = j * (i * k) + k * i + i;
                    newData[++newIndex] = result[index];
                }
            }
        }
        this.postProcess.run(newData, input, callback, feed[0].canvas);
        // log.end('后处理'); // eslint-disable-line
        this.flags.isRunning = false;
        // log.end('总耗时'); // eslint-disable-line
    }

    // 传入获取图片的function
    async runStream(getMedia, callback) {
        await this.run(getMedia, callback);
        if (!this.flags.runVideoPaused) {
            setTimeout(async () => {
                await this.runStream(getMedia, callback);
            }, 0);
        }
    }

    stopStream() {
        this.flags.runVideoPaused = true;
    }

    startStream(getMedia, callback) {
        this.flags.runVideoPaused = false;
        this.runStream(getMedia, callback);
    }
}
