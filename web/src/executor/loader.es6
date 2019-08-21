/* eslint-disable */
import GraphExecutor from './executor';
import IO from '../feed/imageFeed';
import Runtime from '../../src/runtime/runtime';
import OpData from '../utils/opData';
import Factory from '../factory/fshader/factory';
import Utils from '../utils/utils';
/**
 * @file GraphModel，绘制生成model网络
 * @author wangqun@baidu.com
 */
// 生成factory实例
const factory = new Factory({});

// 获取op的输入配置
const opConfs = factory.getOpConfs();
export default class GraphModel  {
    constructor(modelGonfig, loadOptions) {
        this.version  = '0.0.1';
        this.handler = 'io.IOHandler';
        this.modelGonfig = modelGonfig;
        this.loadOptions = loadOptions;
        this.multipart = false;
        // feed数据
        this.feed = null;
        this.index = 0;
        this.feedOp = null;
        this.feedItem = null;
        this.isExecuted = false;
        // 网络层数
        this.iLayer = 0;
        // fetch xhr jsonp
        this.params = {type: 'fetch'};
        // 设置分片加载model
        if (this.loadOptions) {
            this.multipart = this.loadOptions.multipart;
            this.feed = {input: this.loadOptions.feed};
            if (loadOptions.dataType === 'binary') {
                this.binaryOption = loadOptions.binaryOption;
            }
        }

        if (!this.loadOptions) {
            this.loadOptions = {};
        } else {
            // op runner
            this.inst = Runtime.init();
            factory.setWebglVersion(this.inst.getWebglVersion());
            // this.fetchJson(this.modelGonfig.dir + 'x.json').then(data => {
            //     const [b, c, h, w] = [1, 3, 320, 320];
            //     const size = data.length;
            //     const total = 3 * 320 * 320;
            //     this.testData = new Float32Array(total);
            //     for (let i = 0; i < size; i++) {
            //         let j = i / (c * w) | 0;
            //         let k = i % (c * w);
            //         let b1 = j / h | 0;
            //         let h1 = j % h;
            //         let c1 = k % c;
            //         let w1 = k / c | 0;
            //         let l = b1 * (c * h * w) + c1 * (h * w) + h1 * (w) + w1;
            //         this.testData[i] = data[l];
            //     }
            // });
        }
    }
    fetchOneChunk(path) {
        return this.fetch(path).then(request => {
            return request.arrayBuffer();
        })
    }
    fetchJson(path) {
        return this.fetch(path).then(request => {
            return request.json();
        })
    }
    fetchAllData() {
        // todo 兼容一下json的模式
        let counts = this.binaryOption.fileCount;
        let chunkArray = [];
        for (let i = 1; i <= counts; i++) {
            chunkArray.push(
                this.fetchOneChunk(this.modelGonfig.dir + this.binaryOption.getFileName(i))
            );
        }
        console.time('加载时间');
        return Promise.all(chunkArray).then(chunks => {
            console.timeEnd('加载时间');
            let chunksLength = 0;
            let f32Array = [];
            let float32Chunk;
            chunks.forEach(i => {
                float32Chunk = new Float32Array(i);
                f32Array.push(float32Chunk);
                chunksLength += float32Chunk.length;
            });
            this.allData = new Float32Array(chunksLength);
            let offset = 0;
            f32Array.forEach(i => {
                i.forEach(num => {
                    this.allData[offset] = num;
                    offset += 1;
                })
            });
        });
    }
    traverse (arr) {
        const TMP_SCHEME_REGEX = /\.tmp/;
        const TMP_REGEX = /\-/;
        let marker = 0; // 读到哪个位置了
        let len; // 当前op长度
        arr.filter(item => {
            return item.name
                && item.name.match(TMP_SCHEME_REGEX) === null
                && item.name.match(TMP_REGEX) === null;
            })
            // .sort((a, b) => {
            //     if (a.name > b.name) {
            //         return 1;
            //     }
            //     if (a.name < b.name) {
            //         return -1;
            //     }
            //     return 0;
            // }) // 按字母顺序排列 在model.json里
            .forEach(item => {
                len = item.shape.reduce((a, b) => a * b); // 长度为shape的乘积
                item.data = this.allData.slice(marker, marker + len);
                marker += len;
            });
    }

    fetch(path, params) {
        params = params || this.params;
        let method = params.method || 'get';
        let mode = params.mode || 'cors';
        let myHeaders = new Headers();
        return fetch(path, {
            method: method,
            mode: mode,
            credentials: 'include',
            headers: myHeaders
        });
    }

    fetchModel(params) {
        params = params || this.params;
        const path = this.modelGonfig.dir + this.modelGonfig.main;
        let load = null;
        // jsonp请求方式
        if (params && params.type === 'jsonp') {
            let json;
            let s = document.createElement('script');
            s.src = path + '&jsonpCallback=fn';
            window.fn = function(data) {
                json = data;
                // console.log(json);
            };
            //当script被插入文档中时，src中的资源就会开始加载
            document.body.appendChild(s);
            load = new Promise((resolve, reject) => {
                s.onload = function(e) {
                    resolve(json);
                }
                s.onerror = function() {
                    reject(json);
                }
            });
            this.handler = load;
        }
        // 原生fetch
        else if (params.type === 'fetch') {
            load = new Promise((resolve, reject) => {
                this.fetch(path, params)
                .then(response => response.json())
                .then(responseData => resolve(responseData))
                .then(err => reject(err))
            });
            this.handler = load;
        }
        // ajax
        else if (params.type === 'xhr') {
            this.handler = load;
        }
        return load;
    }
    async load() {
        let that = this;
        const artifacts = this.handler = await this.fetchModel();
        if (this.multipart === true) {
            await this.fetchAllData()
                .then(() => this.traverse(artifacts.vars));
        }
        const opsMap = this.createOpsMap(artifacts.ops, artifacts.vars);
        this.weightMap = this.constructOpsMap(opsMap);
        // 生成op数据
        this.weightMap.forEach(op => {
            const type = op.type;
            if (type !== 'feed' && type !== 'fetch') {
                that.buildOpData(op);
            }
        });
        return true;
    }
    buildOpData(op) {
        const tensor = this.constructTensor(op);
        const opData = new OpData(op.type, tensor.inputs, tensor.outputs, tensor.attrs);
        const name = opData.name;
        const fsCode = factory.buildShader(name, opData.data);
        opData.fsCode = fsCode;
        opData.program = this.inst.createProgram(fsCode, opData.tensor['out']);
        opData.renderData = opConfs[name].map(elem => {
            let item = Object.assign({}, elem);
            const tensorData = opData.tensor[item.tensor];
            if (item.type === 'texture') {
                item.data = tensorData.data;
                if (this.feedOp.id === op.id && item.tensor === 'origin') {
                    item.shape = tensorData.shape;
                    this.feedItem = item;
                }
                item['width_texture'] = tensorData['width_texture'];
                item['height_texture'] = tensorData['height_texture'];
                item['channel'] = tensorData['channel'];
            } else if (item.type === 'uniform') {
                item.data = tensorData[item.variable];
            }
            return item;
        });
        // console.timeEnd('opData.renderData');
        opData.iLayer = this.iLayer++;
        op.opData = opData;
        // delete op.inputs;
        // delete op.outputs;
        // delete op.attrs;
    }
    execute_(executor) {
        if (executor.type === 'fetch') {
            return;
        }
        executor.execute(this.inst, this.isExecuted);
        if (executor.next) {
            const id = executor.next;
            const next = this.getTensor(id);
            this.execute_(next[0])
        }
    }
    /**
     * Executes inference for the model for given input tensors.
     * @param inputs
     * @param outputs
     * @returns {*}
     */
    execute(inputs) {
        this.feed = inputs;
        const executor = this.getNetsStart(this.weightMap);
        if (!this.inst) {
            this.inst = Runtime.init({
                'width_raw_canvas': 512,
                'height_raw_canvas': 512
            });
        }
        if (this.isExecuted) {
            this.updateFeed();
        }
        let start = +Date.now();
        this.execute_(executor[0]);
        this.isExecuted = true;
        return this.inst;
    }
    updateFeed() {
        this.feedItem.data = this.feed.input[0].data;
        // Utils.img2texture(this.feedItem);
    }
    /**
     * predict enter
     * @param inputs
     * @param config
     */
    predict(inputs, config) {
        return this.execute_(inputs, true, this.outputNodes);
    }
    getTensorAttr(name) {
        return this.handler.vars.filter((item, i) => {
            if (name === item.name)
            return item;
        });
    }
    constructTensor(executor) {
        const that = this;
        const inputName = executor.inputsName[0];
        const input = executor.inputs;
        const output = executor.outputs;
        Object.keys(output).forEach(function(key){
            output[key] = that.getTensorAttr(output[key][0]);
        });
        Object.keys(input).forEach(function(key){
            if ((key === 'Input') && (inputName === 'pixel')) {
                const pixel = that.getTensorAttr(inputName);
                const io = new IO();
                input[key] = io.fromPixels(data, pixel);
            }
            else if ((key === 'Input') && (inputName === 'image' || inputName === 'x')) {
                // that.feed.input[0].data = that.testData;
                input[key] = that.feed.input;
                that.feedOp = executor;
            }
            else {
                input[key] = that.getTensorAttr(input[key][0]);
            }
        });
        // console.log(input);
        const tensor = {
            inputs: input,
            outputs: output,
            attrs: executor.attrs,
            type: executor.type,
            next: executor.next
        };
        return tensor;
    }
    /**
     * Construct Ops Relationship
     * @param ops
     * @returns {*}
     */
    constructOpsMap(ops) {
        return ops.map((item, idx) => {
            const outputsName = item.outputsName[0];
            const next = this.getNextExecutor(ops, outputsName);
            if (next.length > 0) {
                item.next = next[0].id;
            }
            return item;
        });
    }
    /**
     * Get Ops Nets Start Node
     * @param ops
     * @returns {*}
     */
    getNetsStart(ops) {
        return ops.filter((item) => {
            if (item.type === 'feed') {
                return true;
            }
        });
    }
    /**
     * Get Ops Nets Last Node
     * @param ops
     * @returns {*}
     */
    getNetsEnd(ops) {
        return ops.filter((item) => {
            if (item.type === 'fetch') {
                return true;
            }
        });
    }
    /**
     * get tensor by id
     * @param id
     * @returns {*}
     */
    getTensor(id) {
        return this.weightMap.filter((item, i) => {
            if (id === item.id)
                return item;
        });
    }
    /**
     * Create Ops Executor Object Map
     * @param ops
     * @returns {*}
     */
    createOpsMap(ops) {
        return ops.map((item, idx) => {
            item.idx = idx;
            const graphExecutor = new GraphExecutor(item);
            return graphExecutor;
        });
    }
    /**
     * Get The Next Executor need Exec
     * @param ops
     * @param id
     * @returns {*}
     */
    getNextExecutor(ops, id) {
        return ops.filter((item, key) => {
            if (id === item.inputsName[0]) {
                return true;
            }
        });
    }
    /**
     * Load a graph model given a URL to the model definition.
     * @param modelGonfig
     * @param options
     * @returns {Promise<void>}
     */
    async loadGraphModel(modelGonfig, options) {
        if (modelGonfig === null) {
            // todo saniac 报错提示修改
            throw new Error(
                'modelGonfig in loadGraphModel() cannot be null. Please provide a url ' +
                'or an IOHandler that loads the model');
        }
        if (options === null) {
            options = {};
        }
        const model = new GraphModel(modelGonfig, options);
        await model.load();
        return model;
    }
    /**
     * dispose
     */
    dispose() {
        this.executor.dispose();
    }
}
/* eslint-enable */
