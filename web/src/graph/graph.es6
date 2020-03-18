/* eslint-disable */
import GraphExecutor from '../executor/executor';
import IO from '../feed/imageFeed';
import Runtime from '../runtime/runtime';
import OpData from '../utils/opData';
import Factory from '../factory/fshader/factory';
import Utils from '../utils/utils';

/**
 * @file Graph，绘制生成model网络
 * @author wangqun@baidu.com
 */
let start = 0;
// 生成factory实例
const factory = new Factory({});
// 获取op的输入配置
const opConfs = factory.getOpConfs();

export default class Graph {
    constructor(options) {
        this.version  = '0.0.1';
        this.handler = 'io.IOHandler';
        this.weightMap = '';
        this.options = options || {};
        // feed数据
        this.feed = null;
        this.index = 0;
        this.feedOp = null;
        this.feedItem = null;
        this.test = false;
        this.isExecuted = false;
        // 网络层数
        this.iLayer = 0;

        if (this.options && this.options.options && this.options.options.test === true) {
            this.test = true;
        }

        if (!this.inst) {
            // op runner
            this.inst = Runtime.init();
            factory.setWebglVersion(this.inst.getWebglVersion());

        }
    }

    buildOpData(op) {
        const executor = this.constructExecutor(op);
        const opData = new OpData(op.type, executor.inputs, executor.outputs, executor.attrs);
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
        // if (executor.next && start++ < 2) {
        if (executor.next) {
            const id = executor.next;
            const next = this.getTensor(id);
            this.execute_(next[0]);
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
        return this.data.vars.filter((item, i) => {
            if (name === item.name)
            return item;
        });
    }
    constructExecutor(executor) {
        let that = this;
        const inputName = executor.inputsName[0];
        const input = executor.inputs;
        const output = executor.outputs;
        Object.keys(output).forEach(function(key){
            output[key] = that.getTensorAttr(output[key][0]);
        });
        Object.keys(input).forEach(function(key){
            if (that.test && ((key === 'Input') || (key === 'X'))) {
                input[key] = that.getTensorAttr(input[key][0]);
                that.feedOp = executor;
            }
            else if ((key === 'Input') && (inputName === 'pixel')) {
                // const pixel = that.getTensorAttr(inputName);
                // const io = new IO();
                // input[key] = io.fromPixels(that.feed, pixel);
                input[key] = that.feed.input;

                that.feedOp = executor;
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
        return {
            inputs: input,
            outputs: output,
            attrs: executor.attrs,
            type: executor.type,
            next: executor.next
        };
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
     * dispose
     */
    dispose() {
        this.executor.dispose();
    }
}
/* eslint-enable */
