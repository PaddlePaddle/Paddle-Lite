/* eslint-disable */
/**
 * @file GraphExecutor，封装可执行单元
 * @author wangqun@baidu.com
 */
// const fileDownload = require('js-file-download');
let start;
export default class GraphExecutor {

    constructor(model) {
        this.inputs = model.inputs;
        this.outputs  = model.outputs;
        this.attrs = model.attrs;
        this.type = model.type;
        this.finish = false;
        this.next = null;
        this.opData = null;
        this.id = +new Date() + model.type + Math.floor(Math.random() * 10 + 1) + model.idx;
    }

    get inputsName() {

        if (this.type === 'feed') {
            return this.inputs.X;
        }
        else if (this.type === 'batchnorm' || this.type === 'batch_norm') {
            return this.inputs.X;
        }
        else if (this.type === 'conv2d') {
            return this.inputs.Input;
        }
        else if (this.type === 'depthwise_conv2d') {
            return this.inputs.Input;
        }
        else if (this.type === 'elementwise_add') {
            return this.inputs.X;
        }
        else if (this.type === 'relu' || this.type === 'leaky_relu') {
            return this.inputs.X;
        }
        else if (this.type === 'pool2d') {
            return this.inputs.X;
        }
        else if (this.type === 'mul') {
            return this.inputs.X;
        }
        else if (this.type === 'softmax') {
            return this.inputs.X;
        }
        else if (this.type === 'scale') {
            return this.inputs.X;
        }
        else if (this.type === 'fetch') {
            return this.inputs.X;
        }
        return null;
    }

    get outputsName() {
        if (this.type === 'conv2d') {
            return this.outputs.Output;
        }
        else if (this.type === 'depthwise_conv2d') {
            return this.outputs.Output;
        }
        else if (this.type === 'batchnorm' || this.type === 'batch_norm') {
            this.outputs.out = this.outputs.Y;
            return this.outputs.Y;
        }
        else {
            return this.outputs.Out;
        }

    }

    /**
     * 将输入数据和具体op进行关联，触发执行具体每一个op
     * @param inputs
     * @param runtime
     */
    execute(runtime) {
        // console.log(inputs, outputs);
        if (this.type !== 'feed') {
            let time = +Date.now();
            runtime.run(this.type, this.opData);
            // if (runtime.gpu.frameBufferIsComplete().isComplete) {
            //     var result = runtime.read();
            //     let res = Array.prototype.slice.call(result);
            //     fileDownload(res, "result.csv");
            // }
            let length = statistic.length;
            statistic[length - 1].type = this.type;
            statistic[length - 1].runTime = +Date.now() - time;
            // if (this.type === 'scale') {
            //     console.log('时间是：' + (+Date.now() - start));
            // }
        } else {
            start = +Date.now();
        }
    }
}

/* eslint-enable */
