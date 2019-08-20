/**
 * @file 直接数据输入
 * @author hantianjiao@baidu.com
 */

export default class dataFeed {
    toFloat32Array(data) {
        for (let i = 0; i < data.length; i++) {
            this.f32Arr[i] = data[i];
        }
    }

    getLengthFromShape(shape) {
        return shape.reduce((a, b) => a * b);
    }

    loadData() {
        return fetch(this.dataPath).then(res => res.json());
    }

    getOutput() {
        return this.loadData().then(data => {
            this.toFloat32Array(data);
            return [{
                data: this.f32Arr,
                shape: this.shape,
                name: 'x'
            }];
        });
    }

    async process(input) {
        this.len = this.getLengthFromShape(input.shape);
        if (!this.f32Arr || this.len > this.f32Arr.length) {
            this.f32Arr = new Float32Array(this.len);
        }
        this.shape = input.shape;
        this.dataPath = input.input;
        let output = await this.getOutput();
        return output;
    }
}