/* eslint-disable */
/**
 * @file loader，model加载器
 * @author wangqun@baidu.com
 */

export default class Loader  {
    constructor(modelGonfig, options) {
        this.version  = '0.0.1';
        this.data = {};
        this.modelGonfig = modelGonfig;
        this.options = options;
        this.multipart = false;
        this.test = false;
        // fetch xhr jsonp
        this.params = {type: 'fetch'};
        // 设置分片加载model
        if (this.options) {
            this.multipart = this.options.multipart;
            if (options.dataType === 'binary') {
                this.binaryOption = options.options;
                this.dataType = options.dataType;
            }
            if (options.test) {
                this.test = true;
            }
        }

        if (!this.loadOptions) {
            this.loadOptions = {};
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

    fetchChunks() {
        let counts = this.binaryOption.fileCount;
        let chunkArray = [];
        for (let i = 1; i <= counts; i++) {
            chunkArray.push(
                this.fetchOneChunk(this.modelGonfig.dir + this.binaryOption.getFileName(i))
            );
        }
        // console.time('加载时间');
        return Promise.all(chunkArray).then(chunks => {
            // console.timeEnd('加载时间');
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

    fetchData(name) {
        const path = this.modelGonfig.dir + name + '.json';
        let load = new Promise((resolve, reject) => {
            fetch(path, {
                method: 'get', mode: 'cors', credentials: "include",
                headers: { 'Content-Type': 'application/json;charset=utf-8'}})
                .then(response => response.json())
                .then(responseData => resolve(responseData))
                .then(err => reject(err))
        })
        return load;
    }

    async fetchAllDate (arr) {
        const TMP_SCHEME_REGEX = /\.tmp/;
        const TMP_REGEX = /\-/;
        let requesterArr = arr.map(item => {
            if (item.name
                && item.name.match(TMP_SCHEME_REGEX) === null
                && item.name.match(TMP_REGEX) === null) {
                return this.fetchData(item.name).then(data => item.data = data);
            }
            return Promise.resolve();
        });
        return Promise.all(requesterArr);
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
            .forEach(item => {
                len = item.shape.reduce((a, b) => a * b); // 长度为shape的乘积
                item.data = this.allData.slice(marker, marker + len);
                marker += len;
            });
    }

    fetch(path, params) {
        params = params || this.params;
        let method = params.method || 'get';
        let mode = params.mode || 'no-cors';
        let myHeaders = new Headers();
        return fetch(path, {
            method: method,
            // mode: mode,
            // credentials: 'include',
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
            this.data = load;
        }
        // 原生fetch
        else if (params.type === 'fetch') {
            load = new Promise((resolve, reject) => {
                this.fetch(path, params)
                .then(response => response.json())
                .then(responseData => resolve(responseData))
                .then(err => reject(err))
            });
            this.data = load;
        }
        // ajax
        else if (params.type === 'xhr') {
            this.data = load;
        }
        return load;
    }

    async load() {
        let that = this;
        const artifacts = this.data = await this.fetchModel();
        if (this.multipart === true) {
            if (this.dataType === 'binary') {
                await this.fetchChunks()
                    .then(() => this.traverse(artifacts.vars));
            }
            else {
                await that.fetchAllDate(artifacts.vars);
            }
        }
        return artifacts;
    }

}
/* eslint-enable */
