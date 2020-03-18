/**
 * @file 视频流类
 * @author zhangmiao06
 */
import $ from 'webpack-zepto';
export default class Camera {
    constructor(option) {
        this.option = option;
        this.video = option.videoDom;
        // 标志是否可以切换摄像头
        this.haveDevice = false;
        // 设置视频流宽度
        if (option.width) {
            this.video.width = option.width;
        }
        else if (option.height) {
            this.video.height = option.height;
        }
        else {
            this.video.width = window.innerWidth;
        }
        this.deviceInfos = [];
        if(navigator.mediaDevices) {
            this.haveDevice = true;
        }
    }

    // 访问用户媒体设备的兼容方法
    run(deviceId, callback) {
        if (window.stream) {
            window.stream.getTracks().forEach(function (track) {
                track.stop();
            });
        }

        let constraints = {
            video: {}
        };
        const success = stream => {
            this.success(stream, callback);
        };
        const error = this.error.bind(this);
        if (this.deviceInfos.length) {
            constraints.video.deviceId=  {exact: deviceId || this.deviceInfos[0]};
        }

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            // 最新的标准API
            navigator.mediaDevices.getUserMedia(constraints).then(success).catch(error);
        }
        else if (navigator.webkitGetUserMedia) {
            // webkit核心浏览器
            navigator.webkitGetUserMedia(constraints, success, error);
        }
        else if (navigator.mozGetUserMedia) {
            // firfox浏览器
            navigator.mozGetUserMedia(constraints, success, error);
        }
        else if (navigator.getUserMedia) {
            // 旧版API
            navigator.getUserMedia(constraints, success, error);
        }
        else {
            console.log('您的浏览器不支持获取视频流~');
        }
    }

    success(stream, callback) {
        const domElement = this.video;
        // make stream available to console
        window.stream = stream;
        // 旧的浏览器可能没有srcObject
        const URL = window.URL || window.webkitURL || window.mozURL || window.msURL;
        if ('srcObject' in domElement) {
            try {
                domElement.srcObject = stream;
            } catch (error) {
                domElement.src = URL.createObjectURL(stream) || stream;
            }
        } else {
            // 防止再新的浏览器里使用它，应为它已经不再支持了
            domElement.src = URL.createObjectURL(stream) || stream;
        }
        domElement.addEventListener('loadeddata', () => {
            // 设置视频流高度
            if (this.option.height) {
                domElement.width = $(domElement).width();
            }
            else {
                domElement.height = $(domElement).height();
            }
            domElement.play();
            callback && callback();
        }, false);
    }

    error(error) {
        alert(`访问用户媒体设备失败${error.name}, ${error.message}`);
    }
    // 处理摄像头列表
    gotDevices(deviceInfos) {
        const ua = navigator.userAgent;
        const isIos = /iphone|ipod|ipad/ig.test(ua);

        let delt = -1;
        const range = deviceInfos.length;
        let start = range - 1;
        let end = - 1;
        // ios机型camare顺序相反
        if (isIos) {
            delt = 1;
            start = 0;
            end = range;
        }
        for (let i = start; i !== end; i += delt) {
            const deviceInfo = deviceInfos[i];
            if (deviceInfo.kind === 'videoinput') {
                this.deviceInfos.push(deviceInfos[i]);
            }
        }
    }

    get curVideo() {
        return this.video;
    }
    getDevices() {
        return new Promise((resolve, reject)=> {
            if (this.haveDevice) {
                if (this.deviceInfos.length) {
                    resolve(this.deviceInfos);
                }
                else {
                    navigator.mediaDevices.enumerateDevices()
                    .then(this.gotDevices.bind(this))
                    .then(()=> {
                        resolve(this.deviceInfos);
                    });
                }
            }
            else {
                resolve([]);
            }
        });
    }
}
