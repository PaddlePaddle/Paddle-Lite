export default class Camera {
    constructor(option) {
        this.video = option.videoDom;
        this.videoOption = option.videoOption;
    }

    // 访问用户媒体设备的兼容方法
    getUserMedia(constraints, success, error) {
        if (navigator.mediaDevices.getUserMedia) {
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
    }

    success(stream) {
        // 兼容webkit核心浏览器
        let CompatibleURL = window.URL || window.webkitURL;
        // 将视频流设置为video元素的源

        // video.src = CompatibleURL.createObjectURL(stream);
        this.video.srcObject = stream;
        this.video.play();
    }

    error(error) {
        console.log(`访问用户媒体设备失败${error.name}, ${error.message}`);
    }

    run() {
        if (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.mediaDevices.getUserMedia) {
            // 调用用户媒体设备, 访问摄像头
            this.getUserMedia(this.videoOption, this.success.bind(this), this.error);
        }
        else {
            alert('不支持访问用户媒体');
        }
    }

    get curVideo() {
        return this.video;
    }
}
