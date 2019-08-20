import 'babel-polyfill';
import Runner from '../src/executor/runner';
import Camera from './camera';

let startBtn = document.getElementById('start');
let stopBtn = document.getElementById('stop')

const runner = new Runner({
    // 用哪个模型
    modelName: 'separate' // '608' | '320' | '320fused' | 'separate'
});
startBtn.disabled = true;
runner.preheat().then(() => startBtn.disabled = false);

let camera = new Camera({
    videoDom: document.getElementById('video'), // 用来显示摄像头图像的dom
    videoOption: {
        video: {
            width: 480,
            height: 320,
            frameRate: {
                ideal: 8, max: 15
            }
        }
    }
});
camera.run(); // 启动摄像头

startBtn.addEventListener('click', function () {
    startBtn.disabled = true;
    runner.startStream(() => camera.curVideo);
});
stopBtn.addEventListener('click', function () {
    startBtn.disabled = false;
    runner.stopStream();
});