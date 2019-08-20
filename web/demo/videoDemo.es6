import 'babel-polyfill';
import Runner from '../src/executor/runner';
<<<<<<< HEAD
import Camera from '../src/executor/camera';
// 调试工具
// import vConsole from 'vconsole';
// const theConsole = new vConsole();
=======
import Camera from './camera';

>>>>>>> paddle web
let startBtn = document.getElementById('start');
let stopBtn = document.getElementById('stop')

const runner = new Runner({
    // 用哪个模型
    modelName: 'separate' // '608' | '320' | '320fused' | 'separate'
});
startBtn.disabled = true;
<<<<<<< HEAD
runner.preheat()
.then(() =>{
    startBtn.disabled = false
});

const domElement = document.getElementById('video');
const myCanvas = document.getElementById('myDiv');
const videoSelect = document.getElementById('videoSelect');
let camera = new Camera({
    // 用来显示摄像头图像的dom
    videoDom: domElement
});
camera.getDevices().then(devices => {
    if (devices.length) {
        camera.run(devices[0].deviceId);
        devices.forEach((element, index) => {
            let option = document.createElement('option');
            option.value = element.deviceId;
            option.text = (index + 1);
            videoSelect.appendChild(option);
        });
        videoSelect.onchange = () => {
            camera.run(videoSelect.value);
        };
    }
    else {
        camera.run();
    }
});
const handleDiv = function (data) {
    myCanvas.style.width = (data ? data[0] : 0) + 'px';
    myCanvas.style.height = (data ? data[0] : 0) + 'px';
    myCanvas.style.left = (data ? data[2] : 0) + 'px';
    myCanvas.style.top = (data ? data[3] : 0) + 'px';
}
startBtn.addEventListener('click', function () {
    startBtn.disabled = true;
    runner.startStream(() => camera.curVideo, handleDiv);
=======
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
>>>>>>> paddle web
});
stopBtn.addEventListener('click', function () {
    startBtn.disabled = false;
    runner.stopStream();
});