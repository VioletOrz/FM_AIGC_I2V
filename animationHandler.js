document.addEventListener('DOMContentLoaded', function () {
    function initializeVideoOnClick() {
        const mouthCanvas = document.getElementById('mouthCanvas'); // 获取画布元素

        // const streamStartTimeText = document.getElementById('time-to-stream-start'); // 获取流式开始时间文本元素
        // const streamEndTimeText = document.getElementById('time-to-stream-end'); // 获取流式结束时间文本元素
        // const playbackDelayTimeText = document.getElementById('time-to-playback'); // 获取播放延迟时间文本元素
        // const animationLoadTimeText = document.getElementById('animation-load-time'); // 获取动画加载时间文本元素

        // 定义全局变量
        const images = {}; // 存储图片对象的字典
        let audioContext = null; // 音频上下文
        let sourceNode = null; // 音频源节点
        let isPlayingAudio = false; // 是否正在播放音频
        let buttonPressTime = 0; // 按钮点击时间
        let streamStartTime = 0; // 流式开始时间
        let streamEndTime = 0; // 流式结束时间
        const pageLoadTime = performance.now(); // 记录页面刷新时间

        const bufferQueue = []; // 缓冲队列
        const mouthShapeQueue = []; // 嘴形队列
        let remainingData = null; // 剩余数据
        const playingQueue = []; // 播放队列
        const BUFFER_THRESHOLD = 32000; // 缓冲阈值

        let lastFrameIndex = 1; // 最后一帧的索引
        let lastFrameDirection = 1; // 最后一帧的方向
        const totalFrames = 60; // 总帧数

        var navStatus = false;

        preloadImages();

        mouthCanvas.addEventListener('click', handleCanvasClick); // 监听单击事件
        mouthCanvas.addEventListener('contextmenu', handleCanvasRightClick); // 监听右键点击事件

        // 处理单击事件的函数
        function handleCanvasClick(event) {
            // 获取鼠标点击位置
            const rect = mouthCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left; // 获取相对于画布的X坐标
            const y = event.clientY - rect.top; // 获取相对于画布的Y坐标

            if (navStatus) {
                document.getElementById('nav').style = 'opacity:0'; navStatus = false;
                document.getElementById('out').style = 'opacity:0';
                window.electronAPI.resizeWindow(400, 800);
                return
            }
            // 在这里添加处理单击事件的逻辑
            generateVoice("寂寞？或许吧。但风中有着无数的声音，它们陪伴着我。而且，守护璃月的使命让我没有时间去感受孤独。")
            // 例如：可以根据点击位置进行一些动画或其他操作
        }

        function handleCanvasRightClick(event) {
            event.preventDefault(); // 阻止默认右键菜单弹出
            event.stopPropagation(); // 阻止事件冒泡，防止其他监听器干扰
            console.log('右键点击已触发');

            const rect = mouthCanvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            document.getElementById('nav').style = 'opacity:1';
            document.getElementById('out').style = 'opacity:1';
            navStatus = true;
            window.electronAPI.resizeWindow(515, 800);
        }



        // 生成语音函数
        function generateVoice(text) {
            sendPostRequest(text)
        }

        // 发送POST请求函数
        async function sendPostRequest(text) {
            let role = sessionStorage.getItem("role")
            // 发送POST请求
            const response = await fetch('https://human.nijigen.com.cn/otaku_tianzi/synthesize/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    text: text,
                    role_base: 'funingna',
                    ref_ids: 'b716979c-df8d-11ee-9fbb-171ed281fa1b',
                    return_phone: true,
                    text_language: 'zh',
                }),
            });

            // 如果响应体存在，处理响应数据
            if (response.body) {
                const reader = response.body.getReader(); // 获取响应读取器
                let receivedLength = 0; // 接收到的数据长度

                // 循环读取响应数据
                while (true) {
                    const { done, value } = await reader.read(); // 读取数据
                    if (done) break; // 如果读取完成，跳出循环


                    receivedLength += value.byteLength; // 更新接收到的数据长度

                    processChunk(value.buffer); // 处理数据块

                    // 如果缓冲队列长度超过阈值，处理缓冲音频
                    if (getBufferedLength() >= BUFFER_THRESHOLD) {
                        enqueueBufferedAudio();
                    }
                }

                // 如果缓冲队列还有数据，处理缓冲音频
                if (getBufferedLength() > 0) {
                    enqueueBufferedAudio();
                }
            }
        }

        // 处理数据块函数
        function processChunk(buffer) {
            const data = new Uint8Array(buffer); // 将数据块转换为Uint8Array
            let combinedData;

            // 如果剩余数据存在，合并剩余数据和当前数据块
            if (remainingData) {
                combinedData = new Uint8Array(remainingData.byteLength + data.byteLength);
                combinedData.set(remainingData, 0);
                combinedData.set(data, remainingData.byteLength);
                remainingData = null;
            } else {
                combinedData = data;
            }

            let lastIndex = 0;
            const searchString = ';;['; // 搜索字符串
            const searchEndString = ']'; // 搜索结束字符串

            // 循环处理数据块
            while (lastIndex < combinedData.byteLength) {
                const startIndex = findSequence(combinedData, searchString, lastIndex); // 查找搜索字符串的起始位置
                if (startIndex === -1) break; // 如果未找到，跳出循环
                const endIndex = findSequence(combinedData, searchEndString, startIndex); // 查找搜索结束字符串的结束位置
                if (endIndex === -1) break; // 如果未找到，跳出循环

                const pcmBytes = combinedData.slice(lastIndex, startIndex); // 提取PCM数据
                bufferQueue.push(pcmBytes); // 将PCM数据添加到缓冲队列

                const mouthShapeBytes = combinedData.slice(startIndex, endIndex + 1); // 提取嘴形数据
                const mouthShapeString = new TextDecoder().decode(mouthShapeBytes).replace(';;', ''); // 解码嘴形数据
                mouthShapeQueue.push(JSON.parse(mouthShapeString)); // 将嘴形数据添加到嘴形队列

                lastIndex = endIndex + 1; // 更新最后索引
            }

            // 如果还有剩余数据，保存剩余数据
            if (lastIndex < combinedData.byteLength) {
                remainingData = combinedData.slice(lastIndex);
            }
        }

        // 查找序列函数
        function findSequence(data, sequence, startIndex) {
            for (let i = startIndex; i <= data.byteLength - sequence.length; i++) {
                let match = true;
                for (let j = 0; j < sequence.length; j++) {
                    if (String.fromCharCode(data[i + j]) !== sequence[j]) {
                        match = false;
                        break;
                    }
                }
                if (match) return i;
            }
            return -1;
        }

        // 获取缓冲队列长度函数
        function getBufferedLength() {
            return bufferQueue.reduce((acc, buffer) => acc + buffer.byteLength, 0);
        }

        // 处理缓冲音频函数
        function enqueueBufferedAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.AudioContext)(); // 创建音频上下文
            }

            const combinedBuffer = mergeBuffers(bufferQueue); // 合并缓冲队列中的数据
            bufferQueue.length = 0; // 清空缓冲队列

            const wavBuffer = addWavHeader(combinedBuffer); // 添加WAV头
            playingQueue.push(wavBuffer); // 将WAV数据添加到播放队列

            // 如果没有正在播放音频，播放下一个音频块
            if (!isPlayingAudio) {
                playNextAudioChunk();
            }
        }

        // 合并缓冲队列中的数据函数
        function mergeBuffers(buffers) {
            const totalLength = buffers.reduce((sum, buffer) => sum + buffer.byteLength, 0);
            const mergedBuffer = new Uint8Array(totalLength);
            let offset = 0;
            for (const buffer of buffers) {
                mergedBuffer.set(buffer, offset);
                offset += buffer.byteLength;
            }
            return mergedBuffer;
        }

        // 添加WAV头函数
        function addWavHeader(audioBuffer) {
            const numOfChannels = 1; // 单声道
            const sampleRate = 32000; // 采样率
            const bitsPerSample = 16; // 每个样本的位数
            const byteRate = sampleRate * numOfChannels * bitsPerSample / 8;
            const blockAlign = numOfChannels * bitsPerSample / 8;
            const wavHeader = new ArrayBuffer(44);
            const view = new DataView(wavHeader);

            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + audioBuffer.byteLength, true);
            writeString(view, 8, 'WAVE');
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, numOfChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, byteRate, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitsPerSample, true);
            writeString(view, 36, 'data');
            view.setUint32(40, audioBuffer.byteLength, true);

            const wavBuffer = new Uint8Array(44 + audioBuffer.byteLength);
            wavBuffer.set(new Uint8Array(wavHeader), 0);
            wavBuffer.set(audioBuffer, 44);

            return wavBuffer.buffer;
        }

        // 预加载图片函数
        function preloadImages() {
            const imagePackages = ['b_0.0', 'a_1.0', 'a_0.5', 'i_1.0', 'o_1.0'];
            const totalImages = totalFrames;
            let imagesLoaded = 0;
            const totalImageCount = imagePackages.length * totalImages;


            imagePackages.forEach((pkg) => {
                images[pkg] = [];
                for (let i = 1; i <= totalImages; i++) {
                    const img = new Image();
                    // console.log(`https://bucket.nijigen.com.cn/cdn_dir/human_platform_bot_roles/abeiduo/${pkg}/img_${i}.webp`);
                    // img.src = `https://bucket.nijigen.com.cn/cdn_dir/custom_bot_roles/pack/6906d150-9bed-4968-8bab-bf66ed82bbc8/${pkg}/img_${i}.png`
                    // img.src = `https://bucket.nijigen.com.cn/cdn_dir/human_platform_bot_roles/abeiduo/${pkg}/img_${i}.webp`
                    img.src = `C:/Users/Violet/Desktop/facial/阿贝多/Albedo_trans/Difference_S/${pkg}/img_${addPreZero(i)}.webp`

                    img.onload = () => {
                        imagesLoaded++;
                        if (imagesLoaded === totalImageCount) {
                            const animationLoadTime = performance.now() - pageLoadTime;
                            // animationLoadTimeText.innerText = `${animationLoadTime.toFixed(2)} ms`;
                            startDefaultAnimation();
                        }
                    };
                    images[pkg][i] = img;
                }
            });
        }

        // 播放下一个音频块函数
        function playNextAudioChunk() {
            if (playingQueue.length === 0) return;

            const nextBuffer = playingQueue.shift();

            audioContext.decodeAudioData(nextBuffer, function (buffer) {
                if (sourceNode) {
                    sourceNode.stop();
                }
                sourceNode = audioContext.createBufferSource();
                sourceNode.buffer = buffer;
                sourceNode.connect(audioContext.destination);

                const startTime = audioContext.currentTime;
                isPlayingAudio = true;

                const playbackDelayTime = performance.now() - buttonPressTime;
                // playbackDelayTimeText.innerText = `${playbackDelayTime.toFixed(2)} ms`;

                sourceNode.start();

                const currentMouthShapes = mouthShapeQueue.shift() || [];
                const mappedMouthShapeSequences = currentMouthShapes.map(mapMouthShape);

                drawMouthShapeAnimation(mappedMouthShapeSequences, startTime, buffer.duration);

                if (mouthShapeQueue.length > 0) {
                    const nextMouthShapes = mouthShapeQueue[0]; // 获取下一个块的嘴形数据
                    preloadNextMouthShapeFrames(nextMouthShapes);
                }

                sourceNode.onended = function () {
                    isPlayingAudio = false;
                    if (playingQueue.length > 0) {
                        playNextAudioChunk();
                    } else {
                        startDefaultAnimation();
                    }
                };
            });
        }

        // 映射嘴形函数
        function mapMouthShape(mouthShape) {
            const mapping = {
                'd_1.0': 'o_1.0',
                'd_0.75': 'o_1.0',
                'd_0.5': 'a_0.5',
                'd_0.25': 'a_0.5',
                'd_0.0': 'b_0.0',
                'u_1.0': 'o_1.0',
                'u_0.75': 'o_1.0',
                'u_0.5': 'a_0.5',
                'u_0.25': 'a_0.5',
                'u_0.0': 'b_0.0',
                'e_1.0': 'a_1.0',
                'e_0.75': 'a_1.0',
                'e_0.5': 'a_0.5',
                'e_0.25': 'a_0.5',
                'e_0.0': 'b_0.0',
                'a_0.75': 'a_1.0',
                'a_0.25': 'a_0.5',
                'a_0.0': 'b_0.0',
                'o_0.75': 'o_1.0',
                'o_0.5': 'a_0.5',
                'o_0.25': 'a_0.5',
                'o_0.0': 'b_0.0',
                'i_0.75': 'i_1.0',
                'i_0.25': 'a_0.5',
                'i_0.25': 'a_0.5',
                'i_0.0': 'b_0.0',
            };

            return mapping[mouthShape] || mouthShape;
        }

        // 绘制嘴形动画函数
        function drawMouthShapeAnimation(mouthShapeSequences, startTime, duration) {
            const context = mouthCanvas.getContext('2d');

            const canvasWidth = mouthCanvas.width;
            const canvasHeight = mouthCanvas.height;

            let frameIndex = lastFrameIndex;
            let direction = lastFrameDirection;

            const totalSampledFrames = mouthShapeSequences.length;
            const targetFrameRate = 30;
            const frameDuration = 1000 / targetFrameRate;
            let lastFrameTime = 0;

            function updateMouthShape(timestamp) {
                if (!lastFrameTime) lastFrameTime = timestamp;

                const currentTime = audioContext.currentTime;
                const elapsedTime = currentTime - startTime;

                if (elapsedTime >= duration) {
                    lastFrameIndex = frameIndex;
                    lastFrameDirection = direction;
                    return;
                }

                if (timestamp - lastFrameTime >= frameDuration) {
                    lastFrameTime = timestamp;

                    const progress = elapsedTime / duration;
                    const sampledFrameIndex = Math.floor(progress * totalSampledFrames);

                    const mouthShape = mouthShapeSequences[sampledFrameIndex % totalSampledFrames].trim();
                    const imgPackage = images[mouthShape] || images['b_0.0'];
                    const animationFrame = frameIndex;

                    const baseImage = images['b_0.0'][animationFrame];
                    const maskImage = imgPackage[animationFrame];

                    if (baseImage && maskImage) {
                        const scale = canvasHeight / baseImage.height;
                        const scaledWidth = baseImage.width * scale;
                        const offsetX = (canvasWidth - scaledWidth) / 2;

                        // 清除画布
                        context.clearRect(0, 0, mouthCanvas.width, mouthCanvas.height);

                        // 设置合成操作
                        context.globalCompositeOperation = 'source-over';

                        // 启用图像平滑
                        context.imageSmoothingEnabled = true;

                        // 绘制基础图像
                        context.drawImage(baseImage, offsetX, 0, scaledWidth, canvasHeight);

                        // 绘制遮罩图像
                        context.drawImage(maskImage, offsetX, 0, scaledWidth, canvasHeight);

                        // 绘制渐变蒙版
                        // context.fillStyle = 'rgba(0,0,0,0.5)';
                        // context.fillRect(0, 0, mouthCanvas.width, 48);
                        // context.illRect(0, mouthCanvas.height - 133, mouthCanvas.width, 48);
                        // drawMask(context);
                    }

                    frameIndex += direction;
                    if (frameIndex >= totalFrames || frameIndex <= 1) {
                        direction *= -1;
                    }
                }

                requestAnimationFrame(updateMouthShape);
            }

            requestAnimationFrame(updateMouthShape);
        }

        function drawMask(ctx) {
            // 设置蒙版的样式，例如透明度、颜色等
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)'; // 半透明黑色
            ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height); // 覆盖整个画布
        }

        // 开始默认动画函数
        function startDefaultAnimation() {
            const context = mouthCanvas.getContext('2d');
            const canvasWidth = mouthCanvas.width;
            const canvasHeight = mouthCanvas.height;

            let frameIndex = lastFrameIndex;
            let direction = lastFrameDirection;

            const frameDuration = 1000 / 30;
            let lastFrameTime = 0;

            function updateDefaultAnimation(timestamp) {
                if (!lastFrameTime) lastFrameTime = timestamp;

                if (timestamp - lastFrameTime >= frameDuration) {
                    lastFrameTime = timestamp;

                    const baseImage = images['b_0.0'][frameIndex];

                    const scale = canvasHeight / baseImage.height;
                    const scaledWidth = baseImage.width * scale;
                    const offsetX = (canvasWidth - scaledWidth) / 2;

                    // 清除画布
                    context.clearRect(0, 0, mouthCanvas.width, mouthCanvas.height);

                    // 设置合成操作
                    context.globalCompositeOperation = 'source-over';

                    // 启用图像平滑
                    context.imageSmoothingEnabled = true;

                    // 绘制基础图像
                    context.drawImage(baseImage, offsetX, 0, scaledWidth, canvasHeight);

                    frameIndex += direction;
                    if (frameIndex >= totalFrames || frameIndex <= 1) {
                        direction *= -1;
                    }
                }

                if (!isPlayingAudio) {
                    requestAnimationFrame(updateDefaultAnimation);
                }
            }

            requestAnimationFrame(updateDefaultAnimation);
        }

        // 预加载下一组嘴形动画帧函数
        async function preloadNextMouthShapeFrames(mouthShapeSequences) {
            const sampleRate = Math.floor(mouthShapeSequences.length / (mouthShapeSequences.length * 0.3));
            const sampledMouthShapeSequences = mouthShapeSequences.filter((_, index) => index % sampleRate === 0);

            const totalSampledFrames = sampledMouthShapeSequences.length;

            for (let i = 0; i < totalSampledFrames; i++) {
                let frameIndex = (lastFrameIndex + i * lastFrameDirection) % totalFrames;
                frameIndex = Math.max(frameIndex, 1); // 确保frameIndex不小于1
                const mouthShape = sampledMouthShapeSequences[i].trim();
                if (!images[mouthShape]) {
                    // 如果图像包还未加载完成，跳过此帧
                    continue;
                }

                // 异步加载图像
                await loadImage(images[mouthShape][frameIndex]);
            }
        }

        // 加载图像函数
        function loadImage(image) {
            return new Promise((resolve, reject) => {
                if (image.complete) {
                    resolve(); // 如果图像已经加载完毕，立即resolve
                } else {
                    image.onload = () => resolve(); // 图像加载完成后resolve
                    image.onerror = (err) => reject(err); // 加载失败时reject
                }
            });
        }

        // 淡入淡出效果函数
        function fadeInOut(context, baseImage, maskImage, canvasWidth, canvasHeight, duration) {
            const fadeDuration = 500; // 淡入淡出持续时间
            const fadeSteps = 20; // 淡入淡出步数
            const fadeStepDuration = fadeDuration / fadeSteps;

            let opacity = 0;
            let fadeDirection = 1; // 1 表示淡入，-1 表示淡出

            function fade(timestamp) {
                if (opacity >= 1) {
                    fadeDirection = -1;
                } else if (opacity <= 0) {
                    fadeDirection = 1;
                }

                opacity += fadeDirection * (1 / fadeSteps);

                const scale = canvasHeight / baseImage.height;
                const scaledWidth = baseImage.width * scale;
                const offsetX = (canvasWidth - scaledWidth) / 2;

                context.clearRect(0, 0, canvasWidth, canvasHeight);
                context.globalAlpha = opacity;
                context.drawImage(baseImage, offsetX, 0, scaledWidth, canvasHeight);
                context.drawImage(maskImage, offsetX, 0, scaledWidth, canvasHeight);

                if (opacity > 0 && opacity < 1) {
                    requestAnimationFrame(fade);
                } else {
                    context.globalAlpha = 1;
                }
            }

            requestAnimationFrame(fade);
        }
    }
    initializeVideoOnClick()
});
// 选择所有的元素
const navItems = document.querySelectorAll('.roletitle, .community, .rolesetting, .chat, .out');

document.addEventListener('mousemove', (event) => {
    const mouseX = event.clientX;

    // 摇晃的幅度大小
    const maxSwing = 15; // 摇晃的最大像素偏移量

    // 遍历每个 navItem，并给每个元素添加摇晃动画
    navItems.forEach((item) => {
        const swing = (mouseX / window.innerWidth - 0.5) * maxSwing;

        // 设置容器的左右摇晃效果
        item.style.transform = `translateX(${swing}px)`;

        // 如果还需要单独控制图片的摇晃效果，下面的代码可以设置图片的偏移
        const img = item.querySelector('img');
        if (img) {
            img.style.transform = `translateX(${swing / 2}px)`; // 图片摇晃幅度稍微小一点
        }
    });
});

function addPreZero(num){
    if(num < 10){
        return "00"+num;
    }
    else if (num < 100){
        return "0"+num;
    }
}