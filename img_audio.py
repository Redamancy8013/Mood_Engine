from flask import Flask, Response
import cv2
import sounddevice as sd
import numpy as np
import wave
import time
import requests
app = Flask(__name__)

# 视频部分初始化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not start camera.")

# 音频参数
SAMPLE_RATE = 16000  # 采样率
CHANNELS = 1         # 单声道
BLOCK_SIZE = 1024    # 每次采样的数据块

# 视频流生成器
def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 音频流生成器
def audio_generator():
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', blocksize=BLOCK_SIZE) as stream:
        while True:
            block, _ = stream.read(BLOCK_SIZE)
            yield block.tobytes()

# 视频流接口
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 音频流接口
@app.route('/audio_stream')
def audio_stream():
    return Response(audio_generator(),
                    mimetype='application/octet-stream')

# 首页界面
@app.route('/')
def index():
    return '''
    <html>
    <body>
        <h1>Camera Streaming</h1>
        <img src="/video_feed" width="640" height="480" />
        <p>音频流地址：<a href="/audio_stream" target="_blank">/audio_stream</a></p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
