import os
import re
import cv2
import json
import time
import wave
import torch
import langid
import pygame
import asyncio
import pyaudio
import requests
import argparse
import edge_tts
import threading
import webrtcvad
import numpy as np

from time import sleep
from queue import Queue
from funasr import AutoModel
from langdetect import detect
from pypinyin import pinyin, Style
from modelscope.pipelines import pipeline
from visual.detect import build_emo_detector, Detect
from flask import Flask, Response, render_template, jsonify, send_from_directory
from pathlib import Path
from pyngrok import ngrok

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 创建Flask应用
app = Flask(__name__)

# 全局配置
CHUNK = 1024
AUDIO_RATE = 16000
NO_SPEECH_THRESHOLD = 2
AUDIO_CHANNELS = 1  # 单声道
vad = webrtcvad.Vad(2)  # 灵敏度 0~3

OUTPUT_DIR = "./output"   # 输出目录
folder_path = "./Test_QWen2_VL/"
NO_SPEECH_THRESHOLD = 1   # 无效语音阈值，单位：秒
audio_file_count = 0  
audio_file_count_tmp = 0

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(folder_path, exist_ok=True)

# 全局变量
emotion_queue = Queue()
last_active_time = time.time()
recording_active = True
segments_to_save = []
saved_intervals = []
audio_buffer = []
last_vad_end_time = 0  # 上次保存的 VAD 有效段结束时间
_play_thread = None
_stop_event = threading.Event()
_is_playing = False

# UI相关全局变量
current_face_emotion_tag = "peace"
current_face_VAD_tag = "0,0,0"
current_stt_message = ""
ai_thinking = ""
ai_response = ""
emo_url = None

# 用集合存储已知的音频文件
known_files = set()

# --- 唤醒词、声纹变量配置 ---
# set_KWS = "ni hao xiao qian"
# set_KWS = "shuo hua xiao qian"
set_KWS = "zhan qi lai"
flag_KWS = 0

flag_KWS_used = 0
flag_sv_used = 0

flag_sv_enroll = 0
thred_sv = 0.35

# 初始化 WebRTC VAD
vad = webrtcvad.Vad()
vad.set_mode(3)

# 视频流生成函数
def gen_frames(url):
    while True:
        try:
            # 使用与 get_frame_from_stream 相同的逻辑从 URL 获取视频帧
            video_url = url + '/video_feed' if not url.endswith('/video_feed') else url
            response = requests.get(video_url, stream=True)
            if response.status_code == 200:
                bytes_data = b''
                for chunk in response.iter_content(chunk_size=1024):
                    bytes_data += chunk
                    a = bytes_data.find(b'\xff\xd8')
                    b = bytes_data.find(b'\xff\xd9')
                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        img_array = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if frame is not None:
                            ret, buffer = cv2.imencode('.jpg', frame)
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print(f"[video_feed] Failed to get frame from {url}")
                time.sleep(1)
        except Exception as e:
            print(f"[video_feed] Error in gen_frames: {e}")
            time.sleep(1)

# 情感标签检测线程函数
def emotion_detection_thread(url):
    global emotion_queue, current_face_emotion_tag, current_face_VAD_tag
    # 初始化情感检测模型（只需调用一次）
    # 构建情感检测器并获取所需参数
    url, model, device, half, imgsz, stride, opt = build_emo_detector(url)
    
    while True:
        try:
            # 获取当前时间戳
            timestamp = time.time()
            
            # 调用RunInference函数进行情感检测
            face_emotion_tag, face_VAD_tag = Detect(url, model, device, half, imgsz, stride, opt)
            
            # 将face_emotion_tag转换为字符串（如果是列表）
            if face_emotion_tag is not None and isinstance(face_emotion_tag, list):
                face_emotion_tag = ",".join(face_emotion_tag)
            
            # 将face_VAD_tag转换为字符串（如果是numpy数组）
            if face_VAD_tag is not None and hasattr(face_VAD_tag, 'any'):
                face_VAD_tag = ",".join(str(v) for v in face_VAD_tag)
            
            # 将最新的情感标签和时间戳放入队列
            if face_emotion_tag and face_VAD_tag:
                emotion_queue.put((timestamp, face_emotion_tag, face_VAD_tag))
                # 更新UI显示的当前情感标签
                current_face_emotion_tag = face_emotion_tag
                current_face_VAD_tag = face_VAD_tag
                print(f"情感检测时间戳: {timestamp}, 标签: {face_emotion_tag}, {face_VAD_tag}")
            else:
                print("缺少情感tag")
            time.sleep(2)  # 每2秒检测一次
        except Exception as e:
            print(f"情感检测错误: {e}")
            time.sleep(1)  # 出错后等待1秒再重试

def extract_chinese_and_convert_to_pinyin(input_string):
    """
    提取字符串中的汉字，并将其转换为拼音。
    
    :param input_string: 原始字符串
    :return: 转换后的拼音字符串
    """
    # 使用正则表达式提取所有汉字
    chinese_characters = re.findall(r'[\u4e00-\u9fa5]', input_string)
    # 将汉字列表合并为字符串
    chinese_text = ''.join(chinese_characters)
    
    # 转换为拼音
    pinyin_result = pinyin(chinese_text, style=Style.NORMAL)
    # 将拼音列表拼接为字符串
    pinyin_text = ' '.join([item[0] for item in pinyin_result])
    
    return pinyin_text

from collections import deque

class HTTPAudioStream:
    def __init__(self, url, chunk_size=1024):
        self.response = requests.get(url, stream=True)
        self.buffer = deque()
        self.chunk_size = chunk_size
        self._iterator = self.response.iter_content(chunk_size=None)  # 不指定 chunk_size：让服务器自己控制
        self._cache = b''  # 缓存不足 chunk_size 的部分

    def read(self, size):
        while len(self._cache) < size:
            try:
                chunk = next(self._iterator)
                if chunk:
                    self._cache += chunk
            except StopIteration:
                break  # 流断了
        # 返回 size 长度的数据
        data, self._cache = self._cache[:size], self._cache[size:]
        return data

def audio_recorder_from_stream(url):
    global audio_queue, recording_active, last_active_time, segments_to_save, last_vad_end_time
    # p = pyaudio.PyAudio()
    # stream = p.open(format=pyaudio.paInt16,
    #                 channels=AUDIO_CHANNELS,
    #                 rate=AUDIO_RATE,
    #                 input=True,
    #                 frames_per_buffer=CHUNK)
    http_stream = HTTPAudioStream(URL)
    audio_buffer = []
    print("音频录制已开始")
    
    while recording_active:
        data = http_stream.read(CHUNK)
        audio_buffer.append(data)
        
        # 每 0.5 秒检测一次 VAD
        if len(audio_buffer) * CHUNK / AUDIO_RATE >= 0.5:
            # 拼接音频数据并检测 VAD
            raw_audio = b''.join(audio_buffer)
            vad_result = check_vad_activity(raw_audio)
            
            if vad_result:
                print("检测到语音活动")
                last_active_time = time.time()
                segments_to_save.append((raw_audio, time.time()))
            else:
                print("静音中...")
            
            audio_buffer = []  # 清空缓冲区
        
        # 检查无效语音时间
        if time.time() - last_active_time > NO_SPEECH_THRESHOLD:
            # 检查是否需要保存
            if segments_to_save and segments_to_save[-1][1] > last_vad_end_time:
                save_audio_video()
                last_active_time = time.time()
            else:
                pass
                # print("无新增语音段，跳过保存")

# 检测 VAD 活动
def check_vad_activity(audio_data):
    # 将音频数据分块检测
    num, rate = 0, 0.5
    step = int(AUDIO_RATE * 0.02)  # 20ms 块大小
    flag_rate = round(rate * len(audio_data) // step)

    for i in range(0, len(audio_data), step):
        chunk = audio_data[i:i + step]
        if len(chunk) == step:
            if vad.is_speech(chunk, sample_rate=AUDIO_RATE):
                num += 1

    if num > flag_rate:
        return True
    return False

# 保存音频和视频
def save_audio_video():
    global segments_to_save, last_vad_end_time, saved_intervals, current_stt_message

    # 全局变量，用于保存音频文件名计数
    global audio_file_count
    global flag_sv_enroll
    global set_SV_enroll

    if flag_sv_enroll:
        audio_output_path = f"{set_SV_enroll}/enroll_0.wav"
    else:
        audio_file_count += 1
        audio_output_path = f"{OUTPUT_DIR}/audio_{audio_file_count}.wav"

    if not segments_to_save:
        return
    
    # 停止当前播放的音频
    if _is_playing:
        stop_audio()
        print("检测到新的有效音，已停止当前音频播放")

    # 获取有效段的时间范围
    stt_start_time = segments_to_save[0][1]
    stt_end_time = segments_to_save[-1][1]
    
    # 检查是否与之前的片段重叠
    if saved_intervals and saved_intervals[-1][1] >= stt_start_time:
        print("当前片段与之前片段重叠，跳过保存")
        segments_to_save.clear()
        return
    
    # 保存音频
    audio_frames = [seg[0] for seg in segments_to_save]
    if flag_sv_enroll:
        audio_length = 0.5 * len(segments_to_save)
        if audio_length < 3:
            print("声纹注册语音需大于3秒，请重新注册")
            return 1
    wf = wave.open(audio_output_path, 'wb')
    wf.setnchannels(AUDIO_CHANNELS)
    wf.setsampwidth(2)  # 16-bit PCM
    wf.setframerate(AUDIO_RATE)
    wf.writeframes(b''.join(audio_frames))
    wf.close()
    print(f"音频保存至 {audio_output_path}")

    base_url, payload, headers = build_agent()

    if flag_sv_enroll:
        text = "声纹注册完成！现在只有你可以命令我啦！"
        print(text)
        flag_sv_enroll = 0
        system_introduction(text)
    else:
        # 使用线程执行推理
        inference_thread = threading.Thread(target=Inference, args=(audio_output_path, base_url, payload, headers, stt_start_time, stt_end_time))
        inference_thread.start()
        
        # 记录保存的区间
        saved_intervals.append((stt_start_time, stt_end_time))
        
    # 清空缓冲区
    segments_to_save.clear()

# --- 播放音频 -
def real_play(file_path):
    global _is_playing
    _is_playing = True
    while True:
        if _stop_event.is_set():
            break
        pass
        time.sleep(0.1)
    _is_playing = False
    _stop_event.clear()

def play_audio(file_path):
    global _play_thread, _stop_event, _is_playing
    # 如果已经在播放，先停止
    if _is_playing:
        stop_audio()
    _stop_event.clear()
    _play_thread = threading.Thread(target=real_play, args=(file_path,))
    _play_thread.start()

def stop_audio():
    global _stop_event, _is_playing
    if _is_playing:
        _stop_event.set()  # 发出中断信号
        if _play_thread is not None:
            _play_thread.join()  # 等待播放线程退出

async def amain(TEXT, VOICE, OUTPUT_FILE) -> None:
    """Main function"""
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(OUTPUT_FILE)

def is_folder_empty(folder_path):
    """
    检测指定文件夹内是否有文件。
    
    :param folder_path: 文件夹路径
    :return: 如果文件夹为空返回 True，否则返回 False
    """
    # 获取文件夹中的所有条目（文件或子文件夹）
    entries = os.listdir(folder_path)
    # 检查是否存在文件
    for entry in entries:
        # 获取完整路径
        full_path = os.path.join(folder_path, entry)
        # 如果是文件，返回 False
        if os.path.isfile(full_path):
            return False
    # 如果没有文件，返回 True
    return True


# -------- SenceVoice 语音识别 --模型加载-----
model_dir = "./models/SenseVoiceSmall"
model_senceVoice = AutoModel( model=model_dir, trust_remote_code=True, )

# -------- CAM++声纹识别 -- 模型加载 --------
set_SV_enroll = r'.\SpeakerVerification_DIR\enroll_wav\\'
sv_pipeline = pipeline(
    task='speaker-verification',
    model='damo/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

class ChatMemory:
    def __init__(self, max_length=2048):
        self.history = []
        self.max_length = max_length  # 最大输入长度

    def add_to_history(self, user_input, model_response):
        """
        添加用户输入和模型响应到历史记录。
        """
        self.history.append(f"User: {user_input}")
        self.history.append(f"system: {model_response}")

    def get_context(self):
        """
        获取拼接后的对话上下文。
        """
        context = "\n".join(self.history)
        # 截断上下文，使其不超过 max_length
        if len(context) > self.max_length:
            context = context[-self.max_length :]
        return context
    
# -------- memory 初始化 --------
memory = ChatMemory(max_length=512)

def system_introduction(text):
    global audio_file_count, ai_response
    global folder_path
    text = text
    # 更新UI显示的AI回复
    ai_response = text
    # 使用TTS生成音频文件
    used_speaker = "zh-CN-XiaoyiNeural"
    asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_tmp_{audio_file_count}.mp3")))
    # 不再直接播放，而是通过Web界面播放
    print(f"音频已保存至 {folder_path}/sft_tmp_{audio_file_count}.mp3，等待Web界面播放")

# 获取指定时间范围内的情感标签
def get_emotion_tags_in_timerange(start_time, end_time):
    # 添加一个时间缓冲区，例如前后各扩展1秒
    buffer_time = 5.0
    adjusted_start_time = start_time - buffer_time
    adjusted_end_time = end_time + buffer_time

    global emotion_queue
    emotion_tags_list = []
    face_vad_tags_list = []
    
    # 创建一个临时队列来存储处理过的情感标签
    temp_queue = Queue()
    
    # 遍历队列中的所有元素
    while not emotion_queue.empty():
        item = emotion_queue.get()
        timestamp, face_emotion_tag, face_vad_tag = item
        
        # 将所有元素放回临时队列
        temp_queue.put(item)
        
        # 如果时间戳在指定范围内，添加到结果列表
        if adjusted_start_time <= timestamp <= adjusted_end_time:
            emotion_tags_list.append(face_emotion_tag)
            face_vad_tags_list.append(face_vad_tag)
    
    # 将临时队列中的元素放回原队列
    while not temp_queue.empty():
        emotion_queue.put(temp_queue.get())
    
    # 如果没有找到任何情感标签，返回空字符串
    if not emotion_tags_list:
        return "", ""
    
    # 合并所有情感标签
    combined_emotion_tags = ",".join(emotion_tags_list)
    combined_vad_tags = ",".join(face_vad_tags_list)
    
    return combined_emotion_tags, combined_vad_tags

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
        if "Description" in config:
            Description = config["Description"]
    return Description

def build_agent():
    base_url = "https://cloud.infini-ai.com/maas/v1/chat/completions"
    API_KEY = "sk-u7h7tyejibhlmpya"

    # 添加系统消息来设置模型的行为和角色
    agent_info_path = "/root/mood_engine/config/default/agent_info.json"
    system_message = read_config(agent_info_path)
    system_message += """
    在回复用户时，请遵循以下思维链：
    
    1. 精准个人信息提取
    多源信息融合：既要深入分析已有的聊天记录与用户的文本内容，还需结合用户信息中已有的信息，确保对用户个人信息进行全面且准确的挖掘与更新。除了生日、喜好、讨厌的事物外，还可关注用户的职业、兴趣爱好类别、日常活动习惯等方面的信息。
    信息验证与整理：对挖掘到的信息进行验证，去除可能存在的错误或模糊信息。将提取到的信息按照不同的类别进行整理，构建一个清晰的用户信息库，方便后续使用。
    
    2. 深度情绪需求分析
    多模态情绪判断：综合考虑情绪标签和VAD情绪状态模型标签所提供的信息，同时结合已有聊天记录和用户输入中的文本情感倾向，对用户当前的情绪状态进行更全面、准确的判断。例如，注意文本中的语气词、感叹号等表达情感强度的元素。
    情绪分类细化：除了简单判断积极、消极、中性等基本情绪状态外，进一步细化情绪分类，如兴奋、焦虑、悲伤、愤怒等，以便更精准地把握用户的情绪需求。
    情绪趋势分析：分析用户情绪在对话过程中的变化趋势，判断情绪是逐渐缓和、加剧还是保持稳定，从而更好地预测用户后续的情绪反应。
    
    3. 个性化情绪价值创造
    多因素综合考量：依据整理好的用户个人信息、精准判断的用户当前情绪状态，以及用户输入的具体内容，综合考虑多个因素来生成具有针对性的情绪价值反馈输出。例如，如果用户表达了对某部电影的喜爱且情绪积极，可结合其兴趣爱好推荐相关类型的电影。
    反馈方式多样化：根据用户的情绪状态和个人偏好，选择合适的反馈方式，如幽默风趣的回应、安慰鼓励的话语、理性客观的分析等。同时，注意语言表达的风格和用词，确保反馈能够有效地传递情绪价值。
    动态调整反馈：在对话过程中，根据用户的实时反馈和情绪变化，动态调整反馈内容和方式，以保持与用户情绪的同步，持续为用户创造良好的情绪体验。
    
    最终，只返回一个你认为最合适的回应，不要返回思考过程或其他可选择的回应。输出格式必须按照以下格式：
    角色回复: 只包含当前角色设定下模型对用户输入的回复话语，而不包含旁白类的分析文本
    """

    payload = {
        "model": "deepseek-v3",
        "messages": [
            {
                "role": "system",
                "content": system_message
            }
        ]
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
    }

    return base_url, payload, headers

def generate_prompt(user_info, face_emotion_tag, face_VAD_tag, stt_message):
    
    face_emotion_tag = face_emotion_tag or "neutral"
    face_VAD_tag = face_VAD_tag or "0.5,0.5,0.5"
    
    prompt_content = f"""
        用户的个人信息: {user_info}

        情绪状态tag: {face_emotion_tag}
        VAD情绪状态模型tag: {face_VAD_tag}，其中的值对应分别为V,A,D
        愉悦度(V): 用于衡量情绪的积极程度或愉悦程度，范围从消极到积极;
        唤醒度(A): 用于衡量人的兴奋程度，范围从不活跃/平静到兴奋/准备行动;
        支配度(D): 用于衡量一个人对情况的控制程度，范围从顺从/无控制到支配/在控制之中。

        用户的最新语音输入: {stt_message}

        请根据以上信息与已知的思维链生成一个合适的回复
    """

    return prompt_content

def Inference(TEMP_AUDIO_FILE=f"{OUTPUT_DIR}/audio_0.wav", base_url=None, payload=None, headers=None, stt_start_time=0, stt_end_time=0):
    '''
    1. 使用senceVoice做asr，转换为拼音，检测唤醒词
        - 首先检测声纹注册文件夹是否有注册文件，如果无，启动声纹注册
    2. 使用CAM++做声纹识别
        - 设置固定声纹注册语音目录，每次输入音频均进行声纹对比
    3. 以上两者均通过，则进行大模型推理
    '''
    global audio_file_count, current_stt_message, ai_thinking, ai_response

    global set_SV_enroll
    global flag_sv_enroll
    global thred_sv
    global flag_sv_used

    global set_KWS
    global flag_KWS
    global flag_KWS_used

    global folder_path

    user_info_path = "/root/mood_engine/config/default/user_info.json"
    user_info = read_config(user_info_path)

    face_emotion_tag, face_VAD_tag = get_emotion_tags_in_timerange(stt_start_time, stt_end_time)
    print(f"音频时间范围: {stt_start_time} - {stt_end_time}")
    print(f"获取到的情感标签: {face_emotion_tag}, {face_VAD_tag}")
    
    os.makedirs(set_SV_enroll, exist_ok=True)
    # --- 如果开启声纹识别，且声纹文件夹为空，则开始声纹注册。设定注册语音有效长度需大于3秒
    if flag_sv_used and is_folder_empty(set_SV_enroll):
        text = f"无声纹注册文件！请先注册声纹，需大于三秒哦~"
        print(text)
        system_introduction(text)
        flag_sv_enroll = 1
    
    else:
        # -------- SenceVoice 推理 ---------
        input_file = (TEMP_AUDIO_FILE)
        res = model_senceVoice.generate(
            input=input_file,
            cache={},
            language="auto", # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
        )
        stt_message = res[0]['text'].split(">")[-1]
        # 更新UI显示的语音输入
        current_stt_message = stt_message
        stt_pinyin = extract_chinese_and_convert_to_pinyin(stt_message)
        print(stt_message, stt_pinyin)

        # --- 判断是否启动KWS
        if not flag_KWS_used:
            flag_KWS = 1
        if not flag_KWS:
            if set_KWS in stt_pinyin:
                flag_KWS = 1
        
        # --- KWS成功，或不设置KWS
        if flag_KWS:
            sv_score = sv_pipeline([os.path.join(set_SV_enroll, "enroll_0.wav"), TEMP_AUDIO_FILE], thr=thred_sv)
            print(sv_score)
            sv_result = sv_score['text']
            if flag_sv_used == 0:
                sv_result = "yes"
            if sv_result == "yes":
                if stt_message and stt_message.strip():
                    # --- 读取历史对话 ---
                    context = memory.get_context()
                    prompt_template = generate_prompt(user_info, face_emotion_tag, face_VAD_tag, stt_message)
                    prompt = f"{context}\nUser:{prompt_template}\n"
                    print("ASR OUT:", stt_message)
                    
                    # 更新UI显示的AI思考
                    ai_thinking = "正在分析用户输入和情感状态..."

                    # 更新payload以包含历史对话和当前提示
                    updated_payload = payload.copy()
                    
                    # 保留系统消息
                    system_message = updated_payload["messages"][0]
                    
                    # 构建新的消息列表
                    updated_payload["messages"] = [
                        system_message,  # 系统消息始终在最前面
                        {"role": "user", "content": prompt}  # 当前提示（包含历史对话）
                    ]

                    response = requests.post(base_url, json=updated_payload, headers=headers)
                    output_text = response.json()["choices"][0]["message"]["content"]
                    print("answer", output_text)
                    
                    # 更新UI显示的AI回复
                    ai_response = output_text

                    # -------- 更新记忆库 -----
                    memory.add_to_history(stt_message, output_text)

                    # 输入文本
                    text = output_text
                    # 语种识别 -- langid
                    language, confidence = langid.classify(text)

                    language_speaker = {
                    "ja" : "ja-JP-NanamiNeural",            # ok
                    "fr" : "fr-FR-DeniseNeural",            # ok
                    "es" : "ca-ES-JoanaNeural",             # ok
                    "de" : "de-DE-KatjaNeural",             # ok
                    "zh" : "zh-CN-XiaoyiNeural",            # ok
                    "en" : "en-US-AnaNeural",               # ok
                    }

                    if language not in language_speaker.keys():
                        used_speaker = "zh-CN-XiaoyiNeural"
                    else:
                        used_speaker = language_speaker[language]
                        print("检测到语种：", language, "使用音色：", language_speaker[language])

                    # 使用TTS生成音频文件
                    # asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_{audio_file_count}.mp3")))
                    # play_audio(f'{folder_path}/sft_{audio_file_count}.mp3')
                    asyncio.run(amain(text, used_speaker, os.path.join(folder_path,f"sft_{audio_file_count}.mp3")))
                    # 不再直接播放，而是通过Web界面播放
                    print(f"音频已保存至 {folder_path}/sft_{audio_file_count}.mp3，等待Web界面播放")
                else:
                    print("STT消息为空，跳过大模型推理")
            else:
                text = "很抱歉，声纹验证失败，我无法为您服务"
                print(text)
                system_introduction(text)
        else:
            text = "很抱歉，唤醒词错误，请说出正确的唤醒词哦"
            system_introduction(text)

# 添加全局变量来跟踪已播放的音频文件

audio_play_pointer = 1 
played_audio_files = set()
# 获取最新音频文件
@app.route('/latest-audio')
def latest_audio():
    global audio_play_pointer, audio_file_count, played_audio_files
    try:
        while True:
            files_exist = any(f.endswith('.mp3') for f in os.listdir(folder_path))
            if files_exist:
                break
            time.sleep(0.5)
        # 构造当前应该播放的音频文件名
        current_audio_file = f"sft_{audio_play_pointer}.mp3"
        # 检查文件是否存在且未播放过
        audio_path = os.path.join(folder_path, current_audio_file)
        if os.path.exists(audio_path) and current_audio_file not in played_audio_files:
            # 标记为已播放
            played_audio_files.add(current_audio_file)
            audio_play_pointer += 1
            print("audio_play_pointer-----------------------------------:", audio_play_pointer)
            return jsonify({
                'filename': current_audio_file, 
                'path': audio_path
            })
        else:
            # 检查是否有系统提示音频（sft_tmp_开头）且未播放过
            tmp_files = [f for f in os.listdir(folder_path) 
                        if f.startswith('sft_tmp_') and f.endswith('.mp3') 
                        and f not in played_audio_files]
            if tmp_files:
                # 按文件名排序，获取最新的系统提示音频
                latest_tmp = sorted(tmp_files)[-1]
                # 标记为已播放
                played_audio_files.add(latest_tmp)
                return jsonify({
                    'filename': latest_tmp, 
                    'path': os.path.join(folder_path, latest_tmp)
                })
            else:
                return jsonify({'filename': None})
    except Exception as e:
        return jsonify({'error': str(e)})

# 提供音频文件
@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(folder_path, filename)

# 获取当前状态的API
@app.route('/api/status')
def get_status():
    return jsonify({
        'face_emotion_tag': current_face_emotion_tag,
        'face_VAD_tag': current_face_VAD_tag,
        'stt_message': current_stt_message,
        'ai_thinking': ai_thinking,
        'ai_response': ai_response
    })

# 视频流接口
@app.route('/video_feed')
def video_feed():
    global emo_url
    return Response(gen_frames(emo_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# 首页界面
@app.route('/')
def index():
    return render_template('start_index.html',
                          agent_name="情感智能助手",
                          version="1.0.0")

# 主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='情感智能助手')
    parser.add_argument('--url', type=str, default='https://3ead628b2e17.ngrok-free.app', help='情感流URL')
    parser.add_argument('--port', type=int, default=3000, help='WebUI端口')
    parser.add_argument('--ngrok', action='store_true', help='使用ngrok进行端口映射')
    args = parser.parse_args()
    
    emo_url = args.url
    print(f"情感检测URL: {emo_url}")
    
    try:
        # 如果需要使用ngrok进行端口映射
        if args.ngrok:
            try:
                # 启动ngrok隧道
                public_url = ngrok.connect(args.port).public_url
                print(f"Ngrok隧道已启动: {public_url}")
            except Exception as e:
                print(f"启动ngrok隧道时出错: {e}")
                print("继续启动本地服务器...")
        
        # 启动音视频录制线程
        audio_thread = threading.Thread(target=audio_recorder_from_stream, args=(args.url,), daemon=True)
        audio_thread.start()

        # 启动情感检测线程
        emotion_thread = threading.Thread(target=emotion_detection_thread, args=(args.url,), daemon=True)
        emotion_thread.start()

        flag_info = f'{flag_sv_used}-{flag_KWS_used}'
        dict_flag_info = {
            "1-1": "您已开启声纹识别和关键词唤醒，",
            "0-1":"您已开启关键词唤醒",
            "1-0":"您已开启声纹识别",
            "0-0":"",
        }
        if flag_sv_used or flag_KWS_used:
            text = dict_flag_info[flag_info]
            system_introduction(text)

        print(f"启动WebUI，端口: {args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)
    
    except KeyboardInterrupt:
        print("程序停止中...")
        recording_active = False
        if 'audio_thread' in locals():
            audio_thread.join()
        print("程序已停止")