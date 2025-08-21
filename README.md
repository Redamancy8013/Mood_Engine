##### Mood Engine

#### 1. Code Download

```
git clone https://github.com/Redamancy8013/Mood_Engine.git
```

And then, you need to download voice models and visual models from [Baidu CloudDisk](https://pan.baidu.com/s/1GMb3UpshLdLSKmoqs88PuQ?pwd=mdeg).

The ```model.pt``` file is located at  ```/models/SenseVoiceSmall/```

The ```models``` directory is located at ```/visual/emotic/debug_exp/```

#### 2. Environment Build

```
conda env create -f environment.yml
```

If there is something wrong with build the environment, follow the instruction below:

```
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```
```
pip install edge-tts==6.1.17 funasr==1.1.12 ffmpeg==1.4 opencv-python==4.10.0.84 transformers==4.45.2 webrtcvad==2.0.10 qwen-vl-utils==0.0.8 pygame==2.6.1 langid==1.1.6 langdetect==1.0.9 accelerate==0.33.0 PyAudio==0.2.14
```
```
conda install -c conda-forge pynini=2.1.6
pip install WeTextProcessing --no-deps
```
```
pip install HyperPyYAML==1.2.2 modelscope==1.15.0 onnxruntime==1.19.2 openai-whisper==20231117 importlib_resources==6.4.5 sounddevice==0.5.1 matcha-tts==0.0.7.0
```

And if there still some packages remaining to be installed, just ```pip install xxx```

And then, activate your environment:

```
conda activate xxx
```
