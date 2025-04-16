# Automatic Speech Recognition (ASR) and Speech Translation Models for Batch Video Transcription and Machine Translation

Code for the blog post, [Automatic Speech Recognition (ASR) and Speech Translation Models for Batch Video Transcription and Machine Translation](https://garystafford.medium.com/automatic-speech-recognition-asr-and-speech-translation-models-for-batch-video-transcription-and-243ca34bed06): Learn to use smaller task-specific open-weight transformer models to batch transcribe and translate speech from videos.

## Prepare Windows Environment

### Computational Requirement

For the post, I hosted the models locally on an Intel Core i9 Windows 11-based workstation with a NVIDIA RTX 4080 SUPER graphics card containing 16 GB of GDDR6X memory (VRAM). Based on my experience, a minimum of 16 GB of GPU memory is required to run these models.

### Prerequisites

To follow along with this post, please make sure you have installed the free [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) related to C++.

A free Hugging Face account and [User Access Token](https://huggingface.co/docs/hub/security-tokens) are required for access. If you do not download the models in advance, they will be downloaded into the local cache the first time the application loads them.

I tested this code with the latest version of Python 3.12 (3.12.9).

### Install FFmpeg

Download, unzip, and add FFmpeg bin path to PATH.

### Download and Cache Models

```bat
python -m pip install "huggingface_hub[cli]" --upgrade

huggingface-cli login --token <your_hf_token> --add-to-git-credential

huggingface-cli download distil-whisper/distil-large-v3.5
huggingface-cli download openai/whisper-tiny
huggingface-cli download openai/whisper-small
huggingface-cli download openai/whisper-medium
huggingface-cli download openai/whisper-large-v3-turbo
huggingface-cli download openai/whisper-large-v3

huggingface-cli download facebook/nllb-200-distilled-1.3B
```

### Create Python Virtual Environment and Install Dependencies

```bat
python --version

python -m venv .venv
.venv\Scripts\activate

python -m pip install pip --upgrade
python -m pip install -r requirements.txt --upgrade
python -m pip install flash-attn --no-build-isolation --upgrade
```

## Run Script

```bat
py check_gpu_config.py

py asr_demo.py

REM optionally, add model selection as command line argument (0-5)
py asr_demo.py 3
```

## Deactivate and Delete Python Virtual Environment

```bat
deactivate
rmdir /s .venv
```

```bat
C:\Users\garya\.cache\
rmdir /s huggingface
```

---

_The contents of this repository represent my viewpoints and not of my past or current employers, including Amazon Web Services (AWS). All third-party libraries, modules, plugins, and SDKs are the property of their respective owners._
