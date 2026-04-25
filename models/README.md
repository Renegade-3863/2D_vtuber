# Pretrained models (not bundled in repo)

Two model groups need to be downloaded manually. Total ≈ 0.6 GB.

## 1. Faster-Whisper (used by STT and WhisperX forced alignment)

Download `faster-whisper-small` from Hugging Face:

```bash
# inside the repo root
git lfs install
git clone https://huggingface.co/Systran/faster-whisper-small models/faster-whisper-small
```

Or download manually and place files under `models/faster-whisper-small/`:
- `model.bin`
- `config.json`
- `tokenizer.json`
- `vocabulary.txt`

## 2. U²-Net (used by `helpers/remove_background.py`)

Download `u2net.onnx` (~168 MB) from the official U²-Net release and place it at:

```
models/u2net/u2net.onnx
```

Source: https://github.com/xuebinqin/U-2-Net

> If you only run the chat / render pipeline (`ai_runtime/tha_lip_sync_render.py`)
> you can skip U²-Net. It is only required when preparing new character portraits.
