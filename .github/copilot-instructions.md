# Copilot Instructions for 2D_vtuber

## Project Overview

**2D_vtuber** is an end-to-end AI avatar pipeline that brings a 2D anime character to life via:
- **User Input**: Text chat or voice (microphone capture)
- **LLM Brain**: Structured JSON control signal (emotion, intensity, motion hints)
- **TTS Voice**: Pluggable TTS engines (CosyVoice, Edge-TTS, XTTS, GPT-SoVITS) with phoneme-accurate lip-sync
- **Neural Renderer**: Talking-Head-Anime-3 (THA3) @ 60 FPS on CUDA GPU

See [README.md](README.md) for feature highlights and [docs/architecture.md](docs/architecture.md) for detailed system diagrams.

---

## Essential Development Knowledge

### 1. Architecture: Two-Layer Design

This project uses a **clear separation of concerns**:

| Layer | Responsibility | Location | Outputs |
|-------|-----------------|----------|---------|
| **ai_runtime** | LLM brain, TTS pipeline, STT capture, animation timing | `ai_runtime/` | Pose vectors, audio timing, interaction state |
| **tha3** | Bundled THA3 neural renderer (original: MIT license) | `tha3/` | 60-fps animated character frames |

**Golden Rule**: `ai_runtime` computes **what** to animate (_pose vectors_, emotion, mouth shape); `tha3` handles **how** to render (_GPU inference_, smoothing, frame output).

### 2. Main Entry Point: `tha_lip_sync_render.py`

```python
# Run the interactive avatar demo
python -m ai_runtime.tha_lip_sync_render [--text | --mic]
```

- **Thread 1 (input_loop)**: Reads user text/voice, calls LLM, queues TTS audio
- **Thread 2 (render_loop)**: Runs ~20–60 FPS, pulls shared `RenderState`, updates pose, renders with THA3
- **RenderState contract** (thread-safe via lock): `emotion`, `intensity`, `motion_hint`, `mouth_shape`, `speaking` flag
- Shared state machine: `IDLE` → `LISTENING` → `THINKING` → `SPEAKING` → `POST_SPEAK`

### 3. Data Contracts (Invariants)

#### LLM Output JSON Schema
LLM is constrained to respond with exactly:
```json
{
  "reply": "string",
  "emotion": "neutral | happy | sad | angry | surprised | thinking",
  "intensity": 0.0..1.0,
  "motion_hint": "none | nod | shake | tilt_left | tilt_right"
}
```
**Implementation**: `ai_runtime/llm_api_client.py` uses Pydantic + OpenAI API structured outputs.

#### Phoneme–Viseme Pipeline
1. **TTS engines** produce audio + word/phoneme boundaries (or timestamps)
2. **WhisperX** aligns if engine doesn't provide native boundaries
3. **audio_phoneme_aligner.py**: Maps phoneme timeline to viseme frames (5D mouth space: aaa/iii/uuu/eee/ooo)
4. **phoneme_mouth_mapper.py**: Converts viseme → THA3 mouth parameters
5. **tha_pose_mapper.py**: Merges LLM state + mouth + motion hint → 45D THA3 pose vector

### 4. Configuration (`.env` Required)

Copy `.env.example` (if it exists) or create `.env` at repo root:

```bash
# LLM Provider (Azure OpenAI or Aliyun Qwen)
LLM_PROVIDER=azure  # or qwen
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=<endpoint>
OPENAI_MODEL_NAME=gpt-4-turbo

# TTS Engine (cosyvoice, edge, xtts, gpt_sovits)
TTS_ENGINE=cosyvoice
DASHSCOPE_API_KEY=<key>

# Optional: Microphone mode
THA_INPUT_MODE=text  # or mic

# Optional: Character portrait override (default: testPic_tha3_ready.png)
THA_PORTRAIT=data/images/my_character.png

# Optional: Debug phoneme/alignment traces
THA_LIPSYNC_DEBUG=1
```

**Key module**: `ai_runtime/config_api.py` — loads and validates `.env` via `APIConfig` dataclass.

### 5. Model Downloads (Manual, Not Bundled)

After `pip install -r requirements.txt`, download:

1. **THA3 weights** (~1 GB, required):
   - Download from https://github.com/pkhungurn/talking-head-anime-3-demo
   - Extract into `data/models/` so subdirs like `data/models/standard_half/` exist

2. **Faster-Whisper** (optional, for alignment fallback):
   - See `models/README.md` for download instructions

3. **U²-Net** (optional, for portrait preprocessing):
   - See `models/README.md`

### 6. Development Environment Constraints

- **GPU required**: CUDA 11.8 or 12.1 (no CPU fallback)
- **Python**: 3.10+ (tested 3.10)
- **OS**: Windows 10/11 preferred; Linux supported but needs ALSA + pyaudio dev libs
- **Audio**:
  - Windows: Typically works out-of-box
  - Linux: Install `libasound2-dev`, `libportaudio2` before `pip install pyaudio`
  - macOS: Untested; may need Homebrew audio libs

### 7. Code Organization & Patterns

#### ai_runtime Entry Points
- `tha_lip_sync_render.py::main()` — Interactive demo (text or voice)
- Other clients (`llm_api_client.py`, `tts_client.py`, `stt_client.py`) are **self-contained**; test independently

#### Testing & Debugging
- **Standalone tests**: `tests/` folder has no test framework; run scripts directly
  - `test_llm.py` — Validate LLM client
  - `_test_xtts_e2e.py` — TTS end-to-end validation
  - `_test_imports.py` — Check all dependencies installed
- **Debug flag**: `THA_LIPSYNC_DEBUG=1` prints per-phoneme alignment traces to stdout

#### Idle Animation System
- `tha_pose_mapper.py` coordinates multiple animators:
  - `IdleAnimator` — breathing, blinks, saccades
  - `GestureAnimator` — motion hint → head/body rotation
  - `EmotionSmoother` — smooth transitions when emotion changes
  - `PupilAnimator` — iris motion
- Pattern: Each animator computes pose delta independently, then merged into final 45D vector

#### TTS Pluggability
All TTS engines inherit from `BaseTTSClient` (duck-typed via `tts_client.py`):
- `edge` — Microsoft Edge TTS (free, fast, native word boundaries)
- `cosyvoice` — Aliyun DashScope (recommended, cloud, native phoneme timestamps)
- `xtts` — Coqui XTTS v2 (local GPU, voice cloning, needs alignment)
- `gpt_sovits` — GPT-SoVITS (local API, voice cloning, needs alignment)

**Extension pattern**: Add new engine → subclass `TTSClient`, implement `synthesize(text) → audio, word_boundaries`, register in `tts_client.py::get_tts_client()`.

---

## Common Tasks & Recipes

### Task: Run the Avatar Demo
```bash
# Text mode (default)
python -m ai_runtime.tha_lip_sync_render --text

# Voice mode (microphone)
python -m ai_runtime.tha_lip_sync_render --mic
```
Ensure `.env` is configured and THA3 model weights are in `data/models/`.

### Task: Test TTS Pipeline Independently
```bash
python tests/test_llm.py          # Validate LLM integration
python tests/_test_xtts_e2e.py    # Test XTTS with alignment
```

### Task: Debug Lip-Sync Timing
```bash
THA_LIPSYNC_DEBUG=1 python -m ai_runtime.tha_lip_sync_render --text
```
Look for `[ls-peak]`, `[align]`, `[lipsync]` trace lines.

### Task: Add a New Idle Animation
1. Create a new animator class in `ai_runtime/tha_pose_mapper.py` (follow `IdleAnimator` pattern)
2. Instantiate in `THAPoseMapper.__init__()`
3. Call `animator.update_pose()` in the merge loop (around line 200)

### Task: Extend with a New TTS Engine
1. Implement `synthesize(text) → (audio, word_boundaries)` in new client class
2. Register in `tts_client.py::get_tts_client()` switch statement
3. Add env var to `.env` example
4. Test with `tests/_test_<engine>_e2e.py`

### Task: Switch LLM Providers
Update `.env`:
```bash
LLM_PROVIDER=qwen  # or azure
DASHSCOPE_API_KEY=<your key>
```
No code changes needed; `llm_api_client.py` handles provider selection.

---

## Critical Gotchas & Anti-Patterns

### ❌ Don't: Modify THA3 Source Without Forking
- `tha3/` is bundled from upstream (PR reviewed separately)
- Extensions go in `ai_runtime/`, not inside `tha3/`

### ❌ Don't: Assume Pose Synchronization Is Instant
- Render loop runs ~60 FPS; input loop posts async updates to `RenderState`
- Use the smoothing filters (`EMA`, slew limits) in `tha_pose_mapper.py` to avoid jitter

### ❌ Don't: Block the Render Loop
- Render thread calls `poser.pose(image, vector)` every frame (~16 ms budget on 60 FPS)
- Heavy computation (alignment, TTS) must happen in input loop or background threads

### ❌ Don't: Hardcode API Keys
- Always use `.env` + `config_api.py`; `.env` is git-ignored for security

### ❌ Don't: Ignore GPU Memory When Choosing THA3 Model
- `standard_half` (2 GB) vs `separable_half` (1 GB); pick based on VRAM
- Test on your target GPU before shipping

---

## File Navigation Quick Reference

| Path | Purpose |
|------|---------|
| [ai_runtime/tha_lip_sync_render.py](ai_runtime/tha_lip_sync_render.py) | **Main entry**, thread coordination |
| [ai_runtime/tha_pose_mapper.py](ai_runtime/tha_pose_mapper.py) | LLM state + phoneme → 45D pose |
| [ai_runtime/llm_api_client.py](ai_runtime/llm_api_client.py) | LLM interface (structured JSON) |
| [ai_runtime/tts_client.py](ai_runtime/tts_client.py) | TTS factory + audio utils |
| [ai_runtime/audio_phoneme_aligner.py](ai_runtime/audio_phoneme_aligner.py) | Phoneme timeline generation |
| [ai_runtime/config_api.py](ai_runtime/config_api.py) | `.env` config loading |
| [tha3/poser/poser.py](tha3/poser/poser.py) | THA3 renderer (neural inference) |
| [docs/architecture.md](docs/architecture.md) | System architecture diagrams + contracts |
| [README.md](README.md) | Setup, feature highlights, model download links |

---

## Useful Links

- **Upstream THA3 Project**: https://github.com/pkhungurn/talking-head-anime-3-demo
- **Edge-TTS**: https://github.com/rany2/edge-tts
- **Coqui XTTS v2**: https://github.com/coqui-ai/TTS
- **WhisperX Alignment**: https://github.com/m-bain/whisperx
-  **Aliyun DashScope TTS**: https://help.aliyun.com/zh/dashscope/developer-reference/

---

**Last Updated**: May 2026
**Maintained by**: 2D_vtuber team
