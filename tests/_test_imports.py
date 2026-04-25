"""Quick smoke test for the new TTS stack."""
import sys
import time

print("Python:", sys.version.split()[0])

t0 = time.time()
import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} (in {time.time()-t0:.1f}s)")

t0 = time.time()
import transformers
print(f"transformers {transformers.__version__} (in {time.time()-t0:.1f}s)")

t0 = time.time()
from TTS.api import TTS
print(f"coqui TTS imported (in {time.time()-t0:.1f}s)")

t0 = time.time()
import whisperx
print(f"whisperx imported (in {time.time()-t0:.1f}s)")

t0 = time.time()
import imageio_ffmpeg
print(f"ffmpeg at: {imageio_ffmpeg.get_ffmpeg_exe()}")
print(f"all imports successful")
