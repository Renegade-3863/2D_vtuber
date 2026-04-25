@echo off
REM Launch the THA3 AI VTuber in microphone-input mode.
cd /d "%~dp0\.."
set THA_INPUT_MODE=mic
python -m ai_runtime.tha_lip_sync_render %*
