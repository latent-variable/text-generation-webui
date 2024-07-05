@echo off
CALL ".\installer_files\conda\condabin\conda.bat" activate
CALL conda activate ".\installer_files\env"
CALL python ".\00_my_scripts\whisperx_realtime.py"
cmd /k