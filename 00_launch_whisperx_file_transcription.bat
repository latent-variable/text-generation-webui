@echo off
CALL ".\installer_files\conda\condabin\conda.bat" activate
CALL conda activate ".\installer_files\env"
CALL python ".\extensions\whisperx_stt\script.py"
cmd /k