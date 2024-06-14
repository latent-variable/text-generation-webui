@echo off
CALL ".\installer_files\conda\condabin\conda.bat" activate
CALL conda activate ".\installer_files\env"
CALL python ".\extensions\whisper_rt\script.py"
cmd /k