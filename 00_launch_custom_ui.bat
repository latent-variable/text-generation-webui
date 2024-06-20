@echo off
CALL ".\installer_files\conda\condabin\conda.bat" activate
CALL conda activate ".\installer_files\env"
cd ".\00_my_scripts"
CALL python "custom_ui.py"
cmd /k