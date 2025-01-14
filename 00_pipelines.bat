rem Activate conda environment
CALL ".\installer_files\conda\condabin\conda.bat" activate
CALL conda activate ".\installer_files\env"

rem Run the pipeline application
CALL cd "./pipelines"
CALL "start.bat"

rem Keep the command prompt open
cmd /k
