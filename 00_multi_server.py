import os
import sys
import time
import subprocess

script_dir = os.getcwd()
conda_env_path = os.path.join(script_dir, "installer_files", "env")


# Command-line flags
def get_flag_list():
   
    flags = []
    cmd_flags_path = os.path.join(script_dir, "CMD_FLAGS-Multi-Server.txt")
    if os.path.exists(cmd_flags_path):
        with open(cmd_flags_path, 'r') as f:
            for line in f:
                if line.strip().rstrip('\\').strip() and not line.strip().startswith('#'):
                    CMD_FLAGS = line.strip().rstrip('\\').strip() 
                    flags.append(CMD_FLAGS)
                    
    return flags

def is_windows():
    return sys.platform.startswith("win")

def run_cmd(cmd, assert_success=False, environment=False ):
    # Use the conda environment
    if environment:
        if is_windows():
            conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
            cmd = f'"{conda_bat_path}" activate "{conda_env_path}" >nul && {cmd}'
        else:
            conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
            cmd = f'. "{conda_sh_path}" && conda activate "{conda_env_path}" && {cmd}'

    # Run shell commands
    result = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print(f"Command '{cmd}' failed with exit status code '{str(result.returncode)}'.\n\nExiting now.\nTry running the start/update script again.")
        sys.exit(1)

    return result

# Here's how you can modify the `launch_webuis` function to use the conda environment:
def launch_webuis():
    flag_list = get_flag_list()
    for flags in flag_list:
        print(f"Launching webui with flags: {flags}")
        run_cmd(f"python server.py {flags}", environment=False)
        time.sleep(30)
    
    
if __name__ == "__main__":
    launch_webuis()
    