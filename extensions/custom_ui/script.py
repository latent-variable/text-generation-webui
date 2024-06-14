import os
import subprocess
import platform

def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    # Get the current working directory
    cwd = os.getcwd()
    # Construct the relative path to the Python interpreter
    script_path = os.path.join(cwd, "00_my_scripts", "custom_ui.py")

    subprocess.Popen(['python', script_path])
