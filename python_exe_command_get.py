import sys
import datetime
import os
import platform

def get_python_info():
    """
    Get Python execution environment information
    
    Returns:
        tuple: (python path, python version)
    """
    # Get the path of the Python executable
    python_path = sys.executable
    
    # Get Python version information
    python_version = platform.python_version()
    
    return python_path, python_version

def log_command(log_file):
    """
    Save the executed command to log file
    
    Args:
        log_file (str): file name
    """
    # Get the current time
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # python info
    python_path, python_version = get_python_info()
    
    # rebuild the command
    command = f"{python_path} {' '.join(sys.argv)}"
    
    current_dir = os.getcwd()
    
    log_message = (
        "#!/bin/bash\n"
        f"{command}\n"
    )
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message)