import subprocess
import psutil
from library.custom_logging import setup_logging

# Set up logging
log = setup_logging()

class CommandExecutor:
    def __init__(self):
        self.process = None

    def execute_command(self, run_cmd, log_dir=None):
        if self.process and self.process.poll() is None:
            log.info("The command is already running. Please wait for it to finish.")
        else:
            if log_dir is None:
                self.process = subprocess.Popen(run_cmd, shell=True)
            else:
                # HACK: If we have a log file, this is called via API, must block the call
                with open(f"{log_dir}/log.txt", 'a') as f:
                    self.process = subprocess.Popen(run_cmd, shell=True, stdout=f, stderr=subprocess.STDOUT)
                    self.process.wait()

    def kill_command(self):
        if self.process and self.process.poll() is None:
            try:
                parent = psutil.Process(self.process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
                log.info("The running process has been terminated.")
            except psutil.NoSuchProcess:
                log.info("The process does not exist.")
            except Exception as e:
                log.info(f"Error when terminating process: {e}")
        else:
            log.info("There is no running process to kill.")
