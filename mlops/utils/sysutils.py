import platform
import subprocess
import os
from sys import version_info
import re


PYTHON_VERSION = "{major}.{minor}.{micro}".format(
    major=version_info.major, minor=version_info.minor, micro=version_info.micro
)


def is_windows():
    return platform.system() == "Windows"


def run_cmd(args_list):
    print("Running system command: {0}".format(" ".join(args_list)))

    proc = subprocess.Popen(
        args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode

    return s_return, s_output, s_err


def path_splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def parse_rul(url):
    p = "(?:http.*://)?(?P<host>[^:/ ]+).?(?P<port>[0-9]*).*"
    m = re.search(p, url)
    return {"host": m.group("host"), "port": int(m.group("port"))}
