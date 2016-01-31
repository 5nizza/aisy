import os
import shlex
import subprocess
from tempfile import NamedTemporaryFile, mkstemp, mkdtemp


def find_files(directory:str, extension:str= '', ignore_mark=None):
    """
    Walk recursively :arg directory, ignoring dirs that contain :arg ignore_mark,
    and return all files matching :arg extension
    """

    if not extension.startswith('.'):
        extension = '.' + extension

    matching_files = list()
    for top, subdirs, files in os.walk(directory, followlinks=True):
        if ignore_mark in files:
            subdirs.clear()
            continue
        for f in files:
            if f.endswith(extension):
                matching_files.append(os.path.join(top, f))
    return matching_files


def execute_shell(cmd, input=''):
    """
    Execute cmd, send input to stdin.
    :return: returncode, stdout, stderr.
    """

    proc_stdin = subprocess.PIPE if input != '' and input is not None else None
    proc_input = input if input != '' and input is not None else None

    args = shlex.split(cmd)

    p = subprocess.Popen(args,
                         stdin=proc_stdin,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    out, err = p.communicate(proc_input)

    return p.returncode, \
           str(out, encoding='utf-8'), \
           str(err, encoding='utf-8')


def get_tmp_file_name():
    (fd, name) = mkstemp()
    os.close(fd)
    return name


def get_tmp_dir_name():
    return mkdtemp()


def cat(test):
    f = open(test)
    res = f.readlines()
    f.close()
    return res
