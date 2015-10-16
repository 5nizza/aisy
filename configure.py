#!/usr/bin/env python3

import argparse
import os
import stat

CONFIG_PY_NAME = 'config.py'
CONFIG_SH_NAME = 'setup.sh'

CONFIG_PY_TEXT = """
SPEC_FRAMEWORK_DIR="/home/ayrat/projects/spec-framework/"
"""

CONFIG_SH_TEXT = """
#!/bin/bash
# to run
# . ./setup.sh
# yes, with .

export LD_LIBRARY_PATH=/home/ayrat/projects/pycudd2.0.2/cudd-2.4.2/lib
"""


def _get_root_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _user_confirmed(question):
    answer = input(question + ' [y/n] ').strip()
    assert answer in 'yYnN', answer
    return answer in 'yY'


def _check_files_exist(files):
    existing = list(filter(lambda f: os.path.exists(f), files))
    return existing


def main():
    config_py = os.path.join(_get_root_dir(), CONFIG_PY_NAME)
    config_sh = os.path.join(_get_root_dir(), CONFIG_SH_NAME)
    existing = _check_files_exist([CONFIG_PY_NAME, CONFIG_SH_NAME])
    if not existing or \
            _user_confirmed('{files} already exist(s).\n'.format(files=existing) +
                            'Replace?'):
        with open(config_py, 'w') as file:
            file.write(CONFIG_PY_TEXT)
        with open(config_sh, 'w') as file:
            file.write(CONFIG_SH_TEXT)
        # make 'sh' config executable
        os.chmod(config_sh,
                 os.stat(config_sh).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        print('Created {files}.\n'
              'Now edit them with your paths.'.
              format(files=[CONFIG_PY_NAME, CONFIG_SH_NAME]))

        return True

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     'Generate local configuration file')

    args = parser.parse_args()
    res = main()
    print(['not done', 'done'][res])
    exit(res)
