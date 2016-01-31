#!/usr/bin/env python3
import argparse
import os

from func_tests import run_tests
from utils import execute_shell, cat, find_files
from config import SPEC_FRAMEWORK_DIR
from ansistrm import setup_logging


def is_realizable(test):
    spec_status = cat(test)[-1].strip()
    if spec_status == 'realizable':
        return True
    if spec_status == 'unrealizable':
        return False
    assert 0, 'spec status is unknown'


def run_tool(test_file, result_file):
    return execute_shell('./aisy.py  ' + test_file + ' --out ' + result_file)


def check_answer(test_file:str, result_file, rc, out, err):
    exit_status_realizable = 10
    exit_status_unrealizable = 20
    assert rc in [exit_status_realizable, exit_status_unrealizable], rc

    expected = [exit_status_unrealizable, exit_status_realizable][is_realizable(test_file)]

    if rc != expected:
        status_to_str = {exit_status_realizable: 'realizable',
                         exit_status_unrealizable: 'unrealizable'}
        out = 'wrong realizability status: should be {expected}, but the tool found it {res}'.format(
            expected=status_to_str[expected],
            res=status_to_str[rc])
        return 1, out, None

    return 0, None, None


def check_answer_with_mc(test_file, result_file, rc, out, err):
    if check_answer(test_file, result_file, rc, out, err)[0] != 0:
        return check_answer(test_file, result_file, rc, out, err)

    if not is_realizable(test_file):
        return 0, None, None

    tmp_file_name = result_file + '.aag'   # because iimc doesn't understand non-aag extensions
    assert 0 == execute_shell('cat {result_file} >> {result_file_aag}'.format(result_file=result_file,
                                                                              result_file_aag=tmp_file_name))\
                [0]

    rc, out, err = execute_shell(SPEC_FRAMEWORK_DIR + '/check_model.sh ' + tmp_file_name)

    os.remove(tmp_file_name)

    return rc, out, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Functional tests runner')

    parser.add_argument('--mc', action='store_true',
                        required=False, default=False,
                        help='model check the result, default: False')

    args = parser.parse_args()

    TEST_FILES = find_files('./tests/safety/', extension='aag', ignore_mark='notest') + \
                 find_files('./tests/buechi/', extension='aag', ignore_mark='notest') + \
                 find_files('./tests/syntcomp-format/', extension='aag', ignore_mark='notest') + \
                 find_files('./tests/1-streett/', extension='aag', ignore_mark='notest')
    RUN_TOOL = run_tool
    CHECK_RESULT = [check_answer, check_answer_with_mc][args.mc]

    logger = setup_logging()
    exit(run_tests(TEST_FILES,
                   RUN_TOOL,
                   CHECK_RESULT,
                   True,
                   logger))
