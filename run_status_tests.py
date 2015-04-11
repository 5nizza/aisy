#!/usr/bin/env python2.7
import os
from subprocess import call

############################################################################
# change these two paths according to your setup

# should be executable, to make it executable run: chmod +x ./aisy.py
tool = "./aisy.py -r"

tests_dirs = ["./tests/safety/", "./tests/buechi/"]

############################################################################
from aisy import EXIT_STATUS_REALIZABLE, EXIT_STATUS_UNREALIZABLE
import argparse


def cat(test):
    f = open(test)
    res = f.readlines()
    f.close()
    return res


def is_realizable(test):
    spec_status = cat(test)[-1].strip()
    if spec_status == 'realizable':
        return True
    if spec_status == 'unrealizable':
        return False
    assert 0, 'spec status is unknown'


def status_to_str(exit_status):
    return ['unrealizable', 'realizable'][exit_status == EXIT_STATUS_REALIZABLE]


def check_status(test):
    res = call(tool + ' ' + test, shell=True)

    assert res in [EXIT_STATUS_REALIZABLE, EXIT_STATUS_UNREALIZABLE], 'unknown status: ' + str(res)

    expected = [EXIT_STATUS_UNREALIZABLE, EXIT_STATUS_REALIZABLE][is_realizable(test)]

    if res != expected:
        print
        print test
        print 'FAILED (status): should be {expected}: the tool found it {res}'.format(expected=status_to_str(expected),
                                                                                      res=status_to_str(res))
        exit(1)


def main():
    tests = []
    for td in tests_dirs:
        tests.extend(sorted([td + '/' + f
                             for f in os.listdir(td) if f.endswith('.aag')]))

    assert tests, 'not tests found'

    for t in tests:
        print 'running ' + tool + ' ' + t
        check_status(t)

    print 'ALL TESTS PASSED'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests runner')

    parser.add_argument('--mc', action='store_true',
                        required=False, default=False,
                        help='model check the result, default: False')

    args = parser.parse_args()
    exit(main())

