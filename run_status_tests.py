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


tests = []
for td in tests_dirs:
    tests.extend(sorted([td + '/' + f
                        for f in os.listdir(td) if f.endswith('.aag')]))


assert tests, 'not tests found'


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


if __name__ == "__main__":
    for t in tests:
        print 'running ' + tool + ' ' + t
        res = call(tool + ' ' + t, shell=True)

        assert res in [EXIT_STATUS_REALIZABLE, EXIT_STATUS_UNREALIZABLE], 'unknown status: ' + str(res)

        if res == EXIT_STATUS_REALIZABLE and not is_realizable(t):
            print
            print t
            print 'FAILED: should be unrealizable: tool found it realizable. hm.'.format(test=t)
            exit(1)

        if res == EXIT_STATUS_UNREALIZABLE and is_realizable(t):
            print
            print t
            print 'FAILED: should be realizable: tool found it unrealizable. hm.'
            exit(1)

    print 'ALL TESTS PASSED'
