#!/usr/bin/env python2.7
import os
from subprocess import call, Popen

############################################################################
# When used with --mc flag, requires `iimc` model checker (http://iimc.colorado.edu/).
# Change the paths according to your setup.

# Should be executable, to make it executable run: chmod +x ./aisy.py
from tempfile import NamedTemporaryFile

TOOL = "./aisy.py -q"  # '-q' to be quiet (prints only the models)

# Used only if run with --mc
# Usage of iimc: iimc <input_file>
# Returns: 0 -- correct, non zero in other cases
CHECK_MODEL = '/home/ayrat/projects/spec-framework/check_model.sh'
CHECK_MODEL_RC_CORRECT = 0


TESTS_DIR = ["./tests/safety/", "./tests/buechi/"]

############################################################################
from aisy import EXIT_STATUS_REALIZABLE, EXIT_STATUS_UNREALIZABLE
import argparse


def execute_shell(cmd, input=''):
    import shlex
    import subprocess

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

    return p.returncode, out, err
    # for python3:
    # str(out, encoding='utf-8'), \
    # str(err, encoding='utf-8')


def get_tmp_file_name():
    tmp = NamedTemporaryFile(delete=False)
    return tmp.name


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
    rc, out, err = execute_shell(TOOL + ' ' + test)

    assert rc in [EXIT_STATUS_REALIZABLE, EXIT_STATUS_UNREALIZABLE], \
        'unknown status: ' + str(rc) + ', out: \n' + out + '\nerr:\n' + err

    expected = [EXIT_STATUS_UNREALIZABLE, EXIT_STATUS_REALIZABLE][is_realizable(test)]

    if rc != expected:
        print
        print test
        print 'FAILED (status): should be {expected}: the tool found it {res}'.format(expected=status_to_str(expected),
                                                                                      res=status_to_str(rc))
        exit(1)

    if is_realizable(test):
        return out.strip() + '\n' if out.strip() else None   # strip() to remove any funny garbage new lines if any
                                                             # '\n' because aiger tools want this
    
    return None


def to_str_ret_out_err(rc, out, err):
    res = 'ret=' + str(rc) + '\nout=' + str(out) + '\nerr=' + str(err)
    return res


def check_model(synt_model, test_name):
    """ :param synt_model: is not None or empty """

    tmp_file_name = get_tmp_file_name() + '.aag'
    with open(tmp_file_name, 'w') as f:
        f.write(synt_model)
    
    ret, out, err = execute_shell(CHECK_MODEL + ' ' + tmp_file_name)

    if ret != CHECK_MODEL_RC_CORRECT:
        print
        print 'test:', test_name
        print to_str_ret_out_err(ret, out, err)
        print 'Tmp file name', tmp_file_name
        print 'Model checking failed'
        exit(1)

    os.remove(tmp_file_name)  # keep it if we crash


def main(model_check):
    global TOOL
    if not model_check:
        TOOL += ' -r'

    tests = []
    for td in TESTS_DIR:
        tests.extend(sorted([td + '/' + f
                             for f in os.listdir(td) if f.endswith('.aag')]))

    assert tests, 'no tests found'

    for t in tests:
        print 'running ' + TOOL + ' ' + t
        model = check_status(t)
        if model_check and is_realizable(t):
            if not model:
                print
                print 'test:', t, ' failed: is realizale but no model'
                exit(1)

            check_model(model, t)

    print 'ALL TESTS PASSED'
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests runner')

    parser.add_argument('--mc', action='store_true',
                        required=False, default=False,
                        help='model check the result, default: False')

    args = parser.parse_args()

    exit(main(args.mc))
