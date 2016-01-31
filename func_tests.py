from logging import Logger
from os import rmdir, makedirs

from utils import get_tmp_file_name, get_tmp_dir_name, execute_shell


def rc_out_err_to_str(rc, out, err):
    return 'rc:{rc}\n' \
           'out:{out}\n' \
           'err:{err}'.format(rc=rc, out=out or '<empty>', err=err or '<empty>')


def run_tests(test_files,
              run_tool:"func(file, result_file)->(rc, out, err)",
              check_answer:"func(test_file, result_file, rc, out, err)->(rc, out, err)",
              stop_on_error,
              logger:Logger,
              output_folder=None):
    """
    :param output_folder: if not None, intermediate results are saved there.
                          Files in that folder will be overwritten.
    """

    if output_folder:
        output_dir = output_folder
        makedirs(output_dir, exist_ok=True)
    else:
        output_dir = get_tmp_dir_name()

    failed_test = list()
    for test in test_files:
        logger.info('testing {test}..'.format(test=test))
        result_file = '{output_dir}/{test_last}.model'.format(output_dir=output_dir,
                                                              test_last=test.split('/')[-1])
        r_rc, r_out, r_err = run_tool(test, result_file)
        logger.debug(rc_out_err_to_str(r_rc, r_out, r_err))

        c_rc, c_out, c_err = check_answer(test, result_file, r_rc, r_out, r_err)
        logger.debug(rc_out_err_to_str(c_rc, c_out, c_err))

        if c_rc != 0:
            logger.info('    FAILED')
            failed_test.append(test)
            if stop_on_error:
                break

    if failed_test:
        logger.info('The following tests failed:', ''.join('\n    ' + t for t in failed_test))
    else:
        logger.info('ALL TESTS PASSED')

    if not output_folder:
        assert 0 == execute_shell('rm -rf ' + output_dir)[0]  # TODO: how to remove dir and its content in python?

    return not failed_test
