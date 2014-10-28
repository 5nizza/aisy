#!/usr/bin/env python2.7
# coding=utf-8

"""
An example of synthesis tool from Aiger http://fmv.jku.at/aiger/ circuits format.
Implementations of some functions are omitted to give you chance to implement them.

Basic stuff is left: parsing, and some helper functions.

Installation requirements:
  - pycudd library: http://bears.ece.ucsb.edu/pycudd.html
  - swig library: http://www.swig.org/
  - (probably) python2.7 headers

After installing pycudd library add cudd libraries into your LD_LIBRARY_PATH:

export LD_LIBRARY_PATH=/path/to/pycudd2.0.2/cudd-2.4.2/lib

To run:

./aisy.py -h

Some self-testing functionality is included in ``run_status_tests.py``.

More extensive tests are provided by Robert in script ``performance_test.sh``.
This script also runs model checker to check the results.

More details you will find in the code, on web-page of the course.

Email me in case questions/suggestions/bugs: ayrat.khalimov at gmail

----------------------
"""

status = ["TODO: check the case when output functions depend on error fake latch"]


import argparse
import logging
import pycudd
import sys
from aiger_swig.aiger_wrap import *
from aiger_swig.aiger_wrap import aiger

#don't change status numbers since they are used by the performance script
EXIT_STATUS_REALIZABLE = 10
EXIT_STATUS_UNREALIZABLE = 20


#: :type: aiger
spec = None

#: :type: DdManager
cudd = None

# error output can be latched or unlatched, latching makes things look like in lectures, lets emulate this
#: :type: aiger_symbol
error_fake_latch = None


#: :type: Logger
logger = None


def setup_logging(verbose):
    global logger
    level = None
    if verbose is 0:
        level = logging.INFO
    elif verbose >= 1:
        level = logging.DEBUG

    logging.basicConfig(format="%(asctime)-10s%(message)s",
                        datefmt="%H:%M:%S",
                        level=level,
                        stream=sys.stdout)

    logger = logging.getLogger(__name__)


def is_negated(l):
    return (l & 1) == 1


def strip_lit(l):
    return l & ~1


def introduce_error_latch_if():
    global error_fake_latch
    if error_fake_latch:
        return

    error_fake_latch = aiger_symbol()
    #: :type: aiger_symbol
    error_symbol = get_err_symbol()

    error_fake_latch.lit = (int(spec.maxvar) + 1) * 2
    error_fake_latch.name = 'fake_error_latch'
    error_fake_latch.next = error_symbol.lit


def iterate_latches_and_error():
    introduce_error_latch_if()

    for i in range(int(spec.num_latches)):
        yield get_aiger_symbol(spec.latches, i)

    yield error_fake_latch


def parse_into_spec(aiger_file_name):
    global spec
    logger.info('parsing..')
    #: :type: aiger
    spec = aiger_init()

    err = aiger_open_and_read_from_file(spec, aiger_file_name)
    assert not err, err


def get_lit_type(stripped_lit):
    if stripped_lit == error_fake_latch.lit:
        return None, error_fake_latch, None

    input_ = aiger_is_input(spec, stripped_lit)
    latch_ = aiger_is_latch(spec, stripped_lit)
    and_ = aiger_is_and(spec, strip_lit(stripped_lit))

    return input_, latch_, and_


def get_bdd_for_value(lit):  # lit is variable index with sign
    stripped_lit = strip_lit(lit)

    # we faked error latch and so we cannot call directly aiger_is_input, aiger_is_latch, aiger_is_and
    input_, latch_, and_ = get_lit_type(stripped_lit)

    if stripped_lit == 0:
        res = cudd.Zero()

    elif input_ or latch_:
        res = cudd.IthVar(stripped_lit)    # use internal mapping of cudd

    elif and_:
        #: :type: aiger_and
        arg1 = get_bdd_for_value(int(and_.rhs0))
        arg2 = get_bdd_for_value(int(and_.rhs1))
        res = arg1 & arg2

    else:
        assert 0, 'should be impossible: if it is output then it is still either latch or and'

    if is_negated(lit):
        res = ~res

    return res


def get_unprimed_variable_as_bdd(lit):
    stripped_lit = strip_lit(lit)
    return cudd.IthVar(stripped_lit)


def get_primed_variable_as_bdd(lit):
    stripped_lit = strip_lit(lit)
    return cudd.IthVar(stripped_lit + 1)  # we know that odd vars cannot be used as names of latches/inputs


def make_bdd_eq(value1, value2):
    return (value1 & value2) | (~value1 & ~value2)


def compose_transition_bdd():
    """ :return: BDD representing transition function of spec: ``T(x,i,c,x')``
    """

    logger.info('compose_transition_bdd, nof_latches={0}...'.format(len(list(iterate_latches_and_error()))))

    #: :type: DdNode
    transition = cudd.One()
    for l in iterate_latches_and_error():
        #: :type: aiger_symbol
        l = l

        next_value_bdd = get_bdd_for_value(int(l.next))

        next_value_variable = get_primed_variable_as_bdd(l.lit)

        latch_transition = make_bdd_eq(next_value_variable, next_value_bdd)

        transition &= latch_transition

    # print('final transition BDD is')
    # transition.PrintMinterm()

    return transition


def get_err_symbol():
    assert spec.num_outputs == 1
    return spec.outputs  # swig return one element instead of array, to iterate over array use iterate_symbols


def get_cube(variables):
    assert len(variables)

    cube = cudd.One()
    for v in variables:
        cube &= v
    return cube


def _get_bdd_vars(filter_func):
    var_bdds = []

    for i in range(int(spec.num_inputs)):
        input_aiger_symbol = get_aiger_symbol(spec.inputs, i)
        if filter_func(input_aiger_symbol.name.strip()):
            out_var_bdd = get_bdd_for_value(input_aiger_symbol.lit)
            var_bdds.append(out_var_bdd)

    return var_bdds


def get_controllable_vars_bdds():
    return _get_bdd_vars(lambda name: name.startswith('controllable'))


def get_uncontrollable_output_bdds():
    return _get_bdd_vars(lambda name: not name.startswith('controllable'))


def get_all_latches_as_bdds():
    bdds = [get_bdd_for_value(l.lit) for l in iterate_latches_and_error()]
    return bdds


def _prime_unprime_latches_in_bdd(bdd, should_prime):
    latch_bdds = get_all_latches_as_bdds()
    num_latches = len(latch_bdds)
    #: :type: DdArray
    primed_var_array = pycudd.DdArray(num_latches)
    curr_var_array = pycudd.DdArray(num_latches)

    for l_bdd in latch_bdds:
        #: :type: DdNode
        l_bdd = l_bdd
        curr_var_array.Push(l_bdd)

        lit = l_bdd.NodeReadIndex()
        new_l_bdd = get_primed_variable_as_bdd(lit)
        primed_var_array.Push(new_l_bdd)

    if should_prime:
        replaced_states_bdd = bdd.SwapVariables(curr_var_array, primed_var_array, num_latches)
    else:
        replaced_states_bdd = bdd.SwapVariables(primed_var_array, curr_var_array, num_latches)

    return replaced_states_bdd


def prime_latches_in_bdd(states_bdd):
    primed_states_bdd = _prime_unprime_latches_in_bdd(states_bdd, True)
    return primed_states_bdd


def unprime_latches_in_bdd(bdd):
    unprimed_bdd = _prime_unprime_latches_in_bdd(bdd, False)
    return unprimed_bdd


def pre_sys_bdd(dst_states_bdd, transition_bdd):
    """ Calculate predecessor states of given states.

    :return: BDD representation of predecessor states

    :hints: if current states are not primed they should be primed before calculation (why?)
    :hints: calculation of ``∃o t(a,b,o)`` using cudd: ``t.ExistAbstract(get_cube(o))``
    :hints: calculation of ``∀i t(a,b,i)`` using cudd: ``t.UnivAbstract(get_cube(i))``
    """

    #: :type: DdNode
    transition_bdd = transition_bdd
    #: :type: DdNode
    primed_dst_states_bdd = prime_latches_in_bdd(dst_states_bdd)

    #: :type: DdNode
    intersection = transition_bdd & primed_dst_states_bdd  # all predecessors (i.e., if sys and env cooperate)

    # cudd requires to create a cube first..
    if len(get_controllable_vars_bdds()) != 0:
        out_vars_cube_bdd = get_cube(get_controllable_vars_bdds())
        exist_outs = intersection.ExistAbstract(out_vars_cube_bdd)  # ∃o tau(t,i,t',o)
    else:
        exist_outs = intersection

    next_state_vars_cube = prime_latches_in_bdd(get_cube(get_all_latches_as_bdds()))
    exist_next_state = exist_outs.ExistAbstract(next_state_vars_cube)  # ∃o ∃t'  tau(t,i,t',o)

    uncontrollable_output_bdds = get_uncontrollable_output_bdds()
    if uncontrollable_output_bdds:
        in_vars_cube_bdd = get_cube(uncontrollable_output_bdds)
        forall_inputs = exist_next_state.UnivAbstract(in_vars_cube_bdd)  # ∀i ∃o ∃t'  tau(t,i,t',o)
    else:
        forall_inputs = exist_next_state

    return forall_inputs


def calc_win_region(init_state_bdd, transition_bdd, not_error_bdd):
    """ Calculate a winning region for the safety game: win = greatest_fix_point.X [not_error & pre_sys(X)]
    :return: BDD representing the winning region
    """

    logger.info('calc_win_region..')

    new_set_bdd = cudd.One()
    while True:
        curr_set_bdd = new_set_bdd

        new_set_bdd = not_error_bdd & pre_sys_bdd(curr_set_bdd, transition_bdd)

        if (new_set_bdd & init_state_bdd) == cudd.Zero():
            return cudd.Zero()

        if new_set_bdd == curr_set_bdd:
            return new_set_bdd


def get_nondet_strategy(win_region_bdd, transition_bdd):
    """ Get non-deterministic strategy from the winning region.
    If the system outputs controllable values that satisfy this non-deterministic strategy, then the system wins.
    I.e., a non-deterministic strategy describes for each state all possible plausible output values:

    ``strategy(x,i,c) = ∃x' W(x) & T(x,i,c,x') & W(x') ``

    (Why system cannot lose if adhere to this strategy?)

    :return: non deterministic strategy bdd
    :note: The strategy is still not-deterministic. Determinization step is done later.
    """

    logger.info('get_nondet_strategy..')

    #: :type: DdNode
    primed_win_region_bdd = prime_latches_in_bdd(win_region_bdd)
    # TODO: use more efficient exist_and
    intersection = (win_region_bdd & primed_win_region_bdd & transition_bdd)

    next_vars_cube = prime_latches_in_bdd(get_cube(get_all_latches_as_bdds()))
    strategy = intersection.ExistAbstract(next_vars_cube)

    return strategy


def compose_init_state_bdd():
    """ Initial state is 'all latches are zero' """

    logger.info('compose_init_state_bdd..')

    init_state_bdd = cudd.One()
    for l in iterate_latches_and_error():
        #: :type: aiger_symbol
        l = l
        l_curr_value_bdd = get_bdd_for_value(l.lit)
        init_state_bdd &= make_bdd_eq(l_curr_value_bdd, cudd.Zero())

    return init_state_bdd


def extract_output_funcs(non_det_strategy, init_state_bdd, transition_bdd):
    """
    Calculate BDDs for output functions given a non-deterministic winning strategy.
    Cofactor-based approach.

    :return: dictionary ``controllable_variable_bdd -> func_bdd``
    """

    logger.info('extract_output_funcs..')

    output_models = dict()
    controls = get_controllable_vars_bdds()
    for c in get_controllable_vars_bdds():
        logger.info('getting output function for ' + aiger_is_input(spec, strip_lit(c.NodeReadIndex())).name)

        others = set(set(controls).difference({c}))
        if others:
            others_cube = get_cube(others)
            #: :type: DdNode
            c_arena = non_det_strategy.ExistAbstract(others_cube)
        else:
            c_arena = non_det_strategy

        # c_arena.PrintMinterm()

        can_be_true = c_arena.Cofactor(c)  # states (x,i) in which c can be true
        can_be_false = c_arena.Cofactor(~c)

        # We need to intersect with can_be_true to narrow the search.
        # Negation can cause including states from !W (with err=1)
        #: :type: DdNode
        must_be_true = (~can_be_false) & can_be_true
        must_be_false = (~can_be_true) & can_be_false

        care_set = (must_be_true | must_be_false)

        # We use 'restrict' operation, but we could also do just:
        # c_model = must_be_true -> care_set
        # ..but this is (probably) less efficient, since we cannot set c=1 if it is not in care_set, but we could.
        #
        # Restrict on the other side applies optimizations to find smaller bdd.
        # It cannot be expressed using boolean logic operations since we would need to say:
        # must_be_true = ite(care_set, must_be_true, "don't care")
        # and "don't care" cannot be expressed in boolean logic.

        # Restrict operation:
        #   on care_set: must_be_true.restrict(care_set) <-> must_be_true
        c_model = must_be_true.Restrict(care_set)

        output_models[c] = c_model

        non_det_strategy = non_det_strategy & make_bdd_eq(c, c_model)

    return output_models


def synthesize(realiz_check):
    """ Calculate winning region and extract output functions.

    :return: - if realizable: <True, dictionary: controllable_variable_bdd -> func_bdd>
             - if not: <False, None>
    """
    logger.info('synthesize..')

    #: :type: DdNode
    init_state_bdd = compose_init_state_bdd()
    #: :type: DdNode
    transition_bdd = compose_transition_bdd()
    # transition_bdd.PrintMinterm()

    #: :type: DdNode
    not_error_bdd = ~get_bdd_for_value(error_fake_latch.lit)
    win_region = calc_win_region(init_state_bdd, transition_bdd, not_error_bdd)

    if win_region == cudd.Zero():
        return False, None

    if realiz_check:
        return True, None

    non_det_strategy = get_nondet_strategy(win_region, transition_bdd)

    func_by_var = extract_output_funcs(non_det_strategy, init_state_bdd, transition_bdd)

    return True, func_by_var


def negated(lit):
    return lit ^ 1


def next_lit():
    """ :return: next possible to add to the spec literal """
    return (int(spec.maxvar) + 1) * 2


def get_optimized_and_lit(a_lit, b_lit):
    if a_lit == 0 or b_lit == 0:
        return 0

    if a_lit == 1 and b_lit == 1:
        return 1

    if a_lit == 1:
        return b_lit

    if b_lit == 1:
        return a_lit

    if a_lit > 1 and b_lit > 1:
        a_b_lit = next_lit()
        aiger_add_and(spec, a_b_lit, a_lit, b_lit)
        return a_b_lit

    assert 0, 'impossible'


def walk(a_bdd):
    """
    Walk given BDD node (recursively).
    If given input BDD requires intermediate AND gates for its representation, the function adds them.
    Literal representing given input BDD is `not` added to the spec.

    :returns: literal representing input BDD
    :warning: variables in cudd nodes may be complemented, check with: ``node.IsComplement()``
    """

    #: :type: DdNode
    a_bdd = a_bdd
    if a_bdd.IsConstant():
        res = int(a_bdd == cudd.One())   # in aiger 0/1 = False/True
        return res

    # get an index of variable,
    # all variables used in bdds also introduced in aiger,
    # except fake error latch literal,
    # but fake error latch will not be used in output functions (at least we don't need this..)
    a_lit = a_bdd.NodeReadIndex()

    assert a_lit != error_fake_latch.lit, 'using error latch in the definition of output function is not allowed'

    #: :type: DdNode
    t_bdd = a_bdd.T()
    #: :type: DdNode
    e_bdd = a_bdd.E()

    t_lit = walk(t_bdd)
    e_lit = walk(e_bdd)

    # ite(a_bdd, then_bdd, else_bdd)
    # = a*then + !a*else
    # = !(!(a*then) * !(!a*else))
    # -> in general case we need 3 more ANDs

    a_t_lit = get_optimized_and_lit(a_lit, t_lit)

    na_e_lit = get_optimized_and_lit(negated(a_lit), e_lit)

    n_a_t_lit = negated(a_t_lit)
    n_na_e_lit = negated(na_e_lit)

    ite_lit = get_optimized_and_lit(n_a_t_lit, n_na_e_lit)

    res = negated(ite_lit)
    if a_bdd.IsComplement():
        res = negated(res)

    return res


def model_to_aiger(c_bdd, func_bdd, introduce_output):
    """ Update aiger spec with a definition of ``c_bdd``
    """
    #: :type: DdNode
    c_bdd = c_bdd
    c_lit = c_bdd.NodeReadIndex()

    func_as_aiger_lit = walk(func_bdd)

    aiger_redefine_input_as_and(spec, c_lit, func_as_aiger_lit, func_as_aiger_lit)

    if introduce_output:
        aiger_add_output(spec, c_lit, '')


def init_cudd():
    global cudd
    cudd = pycudd.DdManager()
    cudd.SetDefault()
    #CUDD_REORDER_SAME,
    #CUDD_REORDER_NONE,
    #CUDD_REORDER_RANDOM,
    #CUDD_REORDER_RANDOM_PIVOT,
    #CUDD_REORDER_SIFT,
    #CUDD_REORDER_SIFT_CONVERGE,
    #CUDD_REORDER_SYMM_SIFT,
    #CUDD_REORDER_SYMM_SIFT_CONV,
    #CUDD_REORDER_WINDOW2,
    #CUDD_REORDER_WINDOW3,
    #CUDD_REORDER_WINDOW4,
    #CUDD_REORDER_WINDOW2_CONV,
    #CUDD_REORDER_WINDOW3_CONV,
    #CUDD_REORDER_WINDOW4_CONV,
    #CUDD_REORDER_GROUP_SIFT,
    #CUDD_REORDER_GROUP_SIFT_CONV,
    #CUDD_REORDER_ANNEALING,
    #CUDD_REORDER_GENETIC,
    #CUDD_REORDER_LINEAR,
    #CUDD_REORDER_LINEAR_CONVERGE,
    #CUDD_REORDER_LAZY_SIFT,
    #CUDD_REORDER_EXACT
    # cudd.AutodynEnable(4)
    # cudd.AutodynDisable()
    # cudd.EnableReorderingReporting()


def main(aiger_file_name, out_file_name, output_full_circuit, realiz_check):
    """ Open aiger file, synthesize the circuit and write the result to output file.

    :returns: boolean value 'is realizable?'
    """
    init_cudd()

    parse_into_spec(aiger_file_name)

    realizable, func_by_var = synthesize(realiz_check)

    if realiz_check:
        return realizable

    if realizable:
        for (c_bdd, func_bdd) in func_by_var.items():
            model_to_aiger(c_bdd, func_bdd, output_full_circuit)
        
        aiger_reencode(spec)  # some model checkers do not like unordered variable names (when e.g. latch is > add)

        if out_file_name:
            aiger_open_and_write_to_file(spec, out_file_name)
        else:
            res, string = aiger_write_to_string(spec, aiger_ascii_mode, 268435456)
            assert res != 0 or out_file_name is None, 'writing failure'
            logger.info('\n' + string)
        return True

    return False


def exit_if_status_request(args):
    if args.status:
        print('-'*80)
        print('The current status of development')
        print('-- ' + '\n-- '.join(status))
        print('-'*80)
        exit(0)
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aiger Format Based Simple Synthesizer')
    parser.add_argument('aiger', metavar='aiger', type=str, nargs='?', default=None, help='input specification in AIGER format')
    parser.add_argument('--out', '-o', metavar='out', type=str, required=False, default=None,
                        help='output file in AIGER format (if realizable)')
    parser.add_argument('--full', action='store_true', default=False,
                        help='produce a full circuit that has outputs other than error bit')
    parser.add_argument('--status', '-s', action='store_true', default=False,
                        help='Print current status of development')

    parser.add_argument('--realizability', '-r', action='store_true', default=False,
                        help='Check Realizability only (do not produce circuits)')

    args = parser.parse_args()
    
    exit_if_status_request(args)
    
    if not args.aiger:
        print('aiger file is required, exit')
        exit(-1)

    setup_logging(0)
    
    is_realizable = main(args.aiger, args.out, args.full, args.realizability)

    logger.info(['unrealizable', 'realizable'][is_realizable])

    exit([EXIT_STATUS_UNREALIZABLE, EXIT_STATUS_REALIZABLE][is_realizable])
