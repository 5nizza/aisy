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

Some self-testing functionality is included in `run_tests.py`.
It also runs model checker to check the results.

Email me in case questions/suggestions/bugs: ayrat.khalimov at gmail
"""

import argparse
import logging
import sys

import pycudd

from aiger_swig.aiger_wrap import *
from ansistrm import ColorizingStreamHandler
import aiger_swig.aiger_wrap as aiglib


# don't change status numbers since they are used by the performance script
EXIT_STATUS_REALIZABLE = 10
EXIT_STATUS_UNREALIZABLE = 20


#: :type: aiger
spec = None

#: :type: DdManager
cudd = None

#: :type: Logger
logger = None


def setup_logging(verbose_level, filename=None):
    level = None
    if verbose_level == -1:
        level = logging.CRITICAL
    if verbose_level is 0:
        level = logging.INFO
    elif verbose_level >= 1:
        level = logging.DEBUG

    formatter = logging.Formatter(fmt="%(asctime)-10s%(message)s",
                                  datefmt="%H:%M:%S")

    stdout_handler = ColorizingStreamHandler()
    stdout_handler.setFormatter(formatter)
    stdout_handler.stream = sys.stdout

    if not filename:
        filename = 'last.log'
    file_handler = logging.FileHandler(filename=filename, mode='w')
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)

    root.setLevel(level)

    return logging.getLogger(__name__)


def is_negated(l):
    return (l & 1) == 1


def strip_lit(l):
    return l & ~1


def iterate_latches_and_error():
    for i in range(int(spec.num_latches)):
        yield get_aiger_symbol(spec.latches, i)


def parse_into_spec(aiger_file_name):
    logger.info('parsing..')

    global spec
    #: :type: aiger
    spec = aiger_init()

    err = aiger_open_and_read_from_file(spec, aiger_file_name)
    assert not err, err

    # assert the formats
    assert (spec.num_outputs == 1) ^ (spec.num_bad >= 1 or spec.num_justice == 1), 'mix of two formats'
    assert spec.num_outputs + spec.num_justice + spec.num_bad >= 1, 'no properties'

    # assert aisy's pre
    assert spec.num_justice <= 1, 'not supported'
    assert spec.num_fairness == 0, 'not supported'


def get_lit_type(stripped_lit):
    input_ = aiger_is_input(spec, stripped_lit)
    latch_ = aiger_is_latch(spec, stripped_lit)
    and_ = aiger_is_and(spec, stripped_lit)

    return input_, latch_, and_


def get_bdd_for_value(lit):  # lit is variable index with sign
    """
    We use the following mapping of AIGER indices to CUDD indices:
    AIGER's stripped_lit -> CUDD's index
    For latches: primed value of a variable with CUDD's index is index+1
    Note that none of the AIGER inputs have primed version equivalents in CUDD.
      Thus we lose some number of indices in CUDD.
    Note that only AIGER's latches and inputs have equivalents in CUDD.
    """
    stripped_lit = strip_lit(lit)

    if stripped_lit == 0:
        res = cudd.Zero()
    else:
        input_, latch_, and_ = get_lit_type(stripped_lit)

        if input_ or latch_:
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

    return transition


def get_cube(variables):
    if not variables:
        return cudd.One()

    cube = cudd.One()
    for v in variables:
        cube &= v
    return cube


def _get_bdd_vars(filter_func):
    var_bdds = []

    for i in range(spec.num_inputs):
        input_aiger_symbol = get_aiger_symbol(spec.inputs, i)
        if filter_func(input_aiger_symbol.name.strip()):
            out_var_bdd = get_bdd_for_value(input_aiger_symbol.lit)
            var_bdds.append(out_var_bdd)

    return var_bdds


def get_controllable_vars_bdds():
    return _get_bdd_vars(lambda name: name.startswith('controllable'))


def get_uncontrollable_vars_bdds():
    return _get_bdd_vars(lambda name: not name.startswith('controllable'))


def get_all_latches_as_bdds():
    bdds = [get_bdd_for_value(l.lit) for l in iterate_latches_and_error()]
    return bdds


def _prime_unprime_latches_in_bdd(bdd, should_prime):
    if bdd == cudd.Zero() or bdd == cudd.One():
        return bdd

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


def modified_pre_sys_bdd(dst_states_bdd, transition_bdd, inv_bdd, err_bdd):
    """
    Calculate predecessor states of Dst(t') accounting for invariant and error
    transitions:

         ∀i ∃o:
           inv(t,i,o)  ->  [ ~err(t,i,o) & ∃t' tau(t,i,t',o) & Dst(t') ]

    :return: BDD representation of the predecessor states
    """

    #: :type: DdNode
    primed_dst_states_bdd = prime_latches_in_bdd(dst_states_bdd)

    #: :type: DdNode
    tau_and_dst = transition_bdd & primed_dst_states_bdd  # all predecessors (i.e., if sys and env cooperate)

    # cudd requires to create a cube first
    next_state_vars_cube = prime_latches_in_bdd(get_cube(get_all_latches_as_bdds()))
    exist_tn__tau_and_dst = tau_and_dst.ExistAbstract(next_state_vars_cube)  # ∃t'  tau(t,i,t',o) & dst(t')

    assert len(get_controllable_vars_bdds()) > 0  # TODOfut: without outputs make it model checker

    nerr_and_tau = ~err_bdd & exist_tn__tau_and_dst
    inv_impl_nerr_tau = ~inv_bdd | nerr_and_tau

    out_vars_cube = get_cube(get_controllable_vars_bdds())
    exist_outs = inv_impl_nerr_tau.ExistAbstract(out_vars_cube)  # ∃o: inv-> ~err &  ∃t' tau(t,i,t',o)

    inp_vars_bdds = get_uncontrollable_vars_bdds()
    if inp_vars_bdds:
        inp_vars_cube = get_cube(inp_vars_bdds)
        forall_inputs = exist_outs.UnivAbstract(inp_vars_cube)  # ∀i ∃o: inv -> ..
    else:
        forall_inputs = exist_outs

    return forall_inputs


def calc_win_region(init_bdd, transition_bdd, inv_bdd, err_bdd, f_bdd):
    """
    Calculate a winning region for the Buchi game.
    The win region for the Buchi game is:

        gfp.Y lfp.X [F & pre_sys(Y)  |  pre_sys(X)]

    Note that for Buchi game with invariants we use modified_pre_sys operator.
    To see docs for function `modified_pre_sys_bdd` and for more details visit
    https://verify.iaik.tugraz.at/research/bin/view/Ausgewaehltekapitel/BuchiWithInvariantsAndSafety

    :return: list of BDDs [Recur(F), Force1(Recur(F)), Force2(Recur(F)), ...],
             thus, [..][-1] is the winning region
             or None if unrealizable
    """

    logger.info('calc_win_region..')
    # TODOopt1: try that weird: compute attractors for rings,
    # then unite with the previous attractor (attr_i+1 = Force(attr_i\attr_i-1) \cup attr_i)

    # - how about computing certainly losing positions?
    # Does not help.
    # Attr_1(inv & err)
    # safety_losing_positions = calc_attr_err(transition_bdd, inv_bdd, err_bdd)
    # if init_bdd <= safety_losing_positions:
    #     print 'init bdd is in safety losing set. no chance -- unrealizable.'
    #     return None

    # The notations are taken from "Infinite Games" by Martin Zimmerman/Felix Klein.
    # They use:
    # (0 is the system player, 1 is the environment player, 
    #  and thus, W1 - env win region, etc.)
    # Rec_0 = F
    # W1_n = V \ Attr0(Rec_n(F))
    # Rec_n+1 = F \ CPre1(W1_n)
    #
    # If you transform, then you get
    # Rec_n+1 = F & CPre0(Attr0(Rec_n))
    # and we use it below.
    #

    Rec = f_bdd   # the current recurrence set
    while True:   # the outer loop that computes recurrence sets
        attractors = list()   # we save the attractors in order to compute the strategy later
        AttrRec = Rec
        while True:  # computing the attractor of a current recurrence set
            attractors.append(AttrRec)
            nAttrRec = AttrRec | modified_pre_sys_bdd(AttrRec, transition_bdd, inv_bdd, err_bdd)
            if nAttrRec == AttrRec:
                break
            AttrRec = nAttrRec

        logger.info('calc_win_region: # attractors: ' + str(len(attractors)))

        if not (init_bdd <= AttrRec):  # I 'believe' this speeds up by 10% on unrealizable examples
            return None                # is there the proper way to check init \in AttrRec?

        nRec = Rec & modified_pre_sys_bdd(AttrRec, transition_bdd, inv_bdd, err_bdd)
        if nRec == Rec:
            return attractors
        Rec = nRec


def get_nondet_strategy(attractors, transition_bdd, inv_bdd, err_bdd):
    """ Calculate the non-deterministic strategy for the given winning region (or the attractors for Buchi games).
    The non-deterministic strategy is the set of triples `(t,i,o)` s.t.
    if, for given `t,i` the system outputs any value `o` for which `(t,i,o)` is in the set,
    then the system wins.
    Thus, a non-deterministic strategy describes, for each state and input, all winning output values.

    :arg:attractors should be in increasing order
    :return: non deterministic strategy bdd
    :note: The strategy is not-deterministic -- determinization step is done later.
    """

    logger.info('get_nondet_strategy..')

    #   inv -> AND_(Src,Dst): onion_i -> ~err & ∃t' tau(t,i,t',o) & Dst(t'))
    # where onion = Src & ~Dst

    assert_increasing(attractors)
    attractors = list(attractors)
    attractors.reverse()

    src_dst_conjuncts = cudd.One()
    for i in range(len(attractors)):
        src, dst = attractors[i], attractors[(i+1) % len(attractors)]

        onion = src & ~dst if i != len(attractors)-1 \
                else src

        dstP = prime_latches_in_bdd(dst)
        tP = prime_latches_in_bdd(get_cube(get_all_latches_as_bdds()))

        exists_tP__tau_and_dstP = (transition_bdd & dstP).ExistAbstract(tP)  # ∃t' tau(t,i,t',o) & Dst(t'))

        src_dst_conjuncts &= ~onion | (~err_bdd & exists_tP__tau_and_dstP)

    return ~inv_bdd | src_dst_conjuncts


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


def extract_output_funcs(non_det_strategy):
    """
    From a given non-deterministic strategy (the set of triples `(x,i,o)`),
    for each output variable `o`, calculate the set of pairs `(x,i)` where `o` will hold.
    There are different ways -- here we use cofactor-based approach.

    :return: dictionary `controllable_variable_bdd -> func_bdd`
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
        # Negation can cause including states from !W
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


def get_inv_err_j_bdds():
    assert spec.num_justice <= 1

    j_bdd = get_bdd_for_value(aiglib.get_justice_lit(spec, 0, 0)) \
            if spec.num_justice == 1 \
            else cudd.One()

    inv_bdd = cudd.One()
    if spec.num_constraints > 0:
        for i in range(spec.num_constraints):
            bdd = get_bdd_for_value(aiglib.get_aiger_symbol(spec.constraints, i).lit)
            inv_bdd = inv_bdd & bdd

    err_bdd = cudd.Zero()
    if spec.num_bad > 0:
        for i in range(spec.num_bad):
            bdd = get_bdd_for_value(aiglib.get_aiger_symbol(spec.bad, i).lit)
            err_bdd = err_bdd | bdd
    elif spec.num_outputs == 1:
        err_bdd = get_bdd_for_value(aiglib.get_aiger_symbol(spec.outputs, 0).lit)

    return inv_bdd, err_bdd, j_bdd


def assert_increasing(attractors):   # TODOopt: debug only
    previous = attractors[0]
    for a in attractors[1:]:
        if not (previous.Leq(a) and previous != a):
            logger.error('attractors are not strictly increasing')
            print 'a:'
            a.PrintMinterm()
            print 'previous:'
            previous.PrintMinterm()

            print 'len(attractors)', str(len(attractors))

            assert 0

        previous = a


def synthesize(realiz_check):
    """ Calculate winning region and extract output functions.

    :return: - if realizable: <True, dictionary: controllable_variable_bdd -> func_bdd>
             - if not: <False, None>
    """
    logger.info('synthesize..')

    #: :type: DdNode
    init_state_bdd = compose_init_state_bdd()
    # init_state_bdd.PrintMinterm()
    #: :type: DdNode
    transition_bdd = compose_transition_bdd()
    # transition_bdd.PrintMinterm()

    inv_bdd, err_bdd, j_bdd = get_inv_err_j_bdds()

    # ensure that depends on latches only: (\exists i1..in: a&b&c&i1==False) is not True  # TODO: lift to justice(t,i,o)
    all_inputs_bdds = get_uncontrollable_vars_bdds()+get_controllable_vars_bdds()
    assert (~(j_bdd.Support())).ExistAbstract(get_cube(all_inputs_bdds)) != cudd.One(), \
        'Mealy-like J signals are not supported'

    attractors = calc_win_region(init_state_bdd, transition_bdd, inv_bdd, err_bdd, j_bdd)
    if attractors is None:
        return False, None

    if realiz_check:
        return True, None

    non_det_strategy = get_nondet_strategy(attractors, transition_bdd, inv_bdd, err_bdd)

    func_by_var = extract_output_funcs(non_det_strategy)

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


def get_all_vars(bdd):
    #: :type: DdNode
    print(bdd.BddToCubeArray())


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
    # all variables used in BDDs are also present in AIGER
    a_lit = a_bdd.NodeReadIndex()

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
    # CUDD_REORDER_SAME,
    # CUDD_REORDER_NONE,
    # CUDD_REORDER_RANDOM,
    # CUDD_REORDER_RANDOM_PIVOT,
    # CUDD_REORDER_SIFT,
    # CUDD_REORDER_SIFT_CONVERGE,
    # CUDD_REORDER_SYMM_SIFT,
    # CUDD_REORDER_SYMM_SIFT_CONV,
    # CUDD_REORDER_WINDOW2,
    # CUDD_REORDER_WINDOW3,
    # CUDD_REORDER_WINDOW4,
    # CUDD_REORDER_WINDOW2_CONV,
    # CUDD_REORDER_WINDOW3_CONV,
    # CUDD_REORDER_WINDOW4_CONV,
    # CUDD_REORDER_GROUP_SIFT,
    # CUDD_REORDER_GROUP_SIFT_CONV,
    # CUDD_REORDER_ANNEALING,
    # CUDD_REORDER_GENETIC,
    # CUDD_REORDER_LINEAR,
    # CUDD_REORDER_LINEAR_CONVERGE,
    # CUDD_REORDER_LAZY_SIFT,
    # CUDD_REORDER_EXACT
    cudd.AutodynEnable(4)
    # cudd.AutodynDisable()
    # cudd.EnableReorderingReporting()


def main(aiger_file_name, out_file_name, output_full_circuit, realiz_check):
    """ Open AIGER file, synthesize the circuit and write the result to output file.

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

        # some model checkers do not like unordered variable names (when e.g. latch is > add)
        # aiger_reencode(spec)

        if out_file_name:
            aiger_open_and_write_to_file(spec, out_file_name)
        else:
            res, string = aiger_write_to_string(spec, aiger_ascii_mode, 2147483648)
            assert res != 0 or out_file_name is None, 'writing failure'
            print(string)   # print independently of logger level setup
        return True

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aiger Format Based Simple Synthesizer')
    parser.add_argument('aiger', metavar='aiger', type=str,
                        help='input specification in AIGER format')
    parser.add_argument('--out', '-o', metavar='out', type=str, required=False, default=None,
                        help='output file in AIGER format (if realizable)')
    parser.add_argument('--full', action='store_true', default=False,
                        help='produce a full circuit that has outputs other than error bit')
    parser.add_argument('--realizability', '-r', action='store_true', default=False,
                        help='Check Realizability only (do not produce circuits)')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Do not print anything but the model (if realizable)')

    args = parser.parse_args()

    logger = setup_logging(-1 if args.quiet else 0)

    is_realizable = main(args.aiger, args.out, args.full, args.realizability)

    logger.info(['unrealizable', 'realizable'][is_realizable])

    exit([EXIT_STATUS_UNREALIZABLE, EXIT_STATUS_REALIZABLE][is_realizable])
