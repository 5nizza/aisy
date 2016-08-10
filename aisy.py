#!/usr/bin/env python2.7
# coding=utf-8

"""     _ + _ . .
       |_||!_' Y
       | ||._! |

Simple GR1 synthesizer from
[AIGER-like](http://fmv.jku.at/aiger/) GR1 format.
The format is described [here](https://github.org/5nizza/spec-framework).

Gmail me: ayrat.khalimov
"""

import argparse
import pycudd
import aiger_swig.aiger_wrap as aiglib

from aiger_swig.aiger_wrap import *
from ansistrm import setup_logging


EXIT_STATUS_REALIZABLE = 10
EXIT_STATUS_UNREALIZABLE = 20


#: :type: aiger
spec = None

#: :type: DdManager
cudd = None

#: :type: Logger
logger = None

# to cache equal Sub-BDDs
cache_dict = dict()


def is_negated(l):
    return (l & 1) == 1


def strip_lit(l):
    return l & ~1


def iterate_latches():
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
    assert (spec.num_outputs == 1) ^ (spec.num_bad or spec.num_justice or spec.num_fairness), 'mix of two formats'
    assert spec.num_outputs + spec.num_justice + spec.num_bad >= 1, 'no properties'

    assert spec.num_justice <= 1, 'not supported'
    assert spec.num_fairness <= 1, 'not supported'


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

    logger.info('compose_transition_bdd, nof_latches={0}...'.format(len(list(iterate_latches()))))

    #: :type: DdNode
    transition = cudd.One()
    for l in iterate_latches():
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
    bdds = [get_bdd_for_value(l.lit) for l in iterate_latches()]
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


def sys_predecessor(dst_bdd, trans_bdd, env_bdd, sys_bdd):
    """
    Calculate controllable predecessor of dst

       ∀i ∃o: env(t,i,o) -> [ sys(t,i,o) & ∃t' tau(t,i,o,t') & dst(t') ]

    :return: BDD representation of the predecessor states
    """

    next_dst = prime_latches_in_bdd(dst_bdd)
    tau_and_dst = trans_bdd & next_dst

    # cudd requires to create a cube first
    next_state_vars_cube = prime_latches_in_bdd(get_cube(get_all_latches_as_bdds()))
    E_tn_tau_and_dst = tau_and_dst.ExistAbstract(next_state_vars_cube)  # ∃t'  tau(t,i,t',o) & dst(t')

    sys_and_E_tn_tau_and_dst = sys_bdd & E_tn_tau_and_dst
    env_impl_sys_and_tau = ~env_bdd | sys_and_E_tn_tau_and_dst

    # ∃o: env -> (sys & ∃t' tau(t,i,t',o)):
    out_vars_cube = get_cube(get_controllable_vars_bdds())
    E_o_implication = env_impl_sys_and_tau.ExistAbstract(out_vars_cube)

    # ∀i ∃o: inv -> ..
    inp_vars_cube = get_cube(get_uncontrollable_vars_bdds())
    A_i_E_o_implication = E_o_implication.UnivAbstract(inp_vars_cube)

    return A_i_E_o_implication


def calc_win_region(trans_bdd,
                    env_bdd, sys_bdd,
                    fair_bdd, just_bdd):
    """
    The mu-calculus formula for 1-Streett is
      gfp.Z lfp.Y gfp.X [ just & Cpre(Z') | Cpre(Y') | !fair & Cpre(X') ]

    Recall that the mu-calculus formula for Buechi is
      gfp.Z lfp.Y [ just & Cpre(Z') | Cpre(Y') ]

    What 1-Streett formula does is:
    internal lfp.Y computes:
    0. Y0 = 0
    1. Y1 = just, lassos that fall out into just
    2. Y2 = just, lassos that fall out into just | Cpre(Y1)
    3. Y3 = just, lassos that fall out into just | Cpre(Y2)
    ...
    One invariant is:
    from Y[r] sys either reaches just via path that visits <r fair states,
                      or loops in !fair forever except possibly for <r moments where it visits a fair state.

    The external gfp.Z is decreasing, and is somewhat similar to Buechi win set computation:
    it gradually removes states from which we cannot visit just once, or twice, thrice...
    and it accounts for the possibility to end in !fair lassos.

    See also: [notes/1-streett-pair-mu-calculus.jpg](gfp.Y calculation)

    :return: Z:bdd, Y:list(increasing)
    """

    logger.info('calc_win_region..')

    Cpre = lambda dst: sys_predecessor(dst, trans_bdd, env_bdd, sys_bdd)

    Z = cudd.One()
    prevZ = None
    while Z != prevZ:
        Ys = list()
        Y = cudd.Zero()
        prevY = None
        while Y != prevY:
            Ys.append(Y)

            Xs = list()
            X = cudd.One()
            prevX = None
            while X != prevX:
                Xs.append(X)

                prevX = X
                X = just_bdd & Cpre(Z) | Cpre(Y) | ~fair_bdd & Cpre(prevX)
            prevY = Y
            Y = X
        prevZ = Z
        Z = Y

    return Z, Ys


def get_nondet_strategy(Z_bdd, Ys,
                        trans_bdd,
                        env_bdd, sys_bdd,
                        fair_bdd, just_bdd):

    """
    The strategy extraction is:

    - rho1 = just & sys(t,i,o) & Z(t')

    - rho2 = OR{r>1}: Y[r]&~Y[r-1] & sys(t,i,o) & Y[r-1](t')
      (Y[0] = 0,
       Y[1] = lfp.X [just & Cpre(Z') | !fair & Cpre(X')],
       and we take care of this in rho1 or in rho3)

    - rho3 = OR{r}: Y[r]&~Y[r-1] & !fair & sys(t,i,o) & Y[r](t')
      Note: we cannot go higher >r, because this does not guarantee: GFfair -> GFjust
            (we could have GF fair and never visit just)
            (recall that Y[r] may contain fair states)

    - strategy = env(t,i,o) & Z(t) -> ∃t': trans(t,i,o,t') & rho1(t,i,o,t')|rho2(t,i,o,t')|rho3(t,i,o,t')

    :return: non deterministic strategy bdd
    :note: The strategy is not-deterministic -- determinization step is done later.
    """

    logger.info('get_nondet_strategy..')
    # TODO: optimize for special cases: safety, buechi (hangs on the huffman example)

    assert_increasing(Ys)

    onion = lambda i: Ys[i] & ~Ys[i-1] if i>0 else Ys[i]

    rho1 = just_bdd & sys_bdd & prime_latches_in_bdd(Z_bdd)

    rho2 = cudd.Zero()
    for r in range(2, len(Ys)):
        rho2 |= onion(r) & sys_bdd & prime_latches_in_bdd(Ys[r-1])

    rho3 = cudd.Zero()
    for r in range(0, len(Ys)):
        rho3 |= onion(r) & ~fair_bdd & sys_bdd & prime_latches_in_bdd(Ys[r])

    strategy = ~env_bdd | ~Z_bdd | (trans_bdd & (rho1 | rho2 | rho3)).ExistAbstract(
        prime_latches_in_bdd(get_cube(get_all_latches_as_bdds())))

    return strategy


def compose_init_state_bdd():
    """ Initial state is 'all latches are zero' """

    logger.info('compose_init_state_bdd..')

    init_state_bdd = cudd.One()
    for l in iterate_latches():
        l_curr_value_bdd = get_bdd_for_value(l.lit)
        init_state_bdd &= make_bdd_eq(l_curr_value_bdd, cudd.Zero())

    return init_state_bdd


def extract_output_funcs(non_det_strategy_bdd):
    """
    From a given non-deterministic strategy (the set of triples `(x,i,o)`),
    for each output variable `o`, calculate the set of pairs `(x,i)` where `o` will hold.
    There are different ways -- here we use cofactor-based approach.

    :return: dictionary `controllable_variable_bdd -> func_bdd`
    """

    logger.info('extract_output_funcs..')

    output_models = dict()
    controls = get_controllable_vars_bdds()

    # build list with all variables
    all_vars = get_uncontrollable_vars_bdds()
    all_vars.extend(get_all_latches_as_bdds())

    for c in get_controllable_vars_bdds():
        logger.info('getting output function for ' + aiger_is_input(spec, strip_lit(c.NodeReadIndex())).name)

        others = set(set(controls).difference({c}))
        if others:
            others_cube = get_cube(others)
            #: :type: DdNode
            c_arena = non_det_strategy_bdd.ExistAbstract(others_cube)
        else:
            c_arena = non_det_strategy_bdd

        # c_arena.PrintMinterm()

        can_be_true = c_arena.Cofactor(c)  # states (x,i) in which c can be true
        can_be_false = c_arena.Cofactor(~c)

        # We need to intersect with can_be_true to narrow the search.
        # Negation can cause including states from !W
        #: :type: DdNode
        must_be_true = (~can_be_false) & can_be_true
        must_be_false = (~can_be_true) & can_be_false

        # implementation of variable minimization
        for v in all_vars:

            must_be_true_prime = must_be_true.ExistAbstract(v)
            must_be_false_prime = must_be_false.ExistAbstract(v)

            # (must_be_false_prime & must_be_true_prime) should be UNSAT
            if(must_be_false_prime & must_be_true_prime) == cudd.Zero():
                must_be_true = must_be_true_prime
                must_be_false = must_be_false_prime


        care_set = (must_be_true | must_be_false)

        # We use 'restrict' operation, but we could also do just:
        # c_model = care_set -> must_be_true
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

        non_det_strategy_bdd = non_det_strategy_bdd & make_bdd_eq(c, c_model)

    return output_models


def get_inv_err_f_j_bdds():
    assert spec.num_justice <= 1
    assert spec.num_fairness <= 1

    j_bdd = get_bdd_for_value(aiglib.get_justice_lit(spec, 0, 0)) \
            if spec.num_justice == 1 \
            else cudd.One()

    f_bdd = get_bdd_for_value(aiglib.get_aiger_symbol(spec.fairness, 0).lit) \
        if spec.num_fairness == 1 \
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

    return inv_bdd, err_bdd, f_bdd, j_bdd


def assert_increasing(attractors):   # TODOopt: debug only
    previous = attractors[0]
    for a in attractors[1:]:
        if not (previous.Leq(a) and previous != a):
            logger.error('attractors are not strictly increasing')
            print 'a:'
            a.PrintMinterm()
            print 'previous:'
            previous.PrintMinterm()

            print 'len(attractors):', str(len(attractors))

            assert 0

        previous = a


def assert_liveness_is_Moore(bdd, sig):
    all_inputs_bdds = get_uncontrollable_vars_bdds() + get_controllable_vars_bdds()
    assert (~(bdd.Support())).ExistAbstract(get_cube(all_inputs_bdds)) != cudd.One(), \
        'Mealy-like %s signals are not supported' % sig


def synthesize(realiz_check):
    """ Calculate winning region and extract output functions.

    :return: - if realizable: <True, dictionary: controllable_variable_bdd -> func_bdd>
             - if not: <False, None>
    """
    logger.info('synthesize..')

    #: :type: DdNode
    init_bdd = compose_init_state_bdd()
    # init_state_bdd.PrintMinterm()
    #: :type: DdNode
    trans_bdd = compose_transition_bdd()
    # transition_bdd.PrintMinterm()

    env_bdd, err_bdd, f_bdd, j_bdd = get_inv_err_f_j_bdds()

    # ensure that depends on latches only: (\exists i1..in: a&b&c&i1==False) is not True  # TODO: lift to justice(t,i,o)
    assert_liveness_is_Moore(j_bdd, 'J')
    assert_liveness_is_Moore(f_bdd, 'F')

    Z, Ys = calc_win_region(trans_bdd,
                            env_bdd, ~err_bdd,
                            f_bdd, j_bdd)

    if not (init_bdd <= Z):
        return False, None

    if realiz_check:
        return True, None

    non_det_strategy = get_nondet_strategy(Z, Ys,
                                           trans_bdd,
                                           env_bdd, ~err_bdd,
                                           f_bdd, j_bdd)

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

    # caching
    cached_lit = cache_dict.get(a_bdd.Regular(), None)
    if(cached_lit != None):
        return ( negated(cached_lit) if a_bdd.IsComplement() else cached_lit)
    # end caching

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

    # save all bdd's to caching dictionary
    cache_dict[a_bdd.Regular()] = res

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
    parser = argparse.ArgumentParser(description='Simple synthesizer from AIGER GR1 format')
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
