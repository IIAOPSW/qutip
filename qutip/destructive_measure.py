from scipy.sparse import csr_matrix,csc_matrix
import numpy as np
from bisect import bisect_left
from qutip.dimensions import tensor_idx_to_pretensor_idx
from qutip.tensor import tensor
from qutip.operators import qeye
from qutip.states import basis

def measure(state,subspace):
    '''
    state is either a pure or mixed state

    subspace is a list of indicies

    measures the systems in the subspace specified
    causing a wavefunction collapse
    tells you the result
    '''
    bstate = state.sample()
    substate = tuple([bstate[k] for k in subspace])
    post_select(state, subspace, substate)
    return substate
            

def post_select(state,subspace,outcome):
    '''
    state is either a pure or mixed state

    subspace is a list of indicies
    
    outcome may be a pure state, mixed state, or tuple.

    if tuple then it is to be interpreted as an eigenstate
    eg (1,0,1) means post select on outcome |1,0,1>

    if pure state or mixed state then state gets projected
    on to the outcome.
    '''
    # add some sanity checks.
    if isinstance(outcome,tuple):
        # I could just construct |outcome> and then the projector but
        # this way is slightly more efficent than the mul_into function
        ss2out = dict()
        for k in range(len(subspace)):
            ss2out[subspace[k]] = outcome[k]
            # map subspace to outcome
        totens = [] # to tensor
        for k in range(len(state.dims[0])):
            dim = max(state.dims[0][k],state.dims[1][k])
            if k in ss2out: # we are in the subspace, project to outcome
                totens.append(basis(dim, ss2out[k]).proj())
            else: # we are not in the subspace, do nothing (identity)
                totens.append(qeye(dim))
        finalproj = tensor(totens)
        if state.type == 'bra':
            fs = state * finalproj
        else:
            fs = finalproj * state
    else:
        if outcome.type != 'oper':
            outcome = outcome.proj()
        if state.type == 'bra':
            fs = ~mul_into(~outcome, ~state, subspace)
        else:
            fs = mul_into(outcome, state, subspace)
    # Now have final state

    renorm = fs.norm()

    if renorm == 0:
        fs = 0 * fs
        # edge case where post selection must fail
        # maybe this should throw error instead??
    else:
        fs = fs/renorm

    state.set_data(fs.data)
    return renorm**2
    

    
