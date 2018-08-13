import numpy as np
import scipy
from qutip.operators import qeye
from qutip.tensor import tensor

__all__ = ["mul_into","ctrl","U2H","H2U"]

def mul_into(A,B,subspace,left_mult = False):
    '''
    Multiply A into a subspace of B

              _____
    ---------|     |----
             |     |
    ---------|     |----
       ___   |     |
    --|   |--|  B  |----
      | A |  |     |
    --|___|--|     |----
             |     |
    ---------|_____|----
    
    Subspace can be any list of indicies. Need not be sequential.

    For example applying A on subspace [4,0] of B is equivlent to

                                              _____                              
    0--------.     .----------------.     .--|     |--
              \   /                  \   /   |     |           
    1----------\-/--------------------\-/----|     |--
                x       ___            x     |     |           
    2--.   .---/-\---0-|   |--.   .---/-\----|  B  |--
        \ /   /   \    | A |   \ /   /   \   |     |           
    3----x---^     ^-1-|___|----x---^     ^--|     |--
        / \                    / \           |     |           
    4--^   ^------------------^   ^----------|_____|--
    

    Optional arg left_mult flips the order such that A left multiplies B
    '''
    # note: cython this at some point
    # one should test for obvious dim mismatch
    # one should also do obvious case tests
    # one has not done these things
    # one is too busy refering to themselves in third person

    # subspace is continuous. is ordered. subspace is entire space. etc
    
    # obvious testing done. now we must do the operation for real
    dim = B.dims[0]
    ops = [A]
    if left_mult:
        dim = B.dims[1]
    
    for k in range(len(dim)):
        if k not in subspace:
            ops.append(qeye(dim[k]))
    big_A = tensor(ops) # does this work like I think?

    # now permute big_A and we are done.
    rearr = np.arange(len(dim))
    for k in range(len(subspace)):
        rearr[subspace[k]] = k
        rearr[k] = subspace[k]

    if left_mult:
        return B * big_A.permute(rearr)
    else:
        return big_A.permute(rearr) * B
    

    
    # My original approach to this problem is below. I couldn't make it work but
    # maybe I'll try again. I'm leaving it here commented out.

    
##    dims = B.dims[1]
##    if B.shape[0] > B.shape[1]:
##        dims = B.dims[0]
##    
##    adj_dims = list(dims)
##    adj_dims[-1] = 1
##    for k in range(len(dims)-1, 0, -1):
##        adj_dims[k-1] = adj_dims[k]*dims[k]
##
##    rr_ind = [] # soon to be the indices in a csr
##    rc_ind = [] # soon to be the indices in a csc
##    
##    for to in range(max(B.shape)):
##        from_row = 0
##        from_col = 0
##        # adj_dims will let us convert between matrix index and subsystem index
##        for k in range(len(subspace)):
##            from_row *= A.dims[0][k]
##            from_col *= A.dims[1][k]
##            from_row += (to//adj_dims[subspace[k]]) % dims[subspace[k]]
##            from_col += (to//adj_dims[subspace[k]]) % dims[subspace[k]]
##            #maybe an if statement here
##
##        rr_ind.append(from_col)
##        rc_ind.append(from_row)
##
##    vals = np.zeros(max(B.shape)) + 1
##    ptr = np.arange(max(B.shape)+1)
##
##    rr_mat = csr_matrix((vals,rr_ind,ptr),(max(B.shape),max(A.shape)))
##    rc_mat = csc_matrix((vals,rc_ind,ptr),(max(A.shape),max(B.shape)))
##
##    # now if everything is right rr_mat * A * rc_mat should put A in the full space of B
##    
##    return rr_mat, A.data, rc_mat


def ctrl(U):
    '''turns an operator into a controlled operator


    -------      ---.---
             ==>    |
    ---U---      ---U---

    '''
    #should I be doing something special for super operators? 
    if U.shape[0] != U.shape[1]:
        raise Exception("There's no such thing as a controlled ket")
    size_inc = U.shape[0] # add this many 1s to the diagonal
    newdat = np.append(np.zeros([size_inc],dtype="int32")+1,U.data.data)
    newindc = np.append(np.arange(size_inc,dtype="int32"),U.data.indices + size_inc)
    newptr = np.append(np.arange(size_inc,dtype="int32"),U.data.indptr + size_inc)

    newU = Qobj(U)
    newU.data = qutip.fastsparse.fast_csr_matrix((newdat,newindc,newptr),
                shape = (2*size_inc,2*size_inc))

    newU.dims = [[2] + U.dims[0], [2] + U.dims[1]]
    
    return newU
    
def U2H(U):
    pass # this should be a pretend optimal control function

def H2U(H, start = 0, end = np.pi):
    pass
