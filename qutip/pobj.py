import numpy as np
from qutip import *
import scipy

from qutip.dimensions import pretensor_idx_to_tensor_idx, tensor_idx_to_pretensor_idx

# mind your p's and q's
class Pobj:
    '''
    A P object is a programatic Q object. It is multiply only.
    Pobj*Qobj => Qobj
    Qobj*Pobj => error
    A Pobj is meant to represent operators which can be defined
    with far less data than what is needed in their full matrix
    description.
    
    There is no data per se in a Pobj. A Pobj is defined
    by functions which map basis states (represented by integers)
    to vectors or kets
    
    For example suppose you wanted to make the classical function
    f(x) = x^2 into the quantum oracle:
            _____
    |x>----|    |----|x>
           | U  |
    |y>----|____|----|f(x) xor y>

    Pobj lets you turn lambda x: x**2 into this operator

    Warning: unitarity is not enforced. Don't do anything unphysical


    Pobj supports scalar multiplication a*Pobj => Pobj
    Pobj supports multiplication Pobj*Qobj => Qobj
    Pobj supports multiplication Qobj*Pobj => Pobj
    Pobj supports addition Pobj + Pobj => Pobj
    Pobj supports addition Pobj + Qobj => Pobj
    Pobj supports addition Qobj + Pobj => Pobj
    Pobj supports tensor Pobj|Pobj => Pobj
    Pobj supports tensor Pobj|Qobj => Pobj
    Pobj supports tensor Qobj|Pobj => Pobj
    Pobj supports dagger when dagger functions are provided
    Pobj supports ctrl(Pobj)

    A Pobj can be converted to a Qobj by multiplying by the identity Qobj
    A Qobj can be converted to a Pobj by multiplying by the identity Pobj
    '''
    # internally we want a lin sum of funcs, their daggers, a plain old matrix
    # and some dims. Missing anything?
    # meta name tag would be nice
    # time dependence is also nice
    def __init__(this, func, dims, func_dag = None, **kwargs):
        if func == 0:
            func = zero_func(np.prod(dims[0]))
        this.mode = kwargs.get("mode","add")
        this.data = kwargs.get("data",[[1, 0, func, func_dag]]) #[weight, time, f(x,t), f^{\dagger}(x,t)]
        this.dims = dims

    def cast(this, mode):
        if this.mode == mode:
            return this
        ret = Pobj(this, this.dims, this.dag())
        ret.mode = mode
        return ret

    def __add__(this, other):
        if isinstance(other, Qobj):
            newq = Q2P(other)
            return this + Pobj(newq, other.dims, newq.dag())
        a = this.cast("add")
        b = other.cast("add")
        return Pobj(0, a.dims, data = a.data + b.data)

    def __radd__(this, other):
        return this.__add__(other)

    def __mul__(this, other):
        if isinstance(other, Qobj):
            col2ket = dict()
            
            row = 0
            col = 0
            for idx in range(other.data.nnz):
                col = other.data.indices[idx]
                while other.data.indptr[row+1] < idx + 1:
                    row += 1
                if col not in col2ket:
                    col2ket[col] = other.data.data[idx]*this(0, row)
                else:
                    col2ket[col] += other.data.data[idx]*this(0, row)

            if len(col2ket) == 1: #output is a ket and we can exit early
                for k in col2ket:
                    return Qobj(col2ket[k], other.dims)
            
            for k in col2ket:
                if isinstance(col2ket,Qobj):
                    col2ket[k] = col2ket[k].data.tocsc()
                else:
                    col2ket[k] = col2ket[k].data.tocsc()
                    
            nothing = csc_matrix((0,0))
            
            indptr = np.array([0])
            indices = np.array([])
            data = np.array([])
            
            for i in range(other.shape[1]):
                indptr = np.append(indptr, col2ket.get(i,nothing).nnz + indptr[-1])
                indices = np.append(indices, col2ket.get(i,nothing).indices)
                data = np.append(data, col2ket.get(i,nothing).data)
            
            outbound = scipy.sparse.csc_matrix((data, indices, indptr), other.shape)
            return Qobj(outbound.tocsr(), other.dims)
        
        elif type(this) == type(other):
            newf = lambda t,x: this*this.ensureq(other(t,x))
            newfdag = lambda t,x: other.dag()*this.ensureq(this.dag()(t,x))
            return Pobj(newf, this.dims, newfdag)
        else: # must be numeric
            if this.mode == "add":
                newd = []
                for x in this.data:
                    newd += [[x[0]*other, x[1], x[2], x[3]]]
                return Pobj(0, this.dims, data = newd, mode = this.mode)
            else:
                newd = this.data[:]
                newd[0][0] *= other
                return Pobj(0, this.dims, data = newd, mode = this.mode)

    def __rmul__(this, other):
        if isinstance(other, Qobj):
            newq = Q2P(other)
            return Pobj(newq, other.dims, newq.dag()).__mul__(this)
        else:
            return this.__mul__(other)
        
    def __or__(this, other):
        if isinstance(other, Qobj):
            newq = Q2P(other)
            return this|Pobj(newq, other.dims, newq.dag())
        a = this.cast("tensor")
        b = this.cast("tensor")
        d0 = a.dims[0] + b.dims[0]
        d1 = a.dims[1] + b.dims[1]
        return Pobj(0, [d0,d1], data = a.data + b.data, mode = "tensor")

    def __ror__(this, other):
        if isinstance(other, Qobj):
            newq = Q2P(other)
            return Pobj(newq, other.dims, newq.dag()).__or__(this)
        else:
            return other.__or__(this)
    
    def __call__(this, t, x = None):
        if x == None: # caller is asking for a copy of the pobj phase shifted to time t
            newd = []
            for x in this.data:
                newd += [[x[0], x[1] + t, x[2], x[3]]]
            return Pobj(0, this.dims, data = newd, mode = this.mode)
        if this.mode == "add":
            d = this.data[0]
            res = d[0]*d[2](t, x)
            for d in this.data[1:]:
                temp = d[0]*d[2](t, x)
                if type(temp) == type(res):
                    res += temp
                else:
                    res = this.ensureq(res) + this.ensureq(temp)
            return res
        if this.mode == "tensor":
            rdims = 1
            subx = []
            # break the input argument |x> into |x0>|x1>|x2>...
            for d in range(len(this.data)-1,-1,-1):
                cdims = np.prod(this.data[d][2].dims[0])
                subx += [(x//rdims)%cdims]
                rdims *= cdims
            subx.reverse()
            subu = [] # sub u = [u0|x0>, u1|x1>, u2|x2>...]
            for i in range(len(this.data)):
                d = this.data[i][2].dims[0]
                subu += [this.data[i][0]*this.ensureq(this.data[i][2](this.data[i][1] + t, subx[i]), [d,[1]*len(d)])]
            return tensor(subu)
        if this.mode == "ctrl":
            tot = np.prod(this.dims[0])//2
            if x//tot == 1:
                d = this.dims[0][1:]
                ret = this.data[0][0]*this.data[0][2](t, x)
                return basis(2,1)|this.ensureq(ret,[d,[1]*len(d)])
            else:
                ret = scipy.sparse.dok_matrix((2*tot,1))
                ret[x,0] = this.data[0][0]
                return ret

    def ctrl(this): 
        return Pobj(this, [[2]+this.dims[0], [2]+this.dims[1]], this.dag(), mode = "ctrl")

    def permute(this, rearr): #test this at some point
        f = lambda t,x: this.ensureq(this(t,x)).permute(rearr)
        a = this.dag()
        g = lambda t,x: d.ensureq(a(t,x)).permute(rearr)
        return Pobj(f,this.dims,g)
    
    def dag(this):
        newd = []
        for x in this.data:
            newd += [[x[0], x[1], x[3], x[2]]]
        return Pobj(0, this.dims, data = newd, mode = this.mode)

    def ensureq(this, a, dim = None):# something missing here. 
        if isinstance(a,Qobj):
            return a
        if dim == None:
            dim = [this.dims[0],[1]*len(this.dims[1])]
        return Qobj(a, dim)

    
class Q2P:
    def __init__(this, q, dagger = False):
        if type(q) == type(this):
            this.qd = q.qd
            this.q = q.q
        else:
            this.qd = q.data.conjugate()
            this.q = q.data.tocsc()
        this.dagger = dagger

    def __call__(this, t, x):
        if this.dagger:
            return this.qd.getrow(x)
        else:
            return this.q.getcol(x)
            
    def dag(this):
        return Q2P(this, not this.dagger)

class id_func:
    def __init__(this, N):
        this.N = N

    def __call__(this, t, x):
        return scipy.sparse.csr_matrix(([1],([x],[0])), shape = (this.N,1))

    def dag(this):
        this

class zero_func:
    def __init__(this, N):
        this.N = N

    def __call__(this, t, x):
        return scipy.sparse.csr_matrix((this.N,1))

    def dag(this):
        return this

def peye(N):
    f = id_func(N)
    return Pobj(f, [[N],[N]], f)


# this stuff will eventually go in qubit_qc
class oracle_func:
    def __init__(this, f, m, n):
        this.f = f
        this.m = m
        this.n = n

    def __call__(this, t, x):
        y_reg = x % 2**this.n
        x_reg = (x // 2**this.n) % 2**this.m
        y_reg = this.f(x_reg) ^ y_reg
        retket = scipy.sparse.dok_matrix((2**(this.m + this.n),1))
        whereat = (y_reg + x_reg*2**this.n) % 2**(this.m + this.n)
        retket[whereat,0] = 1
        return retket

def oraclize(f,m,n):
    '''
    Turns a classical function into a quantum oracle with m qubits on the input
    register and n qubits on the output register.

           m    ______
    |x> ---/---|      |---|x>
           n   |oracle|
    |y> ---/---|______|---|f(x) xor y>

    
    The classical function is literally a function written in python
    eg
    
    def f(x):
        return x**2

    Only requirement is that f(x) maps positive ints to positive ints
    '''
    of = oracle_func(f,m,n)
    return Pobj(of, [2]*(m+n), of)

