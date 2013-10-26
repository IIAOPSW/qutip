# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################

import numpy as np
from scipy import arange, conj, prod
import scipy.sparse as sp

try:  # for scipy v <= 0.90
    from scipy import factorial
except:  # for scipy v >= 0.10
    from scipy.misc import factorial

from qutip.qobj import Qobj
from qutip.operators import destroy
from qutip.tensor import tensor


def basis(N, n=0, offset=0):
    """Generates the vector representation of a Fock state.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    n : int
        Integer corresponding to desired number state, defaults
        to 0 if omitted.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    Returns
    -------
    state : qobj
      Qobj representing the requested number state ``|n>``.

    Examples
    --------
    >>> basis(5,2)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]]

    Notes
    -----

    A subtle incompatibility with the quantum optics toolbox: In QuTiP::

        basis(N, 0) = ground state

    but in the qotoolbox::

        basis(N, 1) = ground state

    """
    if (not isinstance(N, (int, np.int64))) or N < 0:
        raise ValueError("N must be integer N >= 0")

    if (not isinstance(n, (int, np.int64))) or n < offset:
        raise ValueError("n must be integer n >= 0")

    if n - offset > (N - 1):  # check if n is within bounds
        raise ValueError("basis vector index need to be in n <= N-1")

    bas = sp.lil_matrix((N, 1))  # column vector of zeros
    bas[n-offset, 0] = 1  # 1 located at position n
    bas = bas.tocsr()

    return Qobj(bas)


def qutrit_basis():
    """Basis states for a three level system (qutrit)

    Returns
    -------
    qstates : array
        Array of qutrit basis vectors

    """
    return np.array([basis(3, 0), basis(3, 1), basis(3, 2)])


def _sqrt_factorial(n_vec):
    # take the square root before multiplying
    return np.array([np.prod(np.sqrt(np.arange(1, n + 1))) for n in n_vec])


def coherent(N, alpha, offset=0, method='operator'):
    """Generates a coherent state with eigenvalue alpha.

    Constructed using displacement operator on vacuum state.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue of coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state. Using a non-zero offset will make the 
        default method 'analytic'.

    method : string {'operator', 'analytic'}
        Method for generating coherent state.

    Returns
    -------
    state : qobj
        Qobj quantum object for coherent state

    Examples
    --------
    >>> coherent(5,0.25j)
    Quantum object: dims = [[5], [1]], shape = [5, 1], type = ket
    Qobj data =
    [[  9.69233235e-01+0.j        ]
     [  0.00000000e+00+0.24230831j]
     [ -4.28344935e-02+0.j        ]
     [  0.00000000e+00-0.00618204j]
     [  7.80904967e-04+0.j        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent state is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting state is normalized. With 'analytic' method the coherent state
    is generated using the analytical formula for the coherent state
    coefficients in the Fock basis. This method does not guarantee that the
    state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    if method == "operator" and offset == 0:

        x = basis(N, 0)
        a = destroy(N)
        D = (alpha * a.dag() - conj(alpha) * a).expm()
        return D * x

    elif method == "analytic" or offset > 0:

        data = np.zeros([N, 1], dtype=complex)
        n = arange(N) + offset
        data[:, 0] = np.exp(-(abs(alpha) ** 2) / 2.0) * (alpha ** (n)) / \
            _sqrt_factorial(n)
        return Qobj(data)

    else:
        raise TypeError(
            "The method option can only take values 'operator' or 'analytic'")


def coherent_dm(N, alpha, offset=0, method='operator'):
    """Density matrix representation of a coherent state.

    Constructed via outer product of :func:`qutip.states.coherent`

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    alpha : float/complex
        Eigenvalue for coherent state.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the state.

    method : string {'operator', 'analytic'}
        Method for generating coherent density matrix.

    Returns
    -------
    dm : qobj
        Density matrix representation of coherent state.

    Examples
    --------
    >>> coherent_dm(3,0.25j)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.93941695+0.j          0.00000000-0.23480733j -0.04216943+0.j        ]
     [ 0.00000000+0.23480733j  0.05869011+0.j          0.00000000-0.01054025j]
     [-0.04216943+0.j          0.00000000+0.01054025j  0.00189294+0.j\
        ]]

    Notes
    -----
    Select method 'operator' (default) or 'analytic'. With the
    'operator' method, the coherent density matrix is generated by displacing
    the vacuum state using the displacement operator defined in the
    truncated Hilbert space of size 'N'. This method guarantees that the
    resulting density matrix is normalized. With 'analytic' method the coherent
    density matrix is generated using the analytical formula for the coherent
    state coefficients in the Fock basis. This method does not guarantee that
    the state is normalized if truncated to a small number of Fock states,
    but would in that case give more accurate coefficients.

    """
    if method == "operator":
        psi = coherent(N, alpha, offset=offset)
        return psi * psi.dag()

    elif method == "analytic":
        psi = coherent(N, alpha, offset=offset, method='analytic')
        return psi * psi.dag()

    else:
        raise TypeError(
            "The method option can only take values 'operator' or 'analytic'")


def fock_dm(N, n=0, offset=0):
    """Density matrix representation of a Fock state

    Constructed via outer product of :func:`qutip.states.fock`.

    Parameters
    ----------
    N : int
        Number of Fock states in Hilbert space.

    n : int
        ``int`` for desired number state, defaults to 0 if omitted.

    Returns
    -------
    dm : qobj
        Density matrix representation of Fock state.

    Examples
    --------
     >>> fock_dm(3,1)
     Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
     Qobj data =
     [[ 0.+0.j  0.+0.j  0.+0.j]
      [ 0.+0.j  1.+0.j  0.+0.j]
      [ 0.+0.j  0.+0.j  0.+0.j]]

    """
    psi = basis(N, n, offset=offset)

    return psi * psi.dag()


def fock(N, n=0, offset=0):
    """Bosonic Fock (number) state.

    Same as :func:`qutip.states.basis`.

    Parameters
    ----------
    N : int
        Number of states in the Hilbert space.

    n : int
        ``int`` for desired number state, defaults to 0 if omitted.

    Returns
    -------
        Requested number state :math:`\\left|n\\right>`.

    Examples
    --------
    >>> fock(4,3)
    Quantum object: dims = [[4], [1]], shape = [4, 1], type = ket
    Qobj data =
    [[ 0.+0.j]
     [ 0.+0.j]
     [ 0.+0.j]
     [ 1.+0.j]]

    """    
    return basis(N, n, offset=offset)


def thermal_dm(N, n, method='operator'):
    """Density matrix for a thermal state of n particles

    Parameters
    ----------
    N : int
        Number of basis states in Hilbert space.

    n : float
        Expectation value for number of particles in thermal state.

    method : string {'operator', 'analytic'}
        ``string`` that sets the method used to generate the
        thermal state probabilities

    Returns
    -------
    dm : qobj
        Thermal state density matrix.

    Examples
    --------
    >>> thermal_dm(5, 1)
    Quantum object: dims = [[5], [5]], \
shape = [5, 5], type = oper, isHerm = True
    Qobj data =
    [[ 0.51612903  0.          0.          0.          0.        ]
     [ 0.          0.25806452  0.          0.          0.        ]
     [ 0.          0.          0.12903226  0.          0.        ]
     [ 0.          0.          0.          0.06451613  0.        ]
     [ 0.          0.          0.          0.          0.03225806]]


    >>> thermal_dm(5, 1, 'analytic')
    Quantum object: dims = [[5], [5]], \
shape = [5, 5], type = oper, isHerm = True
    Qobj data =
    [[ 0.5      0.       0.       0.       0.     ]
     [ 0.       0.25     0.       0.       0.     ]
     [ 0.       0.       0.125    0.       0.     ]
     [ 0.       0.       0.       0.0625   0.     ]
     [ 0.       0.       0.       0.       0.03125]]

    Notes
    -----
    The 'operator' method (default) generates
    the thermal state using the truncated number operator ``num(N)``. This
    is the method that should be used in computations. The
    'analytic' method uses the analytic coefficients derived in
    an infinite Hilbert space. The analytic form is not necessarily normalized,
    if truncated too aggressively.

    """
    if n == 0:
        return fock_dm(N, 0)
    else:
        i = arange(N)
        if method == 'operator':
            beta = np.log(1.0 / n + 1.0)
            diags = np.exp(-beta * i)
            diags = diags / np.sum(diags)
            # populates diagonal terms using truncated operator expression
            rm = sp.spdiags(diags, 0, N, N, format='csr')
        elif method == 'analytic':
            # populates diagonal terms using analytic values
            rm = sp.spdiags((1.0 + n) ** (-1.0) * (n / (1.0 + n)) ** (i),
                            0, N, N, format='csr')
        else:
            raise ValueError(
                "'method' keyword argument must be 'operator' or 'analytic'")
    return Qobj(rm)


def ket2dm(Q):
    """Takes input ket or bra vector and returns density matrix
    formed by outer product.

    Parameters
    ----------
    Q : qobj
        Ket or bra type quantum object.

    Returns
    -------
    dm : qobj
        Density matrix formed by outer product of `Q`.

    Examples
    --------
    >>> x=basis(3,2)
    >>> ket2dm(x)
    Quantum object: dims = [[3], [3]], \
shape = [3, 3], type = oper, isHerm = True
    Qobj data =
    [[ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  0.+0.j]
     [ 0.+0.j  0.+0.j  1.+0.j]]

    """
    if Q.type == 'ket':
        out = Q * Q.dag()
    elif Q.type == 'bra':
        out = Q.dag() * Q
    else:
        raise TypeError("Input is not a ket or bra vector.")
    return Qobj(out)


#
# projection operator
#
def projection(N, n, m, offset=0):
    """The projection operator that projects state |m> on state |n>: |n><m|.

    Parameters
    ----------

    N : int
        Number of basis states in Hilbert space.

    n, m : float
        The number states in the projection.

    offset : int (default 0)
        The lowest number state that is included in the finite number state
        representation of the projector.

    Returns
    -------
    oper : qobj
         Requested projection operator.
    """
    ket1 = basis(N, n, offset=offset)
    ket2 = basis(N, m, offset=offset)

    return ket1 * ket2.dag()


#
# composite qubit states
#
def qstate(string):
    """Creates a tensor product for a set of qubits in either
    the 'up' :math:`|0>` or 'down' :math:`|1>` state.

    Parameters
    ----------
    string : str
        String containing 'u' or 'd' for each qubit (ex. 'ududd')

    Returns
    -------
    qstate : qobj
        Qobj for tensor product corresponding to input string.

    Examples
    --------
    >>> qstate('udu')
    Quantum object: dims = [[2, 2, 2], [1, 1, 1]], shape = [8, 1], type = ket
    Qobj data =
    [[ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 0.]
     [ 1.]
     [ 0.]
     [ 0.]]

    """
    n = len(string)
    if n != (string.count('u') + string.count('d')):
        raise TypeError('String input to QSTATE must consist ' +
                        'of "u" and "d" elements only')
    else:
        up = basis(2, 1)
        dn = basis(2, 0)
    lst = []
    for k in range(n):
        if string[k] == 'u':
            lst.append(up)
        else:
            lst.append(dn)
    return tensor(lst)


#
# quantum state number helper functions
#
def state_number_enumerate(dims, state=None, idx=0):
    """
    An iterator that enumerate all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.

    Example:

        >>> for state in state_number_enumerate([2,2]):
        >>>     print state
        [ 0.  0.]
        [ 0.  1.]
        [ 1.  0.]
        [ 1.  1.]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        Current state in the iteration. Used internally.

    idx : integer
        Current index in the iteration. Used internally.

    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.

    """

    if state is None:
        state = np.zeros(len(dims))

    if idx == len(dims):
        yield np.array(state)
    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, state, idx + 1):
                yield s


def state_number_index(dims, state):
    """
    Return the index of a quantum state corresponding to state,
    given a system with dimensions given by dims.

    Example:

        >>> state_number_index([2, 2, 2], [1, 1, 0])
        6.0

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        State number array.

    Returns
    -------
    idx : list
        The index of the state given by `state` in standard enumeration
        ordering.

    """
    return sum([state[i] * prod(dims[i + 1:]) for i, d in enumerate(dims)])


def state_index_number(dims, index):
    """
    Return a quantum number representation given a state index, for a system
    of composite structure defined by dims.

    Example:

        >>> state_index_number([2, 2, 2], 6)
        [1, 1, 0]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    index : integer
        The index of the state in standard enumeration ordering.

    Returns
    -------
    state : list
        The state number array corresponding to index `index` in standard
        enumeration ordering.

    """
    state = np.empty_like(dims)

    D = np.concatenate([np.flipud(np.cumprod(np.flipud(dims[1:]))), [1]])

    for n in range(len(dims)):
        state[n] = index / D[n]
        index -= state[n] * D[n]

    return list(state)


def state_number_qobj(dims, state):
    """
    Return a Qobj representation of a quantum state specified by the state
    array `state`.

    Example:

        >>> state_number_qobj([2, 2, 2], [1, 0, 1])
        Quantum object: dims = [[2, 2, 2], [1, 1, 1]], \
shape = [8, 1], type = ket
        Qobj data =
        [[ 0.]
         [ 0.]
         [ 0.]
         [ 0.]
         [ 0.]
         [ 1.]
         [ 0.]
         [ 0.]]

    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.

    state : list
        State number array.

    Returns
    -------
    state : :class:`qutip.Qobj.qobj`
        The state as a :class:`qutip.Qobj.qobj` instance.


    """
    return tensor([fock(dims[i], s) for i, s in enumerate(state)])


def phase_basis(N, m, phi0=0):
    """
    Basis vector for the mth phase of the Pegg-Barnett phase operator.

    Parameters
    ----------
    N : int
        Number of basis vectors in Hilbert space.
    m : int
        Integer corresponding to the mth discrete phase phi_m=phi0+2*pi*m/N
    phi0 : float (default=0)
        Reference phase angle.

    Returns
    -------
    state : qobj
        Ket vector for mth Pegg-Barnett phase operator basis state.

    Notes
    -----
    The Pegg-Barnett basis states form a complete set over the truncated
    Hilbert space.

    """
    phim = phi0 + (2.0 * np.pi * m) / N
    n = np.arange(N).reshape((N, 1))
    data = 1.0 / np.sqrt(N) * np.exp(1.0j * n * phim)
    return Qobj(data)


def zero_ket(N, dims=None):
    """
    Creates the zero ket vector with shape Nx1 and
    dimensions `dims`.

    Parameters
    ----------
    N : int
        Hilbert space dimensionality
    dims : list
        Optional dimensions if ket corresponds to
        a composite Hilbert space.

    Returns
    -------
    zero_ket : qobj
        Zero ket on given Hilbert space.

    """
    return Qobj(sp.csr_matrix((N, 1), dtype=complex), dims=dims)


def spin_state(j, m, type='ket'):
    """Generates the spin state |j, m>, i.e.  the eigenstate
    of the spin-j Sz operator with eigenvalue m.

    Parameters
    ----------
    j : float
        The spin of the state ().

    m : int
        Eigenvalue of the spin-j Sz operator.

    type : string {'ket', 'bra', 'dm'}
        Type of state to generate.

    Returns
    -------
    state : qobj
        Qobj quantum object for spin state

    """
    J = 2 * j + 1
    
    if type == 'ket':
        return basis(int(J), int(j - m))
    elif type == 'bra':
        return basis(int(J), int(j - m)).dag()
    elif type == 'dm':
        return fock_dm(int(J), int(j - m))
    else:
        raise ValueError("invalid value keyword argument 'type'")

