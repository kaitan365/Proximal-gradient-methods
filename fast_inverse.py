import numpy as np
from scipy import linalg



def fast_inverse_scalars(c, slow_checks=True):
    """
    Compute the inverse of the matrix C with diag(C)=I
    and only nonzero entries -c just above the diagonal, as in
    https://math.stackexchange.com/questions/3690293/inverse-matrix-to-a-bi-diagonal-matrix/3700234#3700234

    That is, if c.shape == (T-1, ),
    C = np.eye(T)-np.diag(c, k=1)
    """
    rows = []
    last_row = np.zeros(c.shape[0]+1)
    last_row[-1] = 1.
    rows.append(last_row)
    for t in range(c.shape[0]-1,-1,-1):
        r = rows[-1] * c[t]
        r[t] = 1.
        rows.append(r)
    inverse = np.stack(reversed(rows))

    # not necessary but useful for verifications
    if slow_checks:
        C = np.eye(c.shape[0]+1)-np.diag(c, k=1)
        assert np.isclose(inverse @ C, np.eye(C.shape[0])).all()
    return inverse

def fast_solve_scalars(c, b, slow_checks=True):
    """
    c: (T-1, ) 1-dimensional array 
    b: (T, M)

    returns: same as np.lianalg.solve(C, b) 
                       where          C = np.eye(n)-np.diag(c, k=1)
    """
    assert c.shape == (b.shape[0] - 1, )
    rows = []
    rows.append(b[-1])
    for t in range(c.shape[0]-1,-1,-1):
        rows.append(b[t] + c[t] * rows[-1])
    solution = np.stack(reversed(rows))

    # not necessary but useful for verifications
    if slow_checks:
        C = np.eye(c.shape[0]+1)-np.diag(c, k=1)
        assert np.isclose(solution, np.linalg.solve(C, b)).all()

    return solution

fast_inverse_scalars(np.random.normal(size=(8, )))

fast_solve_scalars(np.random.normal(size=(8, )),
                   np.random.normal(size=(9, 5 )))



def fast_solve_blocks(c, b, slow_checks=True):
    """
    Now, the large matrix is of size pT,pT
    c: shape (T-1, p,p) with T-1 blocks of size (p,p)
    b: shape (T, p, M) or more dimensions to the right, e.g, (T, p, M, K), (T, p, M, K, L) etc.

    The matrix C has shape (pT,pT) with blocks given by c above the diagonal
    and identity blocks in the diagonal.

    """
    T = c.shape[0]+1
    p = c.shape[1]
    assert c.shape == (T-1, p, p)
    assert b.shape[0:2] == (T, p)
    rows = []
    rows.append(b[-1])
    for t in range(c.shape[0]-1, -1, -1):
        rows.append(b[t] + np.tensordot(c[t],rows[-1], axes=(-1,0)))
    solution = np.stack(reversed(rows))

    # not necessary but useful for verifications
    if slow_checks:
        C_off_diag = np.block([
                    [np.zeros((p*(T-1),p)), linalg.block_diag(*c)],
                    [np.zeros((p,p)), np.zeros((p,p*(T-1)))]
                    ])
        C = np.eye(p*T) - C_off_diag
        K = np.prod(b.shape[2:])
        assert np.isclose(solution, np.linalg.solve(C, b.reshape((T*p, K))).reshape(solution.shape)).all()

    return solution

reuslt = fast_solve_blocks(
        np.random.normal(size=(5, 3, 3)),
        np.random.normal(size=(6, 3, 11))
        )

result = fast_solve_blocks(
        np.random.normal(size=(5, 3, 3)),
        np.random.normal(size=(6, 3, 11, 12))
        )
