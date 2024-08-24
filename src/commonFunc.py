import numpy as np
def __Compute_RowEnergy(A):
    M = A.shape[0]
    energy = np.zeros(M, dtype=np.double)

    for m in range(M):
        energy[m] = np.linalg.norm(A[m, :])
    return energy

def Kaczmarz(A, b, iterations=10, lambd=0, enforceReal=False, enforcePositive=False, shuffle=False):
    M = A.shape[0]
    N = A.shape[1]
    x = np.zeros(N, dtype=b.dtype)
    residual = np.zeros(M, dtype=x.dtype)

    energy = __Compute_RowEnergy(A)

    rowIndexCycle = np.arange(0, M)

    if shuffle:
        np.random.shuffle(rowIndexCycle)

    lambdIter = lambd

    for l in range(iterations):
        for m in range(M):
            k = rowIndexCycle[m]
            if energy[k] > 0:

                beta = (b[k] - A[k, :].dot(x) - np.sqrt(lambdIter) * residual[k]) / (energy[k] ** 2 + lambd)

                x[:] += beta * A[k, :].conjugate()

                residual[k] += np.sqrt(lambdIter) * beta
        if enforceReal and np.iscomplexobj(x):
            x.imag = 0
        if enforcePositive:
            x = x * (x.real > 0)
    # show1dimData(x)
    return x

