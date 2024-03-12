"""
This script is modified from merlin: https://github.com/midas-tum/merlin
"""

import torch


class DCPM(torch.nn.Module):
    def __init__(self, A, AH, weight_init=1.0, weight_scale=1.0, requires_grad=False, **kwargs):
        super().__init__()

        self.A = A
        self.AH = AH

        self.weight_scale = weight_scale
        self.weight_init = weight_init
        self._weight = torch.tensor(1, dtype=torch.float32) * weight_init
        # self._weight = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32)*weight_init)
        # self._weight.requires_grad_(requires_grad)
        # self._weight.proj = lambda: self._weight.data.clamp_(1e-4, 1000)

        max_iter = kwargs.get('max_iter', 10)
        tol = kwargs.get('tol', 1e-10)
        self.prox = CGClass(A, AH, max_iter=max_iter, tol=tol)

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def forward(self, inputs, scale=1.0):
        x = inputs[0]
        y = inputs[1]
        constants = inputs[2:]
        lambdaa = 1.0 / torch.max(self.weight * scale, torch.ones_like(self.weight)*1e-9)
        return self.prox(lambdaa, x, y, *constants)

    def __repr__(self):
        return f'DCPD(lambda_init={self.weight_init:.4g}, weight_scale={self.weight_scale}, requires_grad={self._weight.requires_grad})'


class CGClass(torch.nn.Module):
    def __init__(self, A, AH, max_iter=10, tol=1e-10):
        super().__init__()
        self.A = A
        self.AH = AH
        self.max_iter = max_iter
        self.tol = tol

        self.cg = ComplexCG()

    def forward(self, lambdaa, x, y, *constants):
        out = torch.zeros_like(x)

        for n in range(x.shape[0]):
            cg_out = self.cg(self.A, self.AH, self.max_iter, self.tol, lambdaa, x[n::1], y[n::1],
                                   *[c[n::1] for c in constants])
            out[n] = cg_out[0]
        return out


class ComplexCG(torch.nn.Module):

    def dotp(self, data1, data2):
        if data1.is_complex():
            mult = torch.conj(data1) * data2
        else:
            mult = data1 * data2
        return torch.sum(mult)

    def solve(self, x0, M, tol, max_iter):
        x = torch.zeros_like(x0)
        r = x0.clone()
        p = x0.clone()

        rTr = torch.norm(r).pow(2)

        it = 0
        while rTr > tol and it < max_iter:
            it += 1
            q = M(p)
            alpha = rTr / self.dotp(p, q)
            x = x + alpha * p
            r = r - alpha * q

            rTrNew = torch.norm(r).pow(2)

            beta = rTrNew / rTr

            p = r.clone() + beta * p
            rTr = rTrNew.clone()
        return x

    def forward(self, A, AH, max_iter, tol, lambdaa, x, y, *constants):
        def M(p):
            return AH(A(p, *constants), *constants) + lambdaa * p

        rhs = AH(y, *constants) + lambdaa * x
        return self.solve(rhs, M, tol, max_iter)