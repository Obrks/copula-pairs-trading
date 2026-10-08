from scipy import stats, optimize, special
import numpy as np
import pandas as pd
EPS = 1e-10

class Copula:
    """Bivariate one-family copula. h(u, v) = P(U <= u | V = v) = dC/dv. All families here are exchangeable,
    so P(V <= v | U = u) = h(v, u)."""
    name, npar, bounds = "", 1, None

    def __init__(self, theta=None):
        self.theta = theta

    def x0(self, tau): ...
    def _logpdf(self, u, v, th): ...
    def _h(self, u, v, th): ...
    def tail(self): return 0.0, 0.0                                   # (lambda_L, lambda_U)

    def logpdf(self, u, v, th=None):
        u, v = np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)
        return self._logpdf(u, v, self.theta if th is None else th)

    def h(self, u, v, th=None):
        u, v = np.clip(u, EPS, 1 - EPS), np.clip(v, EPS, 1 - EPS)
        return np.clip(self._h(u, v, self.theta if th is None else th), 0.0, 1.0)

    def fit(self, u, v, w=None, x0=None):
        """(Weighted) maximum likelihood. Weights are the EM responsibilities when used inside a mixture."""
        w = np.ones_like(u) if w is None else w
        if x0 is None:
            x0 = self.x0(stats.kendalltau(u, v)[0])
        nll = lambda th: -np.sum(w * self.logpdf(u, v, th))
        if self.npar == 1:
            self.theta = optimize.minimize_scalar(nll, bounds=self.bounds, method="bounded",
                                                  options={"xatol": 1e-6}).x
        else:
            x0 = np.clip(x0, [b[0] for b in self.bounds], [b[1] for b in self.bounds])
            self.theta = optimize.minimize(nll, x0, bounds=self.bounds, method="L-BFGS-B").x
        return self


class Gaussian(Copula):
    name, bounds = "Gaussian", (-0.999, 0.999)
    def x0(self, tau): return np.sin(np.pi * tau / 2)
    def _logpdf(self, u, v, r):
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        return -0.5 * np.log(1 - r**2) - (r**2 * (x**2 + y**2) - 2 * r * x * y) / (2 * (1 - r**2))
    def _h(self, u, v, r):
        x, y = stats.norm.ppf(u), stats.norm.ppf(v)
        return stats.norm.cdf((x - r * y) / np.sqrt(1 - r**2))


class StudentT(Copula):
    name, npar, bounds = "Student-t", 2, [(-0.999, 0.999), (2.01, 60.0)]
    def x0(self, tau): return np.array([np.sin(np.pi * tau / 2), 6.0])
    def _logpdf(self, u, v, th):
        r, nu = th
        x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        q = (x**2 + y**2 - 2 * r * x * y) / (nu * (1 - r**2))
        return (special.gammaln((nu + 2) / 2) + special.gammaln(nu / 2) - 2 * special.gammaln((nu + 1) / 2)
                - 0.5 * np.log(1 - r**2) - (nu + 2) / 2 * np.log1p(q)
                + (nu + 1) / 2 * (np.log1p(x**2 / nu) + np.log1p(y**2 / nu)))
    def _h(self, u, v, th):
        r, nu = th
        x, y = stats.t.ppf(u, nu), stats.t.ppf(v, nu)
        return stats.t.cdf((x - r * y) / np.sqrt((nu + y**2) * (1 - r**2) / (nu + 1)), nu + 1)
    def tail(self):
        r, nu = self.theta
        lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - r) / (1 + r)), nu + 1)
        return lam, lam


class Clayton(Copula):
    name, bounds = "Clayton", (1e-4, 40.0)
    def x0(self, tau): return max(2 * tau / (1 - tau), 0.1)
    @staticmethod
    def _L(u, v, th):                                  # log(u^-th + v^-th - 1), overflow-safe
        a = np.logaddexp(-th * np.log(u), -th * np.log(v))
        return a + np.log1p(-np.exp(-a))
    def _logpdf(self, u, v, th):
        return np.log1p(th) - (1 + th) * (np.log(u) + np.log(v)) - (2 + 1 / th) * self._L(u, v, th)
    def _h(self, u, v, th):
        return np.exp(-(th + 1) * np.log(v) - (1 / th + 1) * self._L(u, v, th))
    def tail(self): return 2 ** (-1 / self.theta), 0.0


class Frank(Copula):
    name, bounds = "Frank", (-40.0, 40.0)
    def x0(self, tau): return 1.0
    def _logpdf(self, u, v, th):
        th = th if abs(th) > 1e-8 else 1e-8
        D = -np.expm1(-th) - (-np.expm1(-th * u)) * (-np.expm1(-th * v))
        return np.log(th * -np.expm1(-th)) - th * (u + v) - 2 * np.log(np.abs(D))
    def _h(self, u, v, th):
        th = th if abs(th) > 1e-8 else 1e-8
        return np.exp(-th * v) * np.expm1(-th * u) / (np.expm1(-th) + np.expm1(-th * u) * np.expm1(-th * v))


class Gumbel(Copula):
    name, bounds = "Gumbel", (1.0001, 30.0)
    def x0(self, tau): return max(1 / (1 - tau), 1.01)
    def _logpdf(self, u, v, th):
        x, y = -np.log(u), -np.log(v)
        s = x**th + y**th; A = s ** (1 / th)
        return (-A - np.log(u) - np.log(v) + (th - 1) * (np.log(x) + np.log(y))
                + (1 / th - 2) * np.log(s) + np.log(A + th - 1))
    def _h(self, u, v, th):
        x, y = -np.log(u), -np.log(v)
        s = x**th + y**th
        return np.exp(-s ** (1 / th)) * s ** (1 / th - 1) * y ** (th - 1) / v
    def tail(self): return 0.0, 2 - 2 ** (1 / self.theta)




class Mixture:
    """C = sum_k pi_k C_k, fitted by EM. Density and h-function are linear in the weights."""

    def __init__(self, name, components):
        self.name, self.comps = name, components
        self.npar = sum(c.npar for c in components) + len(components) - 1
        self.w = np.full(len(components), 1 / len(components))

    def _logterms(self, u, v):
        return np.column_stack([np.log(max(w, 1e-12)) + c.logpdf(u, v) for w, c in zip(self.w, self.comps)])

    def logpdf(self, u, v):
        return special.logsumexp(self._logterms(u, v), axis=1)

    def h(self, u, v):
        return np.clip(sum(w * c.h(u, v) for w, c in zip(self.w, self.comps)), 0.0, 1.0)

    def tail(self):
        return tuple(np.dot(self.w, [c.tail()[i] for c in self.comps]) for i in (0, 1))

    def fit(self, u, v, max_iter=100, tol=1e-3):
        """Two EM starts (equal weights / 90% on the best single family); keeps the better one, so the
        mixture never ends below its own best component."""
        for c in self.comps:                                   # start from each family's own MLE
            c.fit(u, v)
        lls = [c.logpdf(u, v).sum() for c in self.comps]
        K, starts, best = len(self.comps), [], None
        starts.append(np.full(K, 1 / K))
        starts.append(np.where(np.arange(K) == np.argmax(lls), 0.9, 0.1 / (K - 1)))
        init = [c.theta for c in self.comps]
        for w0 in starts:
            for c, th in zip(self.comps, init):
                c.theta = th
            self.w = w0
            ll = self._em(u, v, max_iter, tol)
            if best is None or ll > best[0]:
                best = (ll, self.w.copy(), [c.theta for c in self.comps])
        k = int(np.argmax(lls))
        if lls[k] > best[0]:                                   # degenerate mixture = best single family
            best = (lls[k], np.eye(K)[k], init)
        _, self.w, ths = best
        for c, th in zip(self.comps, ths):
            c.theta = th
        return self

    def _em(self, u, v, max_iter, tol):
        ll_old = -np.inf
        for _ in range(max_iter):
            L = self._logterms(u, v)
            ll = special.logsumexp(L, axis=1).sum()
            if ll - ll_old < tol:                                  # EM is monotone; stop on small gain
                break
            ll_old = ll
            R = np.exp(L - special.logsumexp(L, axis=1, keepdims=True))     # E-step: responsibilities
            self.w = R.mean(axis=0)                                           # M-step: weights ...
            for k, c in enumerate(self.comps):                                # ... and weighted MLE per family
                c.fit(u, v, w=R[:, k], x0=c.theta)
        return self.logpdf(u, v).sum()

    @property
    def theta(self):
        return {c.name: np.round(c.theta, 3) for c in self.comps} | {"weights": np.round(self.w, 3)}


def candidate_copulas():
    return [Gaussian(), StudentT(), Clayton(), Frank(), Gumbel(),
            Mixture("CFG", [Clayton(), Frank(), Gumbel()]),
            Mixture("CtG", [Clayton(), StudentT(), Gumbel()])]


def fit_copulas(u, v, select_by):
    """Fit every candidate; return the selected one plus a comparison table."""
    rows, fitted = [], {}
    for cop in candidate_copulas():
        cop.fit(u, v)
        ll = cop.logpdf(u, v).sum()
        lamL, lamU = cop.tail()
        rows.append({"copula": cop.name, "k": cop.npar, "loglik": ll,
                     "aic": 2 * cop.npar - 2 * ll, "bic": cop.npar * np.log(len(u)) - 2 * ll,
                     "lambda_L": lamL, "lambda_U": lamU})
        fitted[cop.name] = cop
    tab = pd.DataFrame(rows).set_index("copula")
    best = tab["loglik"].idxmax() if select_by == "loglik" else tab[select_by].idxmin()
    return fitted[best], tab