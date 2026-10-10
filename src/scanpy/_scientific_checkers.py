"""Opt-in scientific runtime checks used by the SciBench pilot.

The module is inert unless ``SCIBENCH_TRIGGER_LOG`` names a log file. Checks
append stable IDs and never raise into scanpy, never change a return value and
never touch global random state. Re-calling checkers work on copies and run with
checking suppressed.

Tolerance convention (SANITIZER.md 5.8): ``_C * eps(dtype) * size * scale``
with each factor justified at the call site.
"""

from __future__ import annotations

import copy as _copy
import functools
import json
import math
import os
import warnings
from contextlib import contextmanager

import numpy as np
from scipy import sparse

_ACTIVE = False
_EPS = float(np.finfo(np.float64).eps)
_C = 64.0
_MAX_ELEMS = 20_000_000  # skip densifying checks above this many entries
_ENV = "SCIBENCH_TRIGGER_LOG"


def enabled():
    return bool(os.environ.get(_ENV)) and not _ACTIVE


def trigger(checker_id):
    path = os.environ.get(_ENV)
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"checker_id": checker_id}) + "\n")
    except Exception:
        _swallowed()


def trigger_if(condition, checker_id):
    _reach(checker_id)
    if bool(condition):
        trigger(checker_id)


SWALLOWED = []
REACHED = {}


def _reach(checker_id):
    """Curator-only: count predicate evaluations (observation reachability)."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        REACHED[checker_id] = REACHED.get(checker_id, 0) + 1


def _swallowed():
    """Curator-only: record exceptions swallowed inside a checker."""
    if os.environ.get("SCIBENCH_CHECKER_DEBUG"):
        import traceback

        SWALLOWED.append(traceback.format_exc())


def _guarded(fn):
    """Run ``fn`` once, never nested, never raising into scanpy."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        global _ACTIVE
        if _ACTIVE or not os.environ.get(_ENV):
            return None
        _ACTIVE = True
        try:
            # the host test suite may turn warnings into errors; a checker must never be aborted by one
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return fn(*args, **kwargs)
        except Exception:
            _swallowed()
            return None
        finally:
            _ACTIVE = False

    return wrapper


@contextmanager
def _quiet():
    """Silence scanpy logging and warnings during a re-call."""
    from ._settings import Verbosity, settings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with settings.override(verbosity=Verbosity.error):
            yield


# ----------------------------------------------------------------- helpers
def _is_sparse(x):
    return sparse.issparse(x)


def _dtype_kind(x):
    try:
        return np.dtype(getattr(x, "dtype", np.float64)).kind
    except Exception:
        return "f"


def _eps_of(x):
    try:
        dt = np.dtype(getattr(x, "dtype", np.float64))
        if dt.kind == "f":
            return float(np.finfo(dt).eps)
    except Exception:
        _swallowed()
    return _EPS


def _size(x):
    return int(np.prod(x.shape))


def _dense64(x):
    if _is_sparse(x):
        return np.asarray(x.toarray(), dtype=np.float64)
    return np.asarray(x, dtype=np.float64)


def _all_finite(x):
    if _is_sparse(x):
        return bool(np.all(np.isfinite(x.data)))
    return bool(np.all(np.isfinite(np.asarray(x))))


def _min_value(x):
    if _is_sparse(x):
        return float(x.data.min()) if x.data.size else 0.0
    a = np.asarray(x)
    return float(a.min()) if a.size else 0.0


def _max_abs(x):
    if _is_sparse(x):
        return float(np.abs(x.data).max()) if x.data.size else 0.0
    a = np.asarray(x)
    return float(np.abs(a).max()) if a.size else 0.0


def _row_sums(x):
    if _is_sparse(x):
        return np.asarray(x.astype(np.float64).sum(axis=1)).ravel()
    return np.asarray(x, dtype=np.float64).sum(axis=1)


def _has_duplicates(x):
    """True for a CSR/CSC matrix that stores the same (row, column) position more than once.

    Such a matrix is not a well-formed array: scipy sums the duplicates, scanpy kernels read the
    stored entries one by one, so the two disagree by construction.
    """
    try:
        if _is_sparse(x) and x.format in {"csr", "csc"} and not x.has_canonical_format:
            y = x.copy()
            y.sum_duplicates()
            return y.nnz < x.nnz
    except Exception:
        _swallowed()
    return False


def _range_limit(dtype):
    """Largest magnitude a sum may reach before dtype overflow, with a factor-2 headroom."""
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return 0.5 * float(np.finfo(dt).max)
    if dt.kind in "iu":
        return 0.5 * float(np.iinfo(dt).max)
    return 0.5 * float(np.finfo(np.float64).max)


_TINY_N = float(np.finfo(np.float64).tiny) / _EPS  # below this, squares/variances lose precision (subnormal range)


def _tiny_n(dtype):
    """finfo(dtype).tiny/eps for float dtypes (float32: 9.9e-32), else the float64 value."""
    dt = np.dtype(dtype)
    if dt.kind == "f":
        fi = np.finfo(dt)
        return float(fi.tiny) / float(fi.eps)
    return _TINY_N


def _as_f64(x):
    """float64 copy for reference statistics (integer/float32 accumulators wrap or underflow)."""
    return x if x.dtype == np.float64 else x.astype(np.float64)


def _row_nnz_max(x):
    """Max stored entries per row of the CSR view: recursive (sequential) sums of k terms err by (k-1)u."""
    try:
        c = x.tocsr() if _is_sparse(x) else sparse.csr_matrix(np.asarray(x))
        return int(np.max(np.diff(c.indptr))) if c.shape[0] else 0
    except Exception:
        _swallowed()
        return int(x.shape[1])


def _log2(n):
    return math.log2(max(int(n), 1) + 2)


def snap_array(x):
    """Copy of a dense or sparse (non-dask, non-backed) matrix, else ``None``."""
    try:
        if _is_sparse(x):
            return x.copy()
        if isinstance(x, np.ndarray) and x.dtype.kind in "fiu":
            return x.copy()
    except Exception:
        _swallowed()
    return None


def to_other_representation(x):
    if _is_sparse(x):
        return np.asarray(x.toarray())
    return sparse.csr_matrix(x)


def _violates(diff, tol):
    """True when ``diff`` is not provably within ``tol`` (NaN alarms)."""
    return not (diff <= tol)


# ================================================== A. normalisation
def _highly_expressed(x, max_fraction, tot):
    """Gene masks (certainly high, ambiguous) for the documented definition.

    An entry within 1e-5 (relative) of ``max_fraction * total`` is ambiguous: float32 vs float64
    evaluation of the threshold may decide it either way.
    """
    n_vars = x.shape[1]
    high = np.zeros(n_vars, dtype=bool)
    amb = np.zeros(n_vars, dtype=bool)
    if _is_sparse(x):
        c = x.tocoo()
        thr = max_fraction * tot[c.row]
        vals = c.data.astype(np.float64)
        near = np.abs(vals - thr) <= 1e-5 * np.maximum(thr, 1e-300)
        high[c.col[(vals > thr) & ~near]] = True
        amb[c.col[near]] = True
    else:
        a = np.asarray(x, dtype=np.float64)
        thr = max_fraction * tot[:, None]
        near = np.abs(a - thr) <= 1e-5 * np.maximum(thr, 1e-300)
        high = np.any((a > thr) & ~near, axis=0)
        amb = np.any(near, axis=0)
    return high, amb & ~high


@_guarded
def check_normalize_total(snap, out, factors, *, target_sum, exclude, max_fraction):
    """SC-NRM-001 / 002 / 003 / 004."""
    if snap is None or not (isinstance(out, np.ndarray) or _is_sparse(out)):
        return
    if snap.dtype.kind != "f" or not _all_finite(snap) or _min_value(snap) < 0:
        return
    if not _all_finite(out):  # non-finite output (e.g. no cell has a usable total): not judged here
        return
    n_obs, n_vars = snap.shape
    eps = _eps_of(out)
    tot = _row_sums(snap)
    nonzero = tot > 0
    if not nonzero.any():
        return
    if _has_duplicates(snap):  # not a well-formed matrix: stored entries and their scipy sum differ
        return
    log_n = _log2(n_vars)
    seq = 2.0 * _row_nnz_max(snap)  # _normalize_csr accumulates each row sequentially: error <= (k-1)u
    # size factors (usable total / target) must be normal floats of the output dtype; a subnormal or
    # zero factor makes the division itself lose the scale (the documented law is undefined there)
    _high0, _ = _highly_expressed(snap, max_fraction, tot) if exclude else (np.zeros(n_vars, dtype=bool), None)
    _use = _row_sums(snap[:, ~_high0]) if exclude and (~_high0).any() else tot
    _pos = _use[_use > 0]
    if _pos.size:
        _tgt = float(target_sum) if target_sum is not None else float(np.median(_pos))
        _fi = np.finfo(out.dtype) if np.dtype(out.dtype).kind == "f" else np.finfo(np.float64)
        _fac = _pos / _tgt if _tgt > 0 else _pos
        if _tgt <= 0 or float(_fac.min()) < float(_fi.tiny) / _EPS or float(_fac.max()) > float(_fi.max) * _EPS:
            return
    if not exclude:
        target = float(target_sum) if target_sum is not None else float(np.median(tot[nonzero]))
        sums = _row_sums(out)[nonzero]
        tol = (_C * log_n + seq) * eps * target
        trigger_if(_violates(float(np.max(np.abs(sums - target))), tol), "SC-NRM-001")
    else:
        high, amb = _highly_expressed(snap, max_fraction, tot)
        certain = ~high & ~amb
        if certain.any():
            lo = _row_sums(out[:, certain])
            hi = lo + (_row_sums(out[:, amb]) if amb.any() else 0.0)
            pos = _row_sums(snap[:, certain]) > 0
            if pos.any():
                ref = float(target_sum) if target_sum is not None else float(np.median(lo[pos]))
                tol = (_C * log_n + seq) * eps * ref
                if target_sum is not None:
                    bad = np.any((lo[pos] > ref + tol) | (hi[pos] < ref - tol))
                else:  # no stated target: the cells' [lo, hi] intervals must share a common value
                    bad = float(np.max(lo[pos])) > float(np.min(hi[pos])) + tol
                trigger_if(bool(bad), "SC-NRM-002")
    # re-calls on transformed / re-represented copies
    from .preprocessing._normalization import _normalize_total_helper

    def run(x):
        with _quiet():
            res, fac, _ = _normalize_total_helper(
                x, exclude_highly_expressed=exclude, max_fraction=max_fraction, target_sum=target_sum
            )
        return res, np.asarray(fac, dtype=np.float64)

    out64 = _dense64(out) if _size(out) <= _MAX_ELEMS else None
    scale = _max_abs(out)
    fac_ref = np.asarray(factors, dtype=np.float64)
    # cells whose size-factor total is zero are left un-normalised by design: not in the law's domain
    if exclude:
        high_m, amb_m = _highly_expressed(snap, max_fraction, tot)
        border = bool(amb_m.any())
        norm_rows = _row_sums(snap[:, ~high_m]) > 0 if (~high_m).any() else np.zeros(n_obs, dtype=bool)
    else:
        border, norm_rows = False, nonzero
    if out64 is None or not norm_rows.any():
        return
    if target_sum is not None:  # scaling by 2 is exact, so threshold decisions cannot differ
        scaled = snap * 2.0 if _is_sparse(snap) else snap * np.asarray(2.0, dtype=snap.dtype)
        if _all_finite(scaled):
            res2, fac2 = run(scaled)
            # 2X may overflow (or underflow) the dtype although X is fine: such a leg is outside the law's domain
            if _all_finite(res2) and np.all(np.isfinite(fac2)) and np.all(np.isfinite(fac_ref)):
                d = float(np.max(np.abs(_dense64(res2)[norm_rows] - out64[norm_rows])))
                # size factors are proportional to depth, so they must double exactly
                f = float(np.max(np.abs(fac2[norm_rows] - 2.0 * fac_ref[norm_rows])))
                trigger_if(
                    _violates(d, _C * eps * max(scale, 1e-300))
                    or _violates(f, _C * eps * max(float(np.max(np.abs(2.0 * fac_ref[norm_rows]))), 1e-300)),
                    "SC-NRM-003",
                )
    if border:  # representation legs may decide ambiguous thresholds differently
        return
    alt = to_other_representation(snap)
    if alt.dtype != snap.dtype:
        alt = alt.astype(snap.dtype)
    res3, fac3 = run(alt)
    d = float(np.max(np.abs(_dense64(res3)[norm_rows] - out64[norm_rows])))
    f = float(np.max(np.abs(fac3[norm_rows] - fac_ref[norm_rows])))
    tol = (_C * log_n + seq) * eps
    trigger_if(
        _violates(d, tol * max(scale, 1e-300))
        or _violates(f, tol * max(float(np.max(np.abs(fac_ref[norm_rows]))), 1e-300)),
        "SC-NRM-004",
    )


# ================================================== B. scaling
@_guarded
def check_scale(snap, out, *, zero_center, max_value, mask):
    """SC-SCL-001 / 002 / 003 (``out`` is the scaled matrix)."""
    if snap is None or not (isinstance(out, np.ndarray) or _is_sparse(out)):
        return
    if _size(snap) > _MAX_ELEMS or not _all_finite(snap) or not _all_finite(out):
        return
    if _has_duplicates(snap):
        return
    zero_center = bool(zero_center)
    n_obs = snap.shape[0]
    rows = np.ones(n_obs, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    n = int(rows.sum())
    if n < 2:
        return
    a_in = _dense64(snap)[rows]
    a_out = _dense64(out)[rows]
    eps = max(_eps_of(out), _eps_of(snap))  # statistics are accumulated in the input dtype
    mean_in = a_in.mean(axis=0)
    std_in = a_in.std(axis=0, ddof=1)
    # a sum of squares past the float64 range makes the production std inf (outputs 0): outside the law
    ok = (std_in > 0) & np.isfinite(std_in) & (np.max(np.abs(a_in), axis=0) ** 2 * n < _range_limit(np.float64))
    # statistics are accumulated in the input dtype: squares must stay in that dtype's normal range
    _dt_in = snap.dtype if snap.dtype.kind == "f" else np.float64
    ok &= (std_in**2 >= _tiny_n(_dt_in)) & (np.max(np.abs(a_in), axis=0) ** 2 * n < _range_limit(_dt_in))
    # SC-SCL-001 (law needs unclipped output)
    if max_value is None and ok.any():
        # the variance comes from a one-pass sum-of-squares mean_var: error ~ eps*(1 + mean^2/var)
        kappa = (1.0 + (mean_in[ok] / std_in[ok]) ** 2) * _log2(n)
        tol = _C * eps * kappa
        bad = np.max(np.abs(a_out[:, ok].var(axis=0, ddof=1) - 1.0) - 2 * tol)
        if zero_center:
            bad = max(bad, np.max(np.abs(a_out[:, ok].mean(axis=0)) - tol))
        trigger_if(_violates(float(bad), 0.0), "SC-SCL-001")
    from .preprocessing._scale import scale_array

    def run(x, mv, m):
        with _quiet():
            return scale_array(x, zero_center=zero_center, max_value=mv, copy=False, mask=m)

    # SC-SCL-002: clipping is truncation
    if max_value is not None and float(max_value) > 0:  # a non-positive clip bound is outside the law's domain
        un = run(snap.copy(), None, None if mask is None else np.asarray(mask, dtype=bool))
        u = _dense64(un)[rows]
        cols = np.isfinite(u).all(axis=0) & np.isfinite(a_out).all(axis=0)  # constant genes give 0/0
        u, a_chk = u[:, cols], a_out[:, cols]
        exp = np.minimum(u, float(max_value))
        if zero_center:
            exp = np.maximum(exp, -float(max_value))
        if u.size:
            d = float(np.max(np.abs(exp - a_chk)))
            trigger_if(_violates(d, _C * eps * max(1.0, float(np.max(np.abs(u))))), "SC-SCL-002")
    # SC-SCL-003: mask locality
    if mask is not None and not rows.all():
        full_in = _dense64(snap)
        full_out = _dense64(out)
        outside = full_in[~rows]
        changed = not np.array_equal(full_out[~rows], outside, equal_nan=True)
        sub = snap[rows]  # same container as the call under test (csc and csr accumulate differently)
        s = _dense64(run(sub.copy(), max_value, None))
        # a numerically constant gene has a rounding-noise std, so its scaled values are ill-conditioned
        well = std_in > 1e-6 * np.abs(mean_in)
        d = float(np.max(np.abs(s[:, well] - a_out[:, well]))) if s.size and well.any() else 0.0
        # the one-pass variance of either computation carries error ~ eps * (1 + mean^2/var)
        kappa = float(np.max((1.0 + (mean_in[well] / std_in[well]) ** 2) * _log2(n))) if well.any() else 1.0
        trigger_if(changed or _violates(d, _C * eps * kappa * max(1.0, float(np.max(np.abs(s[:, well]))) if s.size and well.any() else 1.0)), "SC-SCL-003")


# ================================================== C. downsampling
@_guarded
def check_downsample(snap, out, *, total_counts, counts_per_cell, replace):
    """SC-DSM-001 / 002."""
    if snap is None or not (isinstance(out, np.ndarray) or _is_sparse(out)):
        return
    if _size(snap) > _MAX_ELEMS or not _all_finite(snap) or _min_value(snap) < 0:
        return
    a = _dense64(snap)
    if not np.array_equal(a, np.round(a)):
        return
    b = _dense64(out)
    if total_counts is not None:
        exp = min(float(a.sum()), float(int(total_counts)))
        bad_total = float(b.sum()) != exp
        if bad_total and snap.dtype.kind == "f":  # library compares its total in the input dtype
            ulp = float(np.spacing(np.asarray(a.sum(), dtype=snap.dtype)))
            # target within one input-dtype ulp of the true total (either side): undecidable in that dtype
            bad_total = not (abs(a.sum() - int(total_counts)) <= ulp
                             and float(b.sum()) in (float(a.sum()), float(int(total_counts))))
    else:
        n_obs = a.shape[0]
        t = np.full(n_obs, counts_per_cell) if isinstance(counts_per_cell, (int, np.integer)) else np.asarray(counts_per_cell)
        exp = np.minimum(a.sum(axis=1), t.astype(np.float64))
        rs, bs = a.sum(axis=1), b.sum(axis=1)
        wrong = bs != exp
        if snap.dtype.kind == "f":  # a total within one input-dtype ulp of the target cannot be told apart
            ulp = np.spacing(rs.astype(snap.dtype)).astype(np.float64)
            wrong &= ~((np.abs(rs - t) <= ulp) & ((bs == rs) | (bs == t)))
        bad_total = bool(np.any(wrong))
    top = float(np.max(exp)) if np.ndim(exp) else float(exp)
    if snap.dtype.kind in "iu" and replace:
        # with replacement one gene may receive every read of the cell: a target above the dtype
        # maximum wraps around (compact integer dtype), which is outside both invariants' domain
        if top > float(np.iinfo(snap.dtype).max):
            return
    if snap.dtype.kind == "f":
        # a single entry that may need to hold more reads than the dtype's consecutive-integer limit
        # (float32: 2**24) cannot be counted read by read: outside the law's domain
        lim = 2.0 ** (np.finfo(snap.dtype).nmant + 1)
        # without replacement an entry is bounded by its input; with replacement by what it received
        if (float(b.max(initial=0.0)) >= lim) if replace else (float(a.max(initial=0.0)) > lim):
            return
    trigger_if(bad_total, "SC-DSM-001")
    bad = (not np.array_equal(b, np.round(b))) or bool(np.any(b < 0)) or bool(np.any((a == 0) & (b != 0)))
    if not replace:
        bad = bad or bool(np.any(b > a))
    trigger_if(bad, "SC-DSM-002")


# ================================================== D. regress_out
@_guarded
def check_regress_out(snap, res, regressors, categorical):
    """SC-RGO-001 (continuous covariates only)."""
    if categorical or snap is None or not isinstance(res, np.ndarray):
        return
    if _size(snap) > _MAX_ELEMS:
        return
    try:
        d = np.asarray(regressors, dtype=np.float64)
    except (TypeError, ValueError):
        return
    y = _dense64(snap)
    r = np.asarray(res, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != y.shape[0] or r.shape != y.shape:
        return
    if not (np.all(np.isfinite(d)) and np.all(np.isfinite(y)) and np.all(np.isfinite(r))):
        return
    if np.linalg.matrix_rank(d) < d.shape[1]:
        return
    cond = float(np.linalg.cond(d))
    if not cond <= 1e6:
        return
    nonconst = np.any(y != y[0], axis=0) & (np.max(np.abs(y), axis=0) >= _TINY_N)  # subnormal-range input
    if not nonconst.any():
        return
    eps = max(_eps_of(res), _eps_of(snap))
    dn = np.linalg.norm(d, axis=0)
    # rescale each gene by its max |y| (the law is scale-free) so that norms cannot under/overflow
    sc_ = np.max(np.abs(y[:, nonconst]), axis=0)
    yn = np.linalg.norm(y[:, nonconst] / sc_, axis=0)
    inner = np.abs(d.T @ (r[:, nonconst] / sc_)) / (dn[:, None] * yn[None, :])
    trigger_if(_violates(float(np.max(inner)), _C * eps * cond**2), "SC-RGO-001")


# ================================================== E. highly variable genes
@_guarded
def check_hvg_binned(df, flavor):
    """SC-HVG-001 (seurat) and SC-HVG-002 (cell_ranger)."""
    import pandas as pd

    d = df["dispersions"].to_numpy(dtype=np.float64)
    z = df["dispersions_norm"].to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(df["mean_bin"].astype(object))
    bad = False
    from statsmodels.robust import mad

    for c in np.unique(codes[codes >= 0]):
        idx = codes == c
        if np.any(np.isinf(d[idx])):  # an overflowed dispersion makes the bin's mean/std (and every z) undefined
            continue
        fin = idx & np.isfinite(d) & np.isfinite(z)
        m = int(fin.sum())
        if m < 2:
            continue
        dv, zv = d[fin], z[fin]
        if flavor == "seurat":
            avg, dev = float(dv.mean()), float(dv.std(ddof=1))
            if not dev > 0:
                continue
            tol = _C * _EPS * (1.0 + abs(avg) / dev) * _log2(m)
            bad |= _violates(abs(float(zv.mean())), tol) or _violates(abs(float(zv.std(ddof=1)) - 1.0), tol)
        elif flavor == "cell_ranger":
            avg, dev = float(np.median(dv)), float(mad(dv))
            if not dev > 0:
                continue
            tol = _C * _EPS * (1.0 + abs(avg) / dev) * _log2(m)
            bad |= _violates(abs(float(np.median(zv))), tol) or _violates(abs(float(mad(zv)) - 1.0), tol)
    if flavor in {"seurat", "cell_ranger"}:
        trigger_if(bad, "SC-HVG-001" if flavor == "seurat" else "SC-HVG-002")


def _lex_violation(selected, key_cols):
    """True if some unselected row outranks a selected row lexicographically."""
    sel = [tuple(k[i] for k in key_cols) for i in np.flatnonzero(selected)]
    uns = [tuple(k[i] for k in key_cols) for i in np.flatnonzero(~selected)]
    if not sel or not uns:
        return False
    return max(uns) > min(sel)


@_guarded
def check_hvg_selection(adata, df, *, batch_key, layer, cutoff, n_bins, flavor, filter_unexpressed_genes):
    """SC-HVG-003 (single batch) and SC-HVG-004 (top-n rank consistency)."""
    from .get import _get_arr

    hv = df["highly_variable"].to_numpy(dtype=bool)
    norm = df["dispersions_norm"].to_numpy(dtype=np.float64)
    means = df["means"].to_numpy(dtype=np.float64)
    expressed = means != 0
    fin = ~np.isnan(norm)  # only NaN ranks last; +-inf (zero bin spread) are ordinary extreme scores
    if isinstance(cutoff, (int, np.integer)) and not isinstance(cutoff, bool):
        n = min(int(cutoff), int(expressed.sum()))
        use = expressed
        key = np.where(fin, norm, -np.inf)
        cols = [key[use]]
        if batch_key:
            cols = [df["highly_variable_nbatches"].to_numpy(dtype=np.float64)[use], key[use]]
        n_finite = int((fin & use).sum())
        short = int(hv[use].sum()) < min(n, n_finite)
        trigger_if(_lex_violation(hv[use], cols) or short, "SC-HVG-004")
    # SC-HVG-003
    if batch_key and flavor in {"seurat", "cell_ranger"}:
        obs = adata.obs[batch_key]
        populated = obs.astype("category").cat.remove_unused_categories().cat.categories
        if len(populated) != 1:
            return
        x = _get_arr(adata, layer=layer)
        if not (isinstance(x, np.ndarray) or _is_sparse(x)):
            return
        nnz = x.getnnz(axis=0) if _is_sparse(x) else np.count_nonzero(x, axis=0)
        if not np.all(np.asarray(nnz) > 0):
            return
        from .preprocessing._highly_variable_genes import _highly_variable_genes_single_batch

        with _quiet():
            df2 = _highly_variable_genes_single_batch(
                adata, layer=layer, cutoff=cutoff, n_bins=n_bins, flavor=flavor, filter_unexpressed_genes=False
            )
        df2 = df2.reindex(df.index)  # the batched path may return genes in a different order
        bad = False
        # (disp - avg)/dev amplifies the rounding of disp and of the bin statistics by (|disp|+|avg|)/dev:
        # genes whose dispersions agree to ~1e-8 within a bin have a normalised score set by rounding
        from .preprocessing._highly_variable_genes import _mad
        _d2 = df2["dispersions"].to_numpy(dtype=np.float64)
        _grp = df2.groupby("mean_bin", observed=True)["dispersions"]
        if flavor == "cell_ranger":
            _avg, _dev = _grp.transform("median"), _grp.transform(_mad)
        else:
            _avg, _dev = _grp.transform("mean"), _grp.transform("std")
        _avg = np.abs(_avg.to_numpy(dtype=np.float64))
        _dev = np.abs(_dev.to_numpy(dtype=np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            _cond = np.where(_dev > 0, (np.abs(_d2) + _avg) / _dev, 0.0)
        _cond = np.nan_to_num(_cond, nan=0.0, posinf=0.0)
        for col in ("means", "dispersions", "dispersions_norm"):
            a = df[col].to_numpy(dtype=np.float64)
            b = df2[col].to_numpy(dtype=np.float64)
            if col == "dispersions_norm":  # the batched path fills NaN with 0 (as Seurat does)
                a, b = np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)
            same_nan = np.array_equal(np.isnan(a), np.isnan(b))
            f = np.isfinite(a) & np.isfinite(b)
            amp = (_cond[f] * (2.0 + np.abs(a[f]))) if col == "dispersions_norm" else 0.0
            diff = np.abs(a[f] - b[f]) - _C * _EPS * (1.0 + np.abs(a[f]) + amp) * _log2(len(a))
            bad |= (not same_nan) or bool(np.any(diff > 0))
        n2 = df2["dispersions_norm"].to_numpy(dtype=np.float64)
        f2 = n2[np.isfinite(n2)]
        # asking for more genes than have a (non-NaN) score is undefined: the batched path pads the
        # selection with unscorable genes, the unbatched path returns only the scored ones
        over_asked = (
            isinstance(cutoff, (int, np.integer))
            and not isinstance(cutoff, bool)
            and int(cutoff) > int((~np.isnan(n2)).sum())
        )
        near_tie = False
        if isinstance(cutoff, (int, np.integer)) and not isinstance(cutoff, bool) and 0 < int(cutoff) < f2.size:
            fin2 = np.isfinite(n2)
            tol_n = _C * _EPS * (1.0 + np.abs(n2[fin2]) + _cond[fin2] * (2.0 + np.abs(n2[fin2]))) * _log2(len(n2))
            order = np.argsort(-n2[fin2])
            k = int(cutoff)
            near_tie = bool(n2[fin2][order[k - 1]] - n2[fin2][order[k]] <= tol_n[order[k - 1]] + tol_n[order[k]])
        if len(np.unique(f2)) == len(f2) and not over_asked and not near_tie:  # no (near-)ties: selection must match too
            bad |= not np.array_equal(hv, df2["highly_variable"].to_numpy(dtype=bool))
        trigger_if(bad, "SC-HVG-003")


@_guarded
def check_hvg_v3_selection(df, flavor, n_top_genes):
    """SC-HVG-005."""
    hv = df["highly_variable"].to_numpy(dtype=bool)
    rank = df["highly_variable_rank"].to_numpy(dtype=np.float64)
    nb = df["highly_variable_nbatches"].to_numpy(dtype=np.float64)
    rank_k = np.where(np.isnan(rank), -np.inf, -rank)  # higher is better
    cols = [rank_k, nb] if flavor == "seurat_v3" else [nb, rank_k]
    n = min(int(n_top_genes), len(hv))
    trigger_if(_lex_violation(hv, cols) or int(hv.sum()) != n, "SC-HVG-005")


# ================================================== F. QC proportions
@_guarded
def check_top_segment(mtx, ns, out):
    """SC-QC-001 / 002 / 003."""
    if not (isinstance(mtx, np.ndarray) or _is_sparse(mtx)) or not isinstance(out, np.ndarray):
        return
    if _size(mtx) > _MAX_ELEMS or not _all_finite(mtx) or _min_value(mtx) < 0:
        return
    n_obs, n_vars = mtx.shape
    ns_s = np.sort(np.asarray(ns, dtype=np.int64))
    rs = _row_sums(mtx)
    rows = rs > 0
    if not rows.any() or out.shape != (n_obs, len(ns_s)):
        return
    if _has_duplicates(mtx) or float(rs.max()) > _range_limit(mtx.dtype):  # total not representable
        return
    eps = _eps_of(mtx)
    tol = _C * eps * (math.sqrt(n_vars) + _log2(n_vars)) + 4.0 * eps * n_vars  # sequential sums of <= n_vars terms
    o = out[rows].astype(np.float64)
    bad = bool(np.any(np.diff(o, axis=1) < -tol)) or bool(np.any(~(o <= 1.0 + tol)))
    if ns_s[-1] == n_vars:
        bad |= bool(np.any(np.abs(o[:, -1] - 1.0) > tol))
    trigger_if(bad or not np.all(np.isfinite(o)), "SC-QC-001")
    from .preprocessing import _qc

    alt = to_other_representation(mtx)
    with _quiet():
        o2 = np.asarray(_qc.top_segment_proportions(alt, list(ns_s)), dtype=np.float64)
    trigger_if(_violates(float(np.max(np.abs(o2[rows] - o))), tol), "SC-QC-002")
    with _quiet():
        tp = np.asarray(_qc.top_proportions(mtx.astype(np.float64) if mtx.dtype.kind in "iub" else mtx, int(ns_s[-1])), dtype=np.float64)  # -mtx wraps for unsigned
    ref = tp[:, ns_s - 1][rows]
    trigger_if(_violates(float(np.max(np.abs(ref - o))), tol), "SC-QC-003")


# ================================================== G. autocorrelation
def _feature_rows(vals, k=4):
    n_feat = 1 if vals.ndim == 1 else vals.shape[0]
    pick = np.unique(np.linspace(0, n_feat - 1, num=min(k, n_feat)).astype(int))
    return pick


@_guarded
def check_autocorr(metric, result):
    """SC-AUT-001 / 002 / 003 (``metric`` is the `_SparseMetric` instance)."""
    graph, vals = metric.graph, metric._vals
    if not (isinstance(vals, np.ndarray) or _is_sparse(vals)):
        return
    if not (_all_finite(graph) and _min_value(graph) >= 0 and _all_finite(vals)):
        return
    if _has_duplicates(graph) or _has_duplicates(vals):
        return
    n = graph.shape[0]
    if n < 3 or n * (vals.shape[0] if vals.ndim == 2 else 1) > _MAX_ELEMS:
        return
    res = np.atleast_1d(np.asarray(result, dtype=np.float64))
    pick = _feature_rows(vals)
    rows = []
    for i in pick:
        v = vals if vals.ndim == 1 else vals[int(i)]
        v = np.asarray(v.toarray()).ravel() if _is_sparse(v) else np.asarray(v).ravel()
        _v = v.astype(np.float64)
        if _v.std() > 0 and float(np.sum((_v - _v.mean()) ** 2)) >= _TINY_N and np.isfinite(res[int(i) if vals.ndim == 2 else 0]):
            rows.append((int(i), v))
    if not rows:
        return
    cls = type(metric)
    kind = "sparse" if _is_sparse(vals) else ("vec" if vals.ndim == 1 else "dense")

    def box(v):  # same container kind as production
        if kind == "vec":
            return v
        return sparse.csr_matrix(v[None, :]) if kind == "sparse" else v[None, :]

    perm = np.random.default_rng(12345).permutation(n)
    gp = graph.tocsr()[perm][:, perm]
    bad1 = bad2 = bad3 = False
    for i, v in rows:
        r0 = float(res[i if vals.ndim == 2 else 0])
        v64 = v.astype(np.float64)
        mean, std = float(v64.mean()), float(v64.std())
        eps_v = _eps_of(v) if kind == "vec" else _EPS
        # dense kernels lose eps*|mean|/std; the sparse expanded sum of squares loses eps*mean^2/var
        kappa = n * (1.0 + (mean / std) ** 2) if kind == "sparse" else 1.0 + abs(mean) / std
        # SC-AUT-001: relabelling, same container as production
        with _quiet():
            r1 = float(np.atleast_1d(cls(gp, box(v[perm]))())[0])
        lg = _log2(n) + 2.0 * n / _C  # numba kernels: x.mean() and sum(z*z) are sequential n-term sums
        bad1 |= _violates(abs(r1 - r0), _C * max(eps_v, _EPS) * kappa * lg * (1.0 + abs(r0)))
        # SC-AUT-002: affine invariance against a float64 baseline (dense vector kernels)
        a, b = -4.0, 2.0 ** math.ceil(math.log2(float(np.max(np.abs(v64))) + 1e-300))
        with _quiet():
            base = float(np.atleast_1d(cls(graph, v64)())[0])
            aff = float(np.atleast_1d(cls(graph, a * v64 + b)())[0])
        kap2 = 1.0 + (float(np.max(np.abs(v64))) + b) / std
        # worst-case numerator magnitude (Geary: (n-1)*sum w (y_i-y_j)^2; Moran: n*sum w z_i z_j)
        _wsum = max(1.0, float(abs(graph).sum()))
        in_range = bool(np.float64(2.0 * (abs(a) * float(np.max(np.abs(v64))) + b)) ** 2 * n * _wsum < _range_limit(np.float64))
        bad2 |= in_range and _violates(abs(aff - base), _C * _EPS * kap2 * lg * (1.0 + abs(base)))
        # SC-AUT-003: representation consistency
        tol3 = _C * max(_EPS, eps_v) * n * (1.0 + (mean / std) ** 2) * (1.0 + abs(r0))
        with _quiet():
            r_d = float(np.atleast_1d(cls(graph, v[None, :])())[0])
            r_s = float(np.atleast_1d(cls(graph, sparse.csr_matrix(v[None, :]))())[0])
        bad3 |= _violates(abs(r_d - r0), tol3) or _violates(abs(r_s - r0), tol3)
    trigger_if(bad1, "SC-AUT-001")
    trigger_if(bad2, "SC-AUT-002")
    trigger_if(bad3, "SC-AUT-003")


# ================================================== H. differential expression
_KEY = "_scibench_chk"


def _de_matrix(adata, use_raw, layer, mask_var):
    if layer is not None:
        x, names = adata.layers[layer], adata.var_names
    elif use_raw and adata.raw is not None:
        x, names = adata.raw.X, adata.raw.var_names
    else:
        x, names = adata.X, adata.var_names
    if mask_var is not None:
        x, names = x[:, mask_var], names[mask_var]
    return x, names


def _slim(adata, groupby, x, names, negate=False):
    import pandas as pd
    from anndata import AnnData

    xx = x.copy()
    if negate:
        xx = -xx
    obs = pd.DataFrame({groupby: adata.obs[groupby].astype("category")}, index=adata.obs_names)
    slim = AnnData(X=xx, obs=obs, var=pd.DataFrame(index=names))
    if "log1p" in adata.uns:  # the log base decides how log fold changes exponentiate the data
        slim.uns["log1p"] = dict(adata.uns["log1p"])
    return slim


def _res_by_name(res, field, group):
    names = np.asarray(res["names"][group])
    vals = np.asarray(res[field][group], dtype=np.float64)
    return dict(zip(names.tolist(), vals.tolist(), strict=True))


def _recall_de(slim, groupby, *, groups, reference, method, corr_method, tie_correct, mean_in_log_space):
    from .tools._rank_genes_groups import rank_genes_groups

    with _quiet():
        rank_genes_groups(
            slim, groupby, groups=groups, reference=reference, method=method, corr_method=corr_method,
            tie_correct=tie_correct, mean_in_log_space=mean_in_log_space, key_added=_KEY,
            use_raw=False, n_genes=None, rankby_abs=False, pts=False,
        )
    return slim.uns[_KEY]


@_guarded
def check_rank_genes_groups(adata, key, groupby, *, reference, method, use_raw, layer, mask_var,
                            corr_method, tie_correct, mean_in_log_space):
    """SC-DE-001 .. SC-DE-005."""
    if method not in {"t-test", "t-test_overestim_var", "wilcoxon"}:
        return
    res = adata.uns[key]
    x, names = _de_matrix(adata, use_raw, layer, mask_var)
    if not (isinstance(x, np.ndarray) or _is_sparse(x)) or _size(x) > _MAX_ELEMS or _has_duplicates(x):
        return
    obs = adata.obs[groupby].astype("category")
    cats = {str(c): c for c in obs.cat.categories}
    groups = [g for g in res["scores"].dtype.names if g in cats][:3]
    if not groups:
        return
    n_obs = x.shape[0]
    slim = _slim(adata, groupby, x, names)
    kw = dict(method=method, corr_method=corr_method, tie_correct=tie_correct, mean_in_log_space=mean_in_log_space)
    eps32 = float(np.finfo(np.float32).eps)
    have_lfc = "logfoldchanges" in res
    sizes = obs.value_counts()
    bad = dict.fromkeys(("001", "002", "003", "004", "005"), False)
    _xs = _dense64(x)
    # p-values are functions of statistics built from one-pass moments: error grows with 1 + mean^2/var
    kappa_p = 1.0 + float(np.max(_xs.mean(axis=0) ** 2 / np.maximum(_xs.var(axis=0, ddof=1), 1e-300))) if _xs.shape[0] > 1 else 1.0

    def gap(a, b, tol_extra=0.0):
        return np.abs(a - b) > 0

    # SC-DE-001 / 003: swap groups against an explicit reference
    if reference != "rest" and method != "t-test_overestim_var":
        for g in groups:
            r2 = _recall_de(slim, groupby, groups=[reference], reference=g, **kw)
            if reference not in r2["scores"].dtype.names:
                continue
            n_g, n_r = int(sizes[cats[g]]), int(sizes[cats[reference]])
            N = n_g + n_r
            s1, s2 = _res_by_name(res, "scores", g), _res_by_name(r2, "scores", reference)
            p1, p2 = _res_by_name(res, "pvals", g), _res_by_name(r2, "pvals", reference)
            for nm, s in s1.items():
                if nm in s2 and np.isfinite(s) and np.isfinite(s2[nm]):
                    bad["001"] |= _violates(abs(s + s2[nm]), _C * eps32 * (1.0 + abs(s)))
                    pt = _C * _EPS * N * kappa_p * (1.0 + abs(s))
                    bad["001"] |= _violates(abs(p1[nm] - p2[nm]), pt)
            if have_lfc and "logfoldchanges" in r2:
                l1, l2 = _res_by_name(res, "logfoldchanges", g), _res_by_name(r2, "logfoldchanges", reference)
                for nm, v in l1.items():
                    if nm in l2 and np.isfinite(v) and np.isfinite(l2[nm]):
                        bad["003"] |= _violates(abs(v + l2[nm]), _C * eps32 * (1.0 + abs(v)))
    # SC-DE-002: rest vs the other group when there are exactly two populated groups
    populated = [c for c in obs.cat.categories if sizes[c] > 0]
    if len(populated) == 2 and not obs.isna().any():
        for g in groups:
            other = [str(c) for c in populated if str(c) != g]
            if len(other) != 1:
                continue
            if reference == "rest":
                r2 = _recall_de(slim, groupby, groups=[g], reference=other[0], **kw)
            elif reference == other[0]:
                r2 = _recall_de(slim, groupby, groups=[g], reference="rest", **kw)
            else:
                continue
            if g not in r2["scores"].dtype.names:
                continue
            xs = _dense64(x)
            if not mean_in_log_space:  # statistics are computed on the exponentiated values
                _b = adata.uns.get("log1p", {}).get("base")
                xs = np.expm1(xs * np.log(_b)) if _b is not None else np.expm1(xs)
            kappa = 1.0 + float(np.max(xs.mean(axis=0) ** 2 / np.maximum(xs.var(axis=0, ddof=1), 1e-300)))
            # means_rest = (sum_h n_h m_h - n_g m_g) / n_rest: absolute error ~ eps * sum_h n_h |m_h| / n_rest
            gm = (obs == cats[g]).to_numpy()
            m_rest = xs[~gm].mean(axis=0)
            d_rest = _C * _EPS * _log2(n_obs) * np.abs(xs).sum(axis=0) / max(int((~gm).sum()), 1)
            lfc_tol = d_rest / ((np.abs(m_rest) + 1e-9) * math.log(2.0))
            idx_n = {nm: i for i, nm in enumerate(names.tolist())}
            for fld in ("scores", "logfoldchanges") if have_lfc else ("scores",):
                a, b = _res_by_name(res, fld, g), _res_by_name(r2, fld, g)
                for nm, v in a.items():
                    if nm in b and np.isfinite(v) and np.isfinite(b[nm]):
                        extra = 0.0
                        if fld == "logfoldchanges":
                            j = idx_n[nm]
                            if lfc_tol[j] >= 0.5:  # rest mean not resolvable at all: law undefined
                                continue
                            extra = float(lfc_tol[j])
                        bad["002"] |= _violates(abs(v - b[nm]), _C * (eps32 + _EPS * n_obs * kappa) * (1.0 + abs(v)) + extra)
    # SC-DE-004: log fold change direction vs. the means
    if have_lfc and _min_value(x) >= 0:  # (m + pseudocount) ratios only order means for non-negative data
        xs_all = x
        for g in groups:
            mg_mask = (obs == cats[g]).to_numpy()
            if reference == "rest":
                mr_mask = ~mg_mask
            else:
                mr_mask = (obs == cats[reference]).to_numpy()
            if mg_mask.sum() == 0 or mr_mask.sum() == 0:
                continue
            xg = _dense64(xs_all[mg_mask]) if _size(xs_all[mg_mask]) <= _MAX_ELEMS else None
            xr = _dense64(xs_all[mr_mask]) if _size(xs_all[mr_mask]) <= _MAX_ELEMS else None
            if xg is None or xr is None:
                continue
            if not mean_in_log_space:  # documented: log(mean(e^x)); means are taken over exponentiated values
                base = adata.uns.get("log1p", {}).get("base")
                f = (lambda a: np.expm1(a * np.log(base))) if base is not None else np.expm1
                xg, xr = f(xg), f(xr)
            mg, mr = xg.mean(axis=0), xr.mean(axis=0)
            eps_x = _eps_of(x)
            lf = _res_by_name(res, "logfoldchanges", g)
            idx = {nm: i for i, nm in enumerate(names.tolist())}
            for nm, v in lf.items():
                i = idx.get(nm)
                if i is None or not np.isfinite(v):
                    continue
                dm = mg[i] - mr[i]
                # lfc = log2((mg+1e-9)/(mr+1e-9)): a mean difference below rounding of the pseudocounted means is unresolvable
                tol = _C * max(eps_x, _EPS) * (abs(mg[i]) + abs(mr[i]) + 1e-9) * _log2(n_obs) + 1e-8 * (abs(mg[i]) + abs(mr[i]))
                if abs(dm) > tol and v != 0:
                    bad["004"] |= (np.sign(v) != np.sign(dm))
                elif abs(dm) > tol and v == 0:
                    bad["004"] |= True
    # SC-DE-005: negating the data negates the statistic (t statistics are computed on expm1 of the
    # data when mean_in_log_space=False, which is not odd, so only wilcoxon is in the domain there)
    # (negating unsigned integers wraps around, so those inputs are outside the law's domain)
    if _dtype_kind(x) != "u" and (method == "wilcoxon" or (method == "t-test" and mean_in_log_space)):
        neg = _slim(adata, groupby, x, names, negate=True)
        for g in groups:
            ref_g = reference
            r2 = _recall_de(neg, groupby, groups=[g], reference=ref_g, **kw)
            if g not in r2["scores"].dtype.names:
                continue
            n_g = int(sizes[cats[g]])
            N = n_obs if reference == "rest" else n_g + int(sizes[cats[reference]])
            s1, s2 = _res_by_name(res, "scores", g), _res_by_name(r2, "scores", g)
            p1, p2 = _res_by_name(res, "pvals", g), _res_by_name(r2, "pvals", g)
            for nm, s in s1.items():
                if nm in s2 and np.isfinite(s) and np.isfinite(s2[nm]):
                    bad["005"] |= _violates(abs(s + s2[nm]), _C * eps32 * (1.0 + abs(s)))
                    bad["005"] |= _violates(abs(p1[nm] - p2[nm]), _C * _EPS * N * kappa_p * (1.0 + abs(s)))
    trigger_if(bad["001"], "SC-DE-001")
    trigger_if(bad["002"], "SC-DE-002")
    trigger_if(bad["003"], "SC-DE-003")
    trigger_if(bad["004"], "SC-DE-004")
    trigger_if(bad["005"], "SC-DE-005")


@_guarded
def check_rest_stats(rg, x):
    """SC-DE-006 (``x`` is the matrix the statistics were computed on)."""
    from fast_array_utils import stats

    if not (isinstance(x, np.ndarray) or _is_sparse(x)) or _size(x) > _MAX_ELEMS or _has_duplicates(x):
        return
    masks = np.asarray(rg.groups_masks_obs, dtype=bool)
    k = masks.shape[0]
    if rg.means_rest is None or not _all_finite(x):
        return
    eps = _eps_of(x)
    n = x.shape[0]
    bad = False
    sel_counts = masks.sum(axis=1)
    mean_abs = [np.abs(np.asarray(rg.means[g])) for g in range(k)]
    for g in range(min(k, 4)):
        rest = ~masks[g]
        n_rest = int(rest.sum())
        if n_rest < 2:
            continue
        xr = x[rest]
        xr = _as_f64(xr)  # integer accumulators wrap, float32 ones underflow/round: reference in float64
        m, v = stats.mean_var(xr, axis=0, correction=1)
        m, v = np.asarray(m, dtype=np.float64), np.asarray(v, dtype=np.float64)
        scale = sum(sel_counts[h] * mean_abs[h] for h in range(k)) / n_rest + np.abs(m)
        tol_m = _C * max(eps, _EPS) * (scale + 1e-300) * _log2(n)
        bad |= bool(np.any(np.abs(np.asarray(rg.means_rest[g]) - m) > tol_m))
        if rg.vars_rest is not None:
            kappa = 1.0 + m**2 / np.maximum(v, 1e-300)
            tol_v = _C * max(eps, _EPS) * n * kappa * np.maximum(v, 1e-300)
            bad |= bool(np.any(np.abs(np.asarray(rg.vars_rest[g]) - v) > tol_v))
    trigger_if(bad, "SC-DE-006")


# ================================================== I. gene-set scoring
def copy_rng(rng):
    try:
        return _copy.deepcopy(rng)
    except Exception:
        _swallowed()
        return None


@_guarded
def check_score_genes(adata, gene_list, rng0, *, ctrl_as_ref, ctrl_size, gene_pool, n_bins, score_name,
                      use_raw, layer):
    """SC-SG-001 / 002 (re-calls ``score_genes`` on a slim copy)."""
    from .tools._score_genes import score_genes

    if rng0 is None:
        return
    x, names = _de_matrix(adata, use_raw, layer, None)
    if not (isinstance(x, np.ndarray) or _is_sparse(x)) or _size(x) > _MAX_ELEMS or not _all_finite(x):
        return
    import pandas as pd
    from anndata import AnnData

    obs = pd.DataFrame(index=adata.obs_names)

    def run(xx, genes):
        ad = AnnData(X=xx, obs=obs.copy(), var=pd.DataFrame(index=names))
        with _quiet():
            score_genes(ad, genes, ctrl_as_ref=ctrl_as_ref, ctrl_size=ctrl_size, gene_pool=gene_pool,
                        n_bins=n_bins, score_name=score_name, rng=_copy.deepcopy(rng0), use_raw=False)
        return ad.obs[score_name].to_numpy(dtype=np.float64)

    prod = adata.obs[score_name].to_numpy(dtype=np.float64)
    glist = list(gene_list)
    scale = max(_max_abs(x), 1e-300)
    eps = _eps_of(x)
    # gene means sum n_vars values: past the dtype range the score itself is inf/NaN
    if not np.all(np.isfinite(prod)) or scale * len(names) >= _range_limit(x.dtype):
        return
    # SC-SG-001: order invariance (production dtype)
    rev = run(x.copy(), glist[::-1])
    seq = 4.0 * len(names)  # _sparse_nanmean_* accumulate sequentially: mean error <= (k-1)u*max|x|
    tol = (_C * _log2(len(names)) + seq) * max(eps, _EPS) * scale
    trigger_if(_violates(float(np.max(np.abs(rev - prod))), tol), "SC-SG-001")
    # SC-SG-002: additive shift, float64 baseline vs shifted
    xd = _dense64(x)
    shift = 2.0 ** math.ceil(math.log2(scale + 1.0))
    # rank collisions: a shift can split (or merge) genes whose means agree to rounding but whose columns
    # differ, which changes the expression bins; require such near-equal means to belong to identical columns
    mu = xd.mean(axis=0)
    order = np.argsort(mu, kind="stable")
    _, cid = np.unique(xd, axis=1, return_inverse=True)
    cid = np.asarray(cid).ravel()
    near = np.diff(mu[order]) <= 1e3 * _EPS * (shift + scale)
    if np.any(near & (cid[order][1:] != cid[order][:-1])):
        return
    if (scale + shift) * len(names) >= _range_limit(np.float64):  # shifted copy leaves the dtype range
        return
    base = run(xd, glist)
    moved = run(xd + shift, glist)
    if not (np.all(np.isfinite(base)) and np.all(np.isfinite(moved))):
        return
    trigger_if(_violates(float(np.max(np.abs(moved - base))), (_C * _log2(len(names)) + seq) * _EPS * (scale + shift)), "SC-SG-002")


@_guarded
def check_control_genes(control_genes, gene_list, gene_pool):
    """SC-SG-003."""
    ctrl = set(map(str, control_genes))
    trigger_if(bool(ctrl & set(map(str, gene_list))) or not ctrl <= set(map(str, gene_pool)), "SC-SG-003")


# ================================================== J. connectivities
@_guarded
def check_connectivities(w, method, distances=None):
    """SC-NBR-001."""
    if w is None:
        return
    # the gaussian kernel exp(-d^2/(s_i^2+s_j^2)) amplifies the rounding of float32 distances by an
    # unbounded (width-dependent) factor, so that leg is judged for float64 distances only
    if method == "gauss" and distances is not None and np.dtype(distances.dtype).itemsize < 8:
        return
    eps = _eps_of(w)
    # non-finite weights (e.g. a zero local scale at duplicate points) are not judged here
    if _is_sparse(w):
        a = w.tocsr()
        data = a.data[np.isfinite(a.data)]
        d = abs(a - a.T)
        d = d.data[np.isfinite(d.data)]
        diff = float(d.max()) if d.size else 0.0
        top = float(data.max()) if data.size else 0.0
        low = float(data.min()) if data.size else 0.0
    else:
        a = np.asarray(w, dtype=np.float64)
        fin = np.isfinite(a)
        pair = fin & fin.T
        data = a[fin]
        diff = float(np.max(np.abs(a - a.T)[pair])) if pair.any() else 0.0
        top = float(data.max()) if data.size else 0.0
        low = float(data.min()) if data.size else 0.0
    tol = _C * eps * max(1.0, top)
    trigger_if(_violates(diff, tol) or low < -tol or top > 1.0 + tol, "SC-NBR-001")


# ================================================== K. aggregation
@_guarded
def check_aggregate(data, by, mask, result, dof):
    """SC-AGG-001 (law of total variance)."""
    from fast_array_utils import stats

    if dof != 1 or "mean" not in result or "var" not in result:
        return
    if not (isinstance(data, np.ndarray) or _is_sparse(data)) or not _all_finite(data):
        return
    if _has_duplicates(data):
        return
    codes = np.asarray(by.codes)
    keep = codes >= 0
    if mask is not None:
        keep = keep & np.asarray(mask, dtype=bool)
    n_tot = int(keep.sum())
    if n_tot < 3:
        return
    n_cat = len(by.categories)
    counts = np.bincount(codes[keep], minlength=n_cat).astype(np.float64)
    mean_g = np.asarray(result["mean"], dtype=np.float64)
    var_g = np.asarray(result["var"], dtype=np.float64)
    nonempty = counts > 0
    xk = data[keep] if not _is_sparse(data) else data.tocsr()[keep]
    xk = _as_f64(xk)  # integer accumulators wrap, float32 ones underflow: reference in float64
    m, v = stats.mean_var(xk, axis=0, correction=1)
    m, v = np.asarray(m, dtype=np.float64), np.asarray(v, dtype=np.float64)
    ssw = np.where((counts >= 2)[:, None], (counts[:, None] - 1.0) * np.nan_to_num(var_g), 0.0)[nonempty].sum(axis=0)
    ssb = (counts[nonempty, None] * (mean_g[nonempty] - m[None, :]) ** 2).sum(axis=0)
    sst = (n_tot - 1.0) * v
    kappa = 1.0 + m**2 / np.maximum(v, 1e-300)
    # the statistics are accumulated in the dtype of the data (float32 data: ~1e-6 relative)
    rel = _C * max(_EPS, _eps_of(data)) * n_tot * kappa
    _mx = np.asarray(abs(xk).max(axis=0).todense() if _is_sparse(xk) else np.abs(xk).max(axis=0), dtype=np.float64).ravel()
    # squares past the float64 range (sum of squares overflows in the kernels) are outside the law's domain
    in_rng = _mx**2 * n_tot < _range_limit(np.float64)
    judged = in_rng & (v >= _TINY_N) & (rel < 1.0)  # subnormal-range variances carry only a few bits  # a numerically constant gene makes the identity vacuous
    tol = rel * sst
    bad = judged & ~(np.abs(sst - (ssw + ssb)) <= tol)
    trigger_if(bool(np.any(bad)), "SC-AGG-001")


# ================================================== L. ComBat
@_guarded
def check_combat(adata, data, corrected, key, covariates):
    """SC-CMB-001 / 002 (no covariates)."""
    import pandas as pd
    from anndata import AnnData

    if covariates:
        return
    x = np.asarray(corrected, dtype=np.float64)  # cells x genes
    orig = np.asarray(data.to_numpy(), dtype=np.float64).T
    if x.shape != orig.shape or not (np.all(np.isfinite(x)) and np.all(np.isfinite(orig))):
        return
    batches = adata.obs[key].astype("category")
    cats = list(batches.cat.categories)
    if len(cats) < 2:
        return
    # SC-CMB-002: within-batch affine map with positive scale
    bad = False
    eps = _EPS
    for c in cats:
        idx = np.flatnonzero((batches == c).to_numpy())
        if len(idx) < 3:
            continue
        xo, xc = orig[idx], x[idx]
        xc0 = xo - xo.mean(axis=0)
        yc0 = xc - xc.mean(axis=0)
        sxx = (xc0**2).sum(axis=0)
        ok = sxx > len(idx) * (1e-9 * np.abs(xo.mean(axis=0))) ** 2
        ok &= sxx > 0
        # ComBat adds back the pooled location: a batch whose spread is below the rounding of the gene's
        # overall magnitude cannot carry its affine image (e.g. 1e-50 vs 1e50 across batches)
        ok &= np.sqrt(sxx / len(idx)) > _C * eps * np.max(np.abs(orig), axis=0)
        if not ok.any():
            continue
        slope = (xc0[:, ok] * yc0[:, ok]).sum(axis=0) / sxx[ok]
        resid = yc0[:, ok] - slope * xc0[:, ok]
        yn = np.linalg.norm(yc0[:, ok], axis=0)
        syy = (yc0[:, ok] ** 2).sum(axis=0)
        # outputs are stored at the batch-corrected location: their own rounding is eps*|mean_out|/sd_out
        kappa = (1.0 + np.abs(xo.mean(axis=0)[ok]) / np.sqrt(sxx[ok] / len(idx))
                 + np.abs(xc.mean(axis=0)[ok]) / np.sqrt(np.maximum(syy, 1e-300) / len(idx)))
        rel = np.linalg.norm(resid, axis=0) / np.maximum(yn, 1e-300)
        bad |= bool(np.any(~(rel <= _C * eps * kappa * math.sqrt(len(idx))))) or bool(np.any(slope <= 0))
    trigger_if(bad, "SC-CMB-002")
    # SC-CMB-001: relabelling (reverse category order); tolerance from the EB iteration's own
    # stopping rule (relative change 1e-4 per sweep, so two runs may stop one sweep apart).
    from .preprocessing._combat import combat

    obs = pd.DataFrame({key: pd.Categorical(batches.astype(object), categories=cats[::-1])},
                       index=adata.obs_names)
    ad2 = AnnData(X=orig.copy(), obs=obs, var=pd.DataFrame(index=adata.var_names))
    with _quiet():
        x2 = combat(ad2, key, inplace=False)
    conv = 1e-4
    tol = 50 * conv * max(float(np.max(np.abs(orig))), 1e-300)
    trigger_if(_violates(float(np.max(np.abs(np.asarray(x2) - x))), tol), "SC-CMB-001")


# ================================================== M. PCA
_EXACT_SOLVERS = {"arpack", "covariance_eigh", "full"}


@_guarded
def check_pca(x, x_pca, pca_, zero_center, svd_solver, chunked):
    """SC-PCA-001 / 002."""
    from fast_array_utils import stats

    if chunked or not zero_center or svd_solver not in _EXACT_SOLVERS:
        return
    if not (isinstance(x, np.ndarray) or _is_sparse(x)) or _size(x) > _MAX_ELEMS or not _all_finite(x):
        return
    v = np.asarray(pca_.components_, dtype=np.float64)
    ev = np.asarray(pca_.explained_variance_, dtype=np.float64)
    evr = np.asarray(pca_.explained_variance_ratio_, dtype=np.float64)
    s = np.asarray(x_pca, dtype=np.float64)
    k, n_vars = v.shape
    n_obs = x.shape[0]
    if k < 1 or n_obs < 3 or not (np.all(np.isfinite(v)) and np.all(np.isfinite(s))):
        return
    eps = max(_eps_of(pca_.components_), _eps_of(x_pca), _eps_of(x))
    gap = float(np.min(np.abs(np.diff(ev)))) / max(float(ev[0]), 1e-300) if k > 1 else 1.0
    m, vv = stats.mean_var(_as_f64(x), axis=0, correction=1)  # integer reference would wrap
    m, vv = np.asarray(m, dtype=np.float64), np.asarray(vv, dtype=np.float64)
    total = float(vv.sum())
    if not total >= max(_TINY_N, _tiny_n(x.dtype)) * n_vars:  # dtype-aware: float32 squares underflow below 1e-31  # variances in the subnormal range: products lose precision
        return
    # variances past the dtype range (inf) make every ratio inf/inf: outside the law's domain
    if not (np.isfinite(total) and np.all(np.isfinite(ev)) and np.all(np.isfinite(evr)) and np.all(np.isfinite(m))):
        return
    # explicit-covariance solvers inherit the one-pass conditioning 1 + sum(mean^2)/sum(var)
    kappa_tot = 1.0 + float(np.sum(m**2)) / total
    bad1 = _violates(float(np.max(np.abs(v @ v.T - np.eye(k)))), _C * eps * n_vars)
    var_s = s.var(axis=0, ddof=1)
    bad1 |= _violates(float(np.max(np.abs(var_s - ev))), _C * eps * n_obs * kappa_tot * float(ev[0]))
    if k > 1 and gap >= 1e-2:
        cov = np.cov(s.T)
        off = np.abs(cov - np.diag(np.diag(cov))).max() / max(float(ev[0]), 1e-300)
        bad1 |= _violates(float(off), _C * eps * n_obs * kappa_tot / gap)
        bad1 |= _violates(float(np.max(np.abs(s.mean(axis=0)))), _C * eps * n_obs * kappa_tot * math.sqrt(float(ev[0])) + 1e-300)
    trigger_if(bad1, "SC-PCA-001")
    # absolute error on a ratio: eigenvalue error eps*n_vars*lambda_1 relative to the total, plus the
    # error of the total itself, eps*log2(n)*kappa_tot (per-gene errors scale with v_g + m_g^2)
    tol = _C * eps * (n_vars * float(ev[0]) / total + kappa_tot * _log2(n_obs * n_vars))
    trigger_if(_violates(float(np.max(np.abs(evr - ev / total))), tol), "SC-PCA-002")
