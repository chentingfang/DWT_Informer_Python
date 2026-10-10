
from __future__ import annotations
import numpy as np


def _get_pywt():
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("The DWT frontend needs PyWavelets: pip install PyWavelets") from exc
    return pywt


def sure_soft_threshold(detail: np.ndarray) -> float:
    """Minimize SURE over the soft-threshold candidate set, including zero.

    Noise variance is estimated with MAD for the *highest frequency* band;
    a universal-threshold upper bound prevents implausibly large thresholds.
    SURE is computed for standardized coefficients with unit noise variance.
    """
    detail = np.asarray(detail, dtype=np.float64)
    if detail.ndim != 1 or detail.size == 0:
        raise ValueError("detail must be a nonempty 1D coefficient vector")
    sigma = np.median(np.abs(detail - np.median(detail))) / 0.6744897501960817
    if sigma < 1e-12:
        return 0.0
    z = np.abs(detail) / sigma
    sq = np.sort(z * z)
    n = sq.size
    candidates = np.concatenate(([0.0], np.sqrt(sq)))
    # SURE(t) = n - 2 #{|z_i| <= t} + sum(min(z_i^2, t^2)).
    csum = np.concatenate(([0.0], np.cumsum(sq)))
    t_sq = candidates * candidates
    count = np.searchsorted(sq, t_sq, side="right")
    clipped = csum[count] + (n - count) * t_sq
    risk = n - 2 * count + clipped
    chosen = float(candidates[np.argmin(risk)] * sigma)
    universal = float(sigma * np.sqrt(2 * np.log(max(n, 2))))
    return min(chosen, universal)


def decompose_one(x: np.ndarray, mode: str = "selective", level: int = 3,
                  boundary: str = "symmetric", return_bands: bool = True) -> np.ndarray:
    """Decompose historical precipitation and reconstruct feature channels.

    Modes:
       none      -> raw precipitation, shape [L, 1]
       full      -> threshold every detail scale (M2-M4)
       selective -> threshold only cD1 (M5, proposed extreme-preserving)

    Returns four additive *reconstructed* bands [cA3,cD3,cD2,cD1] of length L.
    Their sum is the denoised precipitation trajectory, which is never clipped.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("x must be a finite 1D array")
    if mode == "none":
        return x.astype(np.float32)[:, None]
    if mode not in {"full", "selective"}:
        raise ValueError(f"Unknown wavelet mode: {mode}")
    pywt = _get_pywt()
    wave = pywt.Wavelet("db6")
    max_level = pywt.dwt_max_level(x.size, wave.dec_len)
    if max_level < level:
        raise ValueError(f"Window length {x.size} is too short for db6 level {level}; "
                         f"maximum supported level is {max_level}")
    coeffs = [np.array(v, dtype=np.float64, copy=True)
              for v in pywt.wavedec(x, wave, mode=boundary, level=level)]
    # [cA3,cD3,cD2,cD1]
    for i in range(1, level + 1):
        is_highest = (i == level)
        if mode == "full" or is_highest:
            t = sure_soft_threshold(coeffs[i])
            coeffs[i] = pywt.threshold(coeffs[i], value=t, mode="soft")
    if not return_bands:
        return pywt.waverec(coeffs, wave, mode=boundary)[:x.size].astype(np.float32)[:, None]
    bands = []
    for i in range(len(coeffs)):
        isolated = [np.zeros_like(c) for c in coeffs]
        isolated[i] = coeffs[i]
        bands.append(pywt.waverec(isolated, wave, mode=boundary)[:x.size])
    return np.stack(bands, axis=-1).astype(np.float32)


def transform_window(precip: np.ndarray, mode: str = "selective") -> np.ndarray:
    """[L,N] -> [L,N,4] or [L,N,1], preserving station identity."""
    precip = np.asarray(precip)
    if precip.ndim != 2:
        raise ValueError("Expected precipitation window with shape [time,stations]")
    return np.stack([decompose_one(precip[:, station], mode)
                     for station in range(precip.shape[1])], axis=1)
