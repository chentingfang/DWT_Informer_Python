
from __future__ import annotations
import numpy as np
import torch


def intensity_weighted_mse(pred: torch.Tensor, truth: torch.Tensor,
                           p95: torch.Tensor, alpha: float = 3.0,
                           beta: float = 0.75) -> torch.Tensor:
    """Eq. (24)-(25): w=1+alpha*1(y>P95)*(y/P95)**beta.

    p95 must be computed from TRAINING targets only and broadcast to stations.
    Targets/predictions must be expressed in original mm/day, not z-scores.
    """
    thr = p95.view(1,1,-1).to(device=truth.device,dtype=truth.dtype)
    ratio = torch.clamp(truth, min=0.0) / thr.clamp_min(1e-6)
    weight = 1.0 + alpha * (truth > thr).to(truth.dtype) * ratio.pow(beta)
    return (weight * (pred - truth).square()).mean()


def report_metrics(truth: np.ndarray, pred: np.ndarray,
                   p95: np.ndarray, absolute_threshold: float = 50.0) -> dict:
    """Aggregate unique station-day forecasts; avoid duplicated rolling windows.

    FAR uses the paper's definition: false alarms/(hits+false alarms).
    PRE/PTE are intentionally not fabricated without an event-matching policy.
    """
    y = np.asarray(truth,float)
    p = np.asarray(pred,float)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("truth/pred should both be [days,stations]")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Non-finite observations/predictions")
    rmse = np.sqrt(np.mean((p-y)**2))
    mae = np.mean(np.abs(p-y))
    y0, p0 = y.ravel(), p.ravel()
    denom = np.sum((y0 - np.mean(y0))**2)
    nse = 1 - np.sum((p0-y0)**2)/denom if denom > 0 else float('nan')
    corr = np.corrcoef(y0,p0)[0,1] if np.std(y0)>0 and np.std(p0)>0 else float('nan')
    threshold = np.maximum(np.asarray(p95,dtype=float), absolute_threshold)[None,:]
    actual_extreme = y >= threshold
    predicted_extreme = p >= threshold
    hit = int(np.logical_and(actual_extreme,predicted_extreme).sum())
    miss = int(np.logical_and(actual_extreme,~predicted_extreme).sum())
    false = int(np.logical_and(~actual_extreme,predicted_extreme).sum())
    def ratio(a,b):return float(a/b) if b else float('nan')
    return {'RMSE_mm':float(rmse),'MAE_mm':float(mae), 'R':float(corr),
            'NSE':float(nse),'POD':ratio(hit,hit+miss),
            'FAR':ratio(false,hit+false),'CSI':ratio(hit,hit+miss+false),
            'hits':hit,'misses':miss,'false_alarms':false,
            'positive_station_days':int(actual_extreme.sum()),
            'evaluated_station_days':int(y.size)}
