
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from wavelet import transform_window

DYNAMIC = ['precip_mm','t2m_c','rh_pct','mslp_hpa',
           'z500_gpm','u850_ms','v850_ms','tcwv_kgm2']
GEO = ['latitude_deg','longitude_deg','elevation_m']


@dataclass
class StationSeries:
    dates: pd.DatetimeIndex
    station_ids: list[str]
    values: np.ndarray           # [time,station,8] original physical units
    geo: np.ndarray              # [station,3] normalized station coordinates
    dynamic_mean: np.ndarray     # [station,8] from training period only
    dynamic_std: np.ndarray      # [station,8] from training period only
    p95: np.ndarray              # [station] TRAIN-only daily precip percentile


def load_series(observations_csv: str, station_csv: str, n_stations: int = 23,
                allow_rain_gap_fill: bool = False) -> StationSeries:
    obs = pd.read_csv(observations_csv, dtype={'station_id':str}, parse_dates=['date'])
    meta = pd.read_csv(station_csv, dtype={'station_id':str})
    needed_obs = ['date','station_id'] + DYNAMIC
    needed_meta = ['station_id'] + GEO
    for name, frame, names in [('observations',obs,needed_obs),('stations',meta,needed_meta)]:
        missing_cols = set(names) - set(frame.columns)
        if missing_cols:
            raise ValueError(f'{name} file missing columns: {sorted(missing_cols)}')
    if obs.duplicated(['date','station_id']).any():
        raise ValueError('Multiple observations for the same date and station')
    if meta['station_id'].duplicated().any():
        raise ValueError('Duplicated station_id in station metadata')
    if len(meta) != n_stations:
        raise ValueError(f'Expected {n_stations} stations, got {len(meta)}')
    ids = meta['station_id'].tolist()
    if set(obs.station_id.unique()) != set(ids):
        raise ValueError('Station IDs in observation and metadata files differ')
    dates = pd.date_range(obs.date.min(),obs.date.max(),freq='D')
    mi = pd.MultiIndex.from_product([dates,ids],names=['date','station_id'])
    dense = obs.set_index(['date','station_id']).reindex(mi)
    if allow_rain_gap_fill:
        for sid in ids:
            s = dense.loc[(slice(None),sid),'precip_mm']
            if int(s.isna().sum()) > 10:
                raise ValueError(f'Station {sid} has >10 missing daily precipitation records')
            # Mirrors manuscript's use of linear interpolation, including its
            # possible look-ahead across missing days. This must be audited for
            # strict operational backtesting and within forecast windows.
            dense.loc[(slice(None),sid),'precip_mm'] = s.interpolate(method='linear',
                                                               limit_area='inside').to_numpy()
    if dense[DYNAMIC].isna().any().any():
        count = int(dense[DYNAMIC].isna().sum().sum())
        raise ValueError(f'{count} missing dynamic observations remain; complete QC before training')
    values = dense[DYNAMIC].to_numpy(dtype=np.float32).reshape(len(dates),len(ids),len(DYNAMIC))
    if np.any(values[:,:,0] < 0):
        raise ValueError('Negative observed precipitation in input file')
    train_mask = dates.year <= 2014
    if not train_mask.any():
        raise ValueError('No training years <= 2014 available')
    train_vals = values[train_mask]
    mean = train_vals.mean(axis=0,dtype=np.float64).astype(np.float32)
    std = train_vals.std(axis=0,dtype=np.float64).astype(np.float32)
    std = np.maximum(std,1e-6)
    p95 = np.quantile(train_vals[:,:,0],.95,axis=0).astype(np.float32)
    xyz = meta[GEO].to_numpy(dtype=np.float32)
    xyz = (xyz-xyz.mean(axis=0,keepdims=True))/np.maximum(xyz.std(axis=0,keepdims=True),1e-6)
    return StationSeries(dates,ids,values,xyz,mean,std,p95)


class ForecastWindows(Dataset):
    """Nonoverlapping evaluation windows; rolling training windows.

    Every forecast target timestamp is inside the designated split; encoder
    context may come from previous chronological splits (past-only).
    """
    YEAR_SPLITS = {'train':(1980,2014),'val':(2015,2019),'test':(2020,2023)}

    def __init__(self, series:StationSeries, split:str, history:int,
                 horizon:int, mode:str='selective', stride:int=1):
        if history < 96:
            raise ValueError("Paper's supported history windows start at 96 days")
        if split not in self.YEAR_SPLITS:
            raise ValueError(f'Unknown split: {split}')
        if stride < 1:
            raise ValueError('stride must be >= 1')
        self.series, self.history, self.horizon = series,history,horizon
        self.mode = mode
        years = series.dates.year
        yr0,yr1 = self.YEAR_SPLITS[split]
        ix = np.flatnonzero((years >= yr0)&(years <= yr1))
        if len(ix)==0:
            raise ValueError(f'No data for {split} set: {yr0}-{yr1}')
        begin=max(int(ix[0]),history)
        last=int(ix[-1])-horizon+1
        self.starts = list(range(begin,last+1,stride))
        if not self.starts:
            raise ValueError(f'No complete windows for {split}, history={history}, horizon={horizon}')
        if split!='train' and stride != horizon:
            raise ValueError('Use stride=horizon for validation/test to prevent duplicated station-days')

    def __len__(self):
        return len(self.starts)

    def __getitem__(self,i:int):
        start=self.starts[i]
        s=self.series
        segment=s.values[start-self.history:start]  # known historical observations ONLY
        rainfall = transform_window(segment[:,:,0],mode=self.mode)
        # Scale wavelet A3 like rainfall mean, details around zero (same mm std).
        rainfall = rainfall / s.dynamic_std[None,:,0,None]
        if self.mode == 'none':
            rainfall = rainfall - (s.dynamic_mean/s.dynamic_std)[None,:,0,None]
        else:
            rainfall[:,:,0] -= (s.dynamic_mean/s.dynamic_std)[None,:,0]
        exog = (segment[:,:,1:]-s.dynamic_mean[None,:,1:])/s.dynamic_std[None,:,1:]
        features=np.concatenate([rainfall,exog],axis=-1).astype(np.float32)
        target=s.values[start:start+self.horizon,:,0].astype(np.float32)
        return features,target

    def target_dates(self) -> list[pd.DatetimeIndex]:
        return [self.series.dates[k:k+self.horizon] for k in self.starts]
