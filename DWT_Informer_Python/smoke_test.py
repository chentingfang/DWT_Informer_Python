
from __future__ import annotations
import numpy as np
import torch
from model import DWTInformer
from losses_metrics import intensity_weighted_mse,report_metrics


def main():
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(2026)
    b,l,n,h=2,96,3,7
    model=DWTInformer(n,8,h,d_model=32,heads=4,enc_layers=2,
                      dec_layers=1,factor=2,station_dim=32,with_geo=True)
    features=torch.randn(b,l,n,8)
    geo=torch.randn(n,3)
    out=model(features,geo)
    assert out.shape==(b,h,n) and torch.isfinite(out).all() and (out>=0).all()
    y=torch.rand(b,h,n)*90
    loss=intensity_weighted_mse(out,y,torch.full((n,),50.0))
    loss.backward()
    assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
    ms=report_metrics(y[0].numpy(),out[0].detach().numpy(),np.full(n,40))
    assert 'RMSE_mm' in ms and ms['evaluated_station_days']==h*n
    print('PASS: model forward/backward, weighted MSE, station-day metrics')
    try:
        from wavelet import decompose_one
        np.random.seed(5)
        raw=np.random.randn(192).astype('float32')+4
        raw[57]+=100
        rawbands=decompose_one(raw,'selective')
        assert rawbands.shape==(192,4)
        assert np.isfinite(rawbands).all()
        assert np.max(np.abs(decompose_one(raw,'none')[:,0]-raw))==0
        print('PASS: db6 DWT selective highest-band threshold and reconstruction')
    except ImportError as exc:
        print(f'SKIP: DWT requires PyWavelets ({exc})')
    print('All available smoke tests completed; these are not paper validation data.')

if __name__=='__main__':main()
