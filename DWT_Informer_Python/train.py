
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data import ForecastWindows, load_series, DYNAMIC
from model import DWTInformer
from losses_metrics import intensity_weighted_mse,report_metrics

# Sequential ablation analogs following manuscript's Table 6:
# M1 Informer baseline; M2 full-band DWT; M3 +weighted loss; M4 +geo;
# M5 switch to peak-preserving selective DWT. The M4->M5 comparison changes
# threshold strategy; it is not an independently added layer.
VARIANTS={
 'M1': {'mode':'none',      'weighted':False,'geo':False},
 'M2': {'mode':'full',      'weighted':False,'geo':False},
 'M3': {'mode':'full',      'weighted':True, 'geo':False},
 'M4': {'mode':'full',      'weighted':True, 'geo':True},
 'M5': {'mode':'selective', 'weighted':True, 'geo':True},
}


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False


def evaluate(model, loader, device, geo, p95, collect=False):
    model.eval()
    obs,preds=[],[]
    with torch.no_grad():
        for x,y in loader:
            out=model(x.to(device),geo)
            preds.append(out.cpu().numpy())
            obs.append(y.numpy())
    if not preds:raise ValueError('Empty evaluation dataloader')
    pred=np.concatenate(preds,axis=0).reshape(-1,model.stations)
    truth=np.concatenate(obs,axis=0).reshape(-1,model.stations)
    stats=report_metrics(truth,pred,p95)
    return (stats,truth,pred) if collect else stats


def fit_one(args, series, horizon, variant):
    cfg=VARIANTS[variant]
    train=ForecastWindows(series,'train',args.history,horizon,cfg['mode'],args.train_stride)
    val=ForecastWindows(series,'val',args.history,horizon,cfg['mode'],horizon)
    test=ForecastWindows(series,'test',args.history,horizon,cfg['mode'],horizon)
    loader_kwargs={'batch_size':args.batch_size,'num_workers':args.workers}
    train_loader=DataLoader(train,shuffle=True,**loader_kwargs)
    val_loader=DataLoader(val,shuffle=False,**loader_kwargs)
    test_loader=DataLoader(test,shuffle=False,**loader_kwargs)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    channels=8 if cfg['mode']=='none' else 11
    model=DWTInformer(stations=len(series.station_ids),input_channels=channels,
            horizon=horizon,d_model=args.d_model,heads=args.heads,
            enc_layers=args.encoder_layers,dec_layers=args.decoder_layers,
            factor=args.factor,dropout=args.dropout,with_geo=cfg['geo'],
            with_station_attention=cfg['geo']).to(device)
    geo=torch.as_tensor(series.geo,device=device,dtype=torch.float32)
    threshold=torch.as_tensor(series.p95,device=device,dtype=torch.float32)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5)
    best_nse=-float('inf');best_epoch=0;wait=0;best_weights=None
    model_dir=Path(args.out)/f'{variant}_{horizon}day'
    model_dir.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train()
        running=0.0
        for features,targets in train_loader:
            features=features.to(device);targets=targets.to(device)
            opt.zero_grad(set_to_none=True)
            predicted=model(features,geo)
            if cfg['weighted']:
                loss=intensity_weighted_mse(predicted,targets,threshold,args.alpha,args.beta)
            else:
                loss=(predicted-targets).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            opt.step()
            running+=float(loss.detach())
        val_scores=evaluate(model,val_loader,device,geo,series.p95)
        scheduler.step(val_scores['RMSE_mm']**2)
        val_nse=val_scores['NSE']
        print(f'{variant} {horizon}d epoch={epoch:03d} '
              f'train_loss={running/len(train_loader):.5f} '
              f'val_NSE={val_nse:.5f} val_RMSE={val_scores["RMSE_mm"]:.4f}',flush=True)
        if np.isfinite(val_nse) and val_nse > best_nse + 1e-4:
            best_nse=val_nse;best_epoch=epoch;wait=0
            best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=args.early_stop:
                break
    if best_weights is None: raise RuntimeError('Validation never produced a finite NSE')
    model.load_state_dict(best_weights)
    test_scores,truth,pred=evaluate(model,test_loader,device,geo,series.p95,collect=True)
    torch.save({'state_dict':best_weights,'config':vars(args),
                'model_kwargs':{'stations':len(series.station_ids),'input_channels':channels,
                 'horizon':horizon,'d_model':args.d_model,'heads':args.heads,
                 'enc_layers':args.encoder_layers,'dec_layers':args.decoder_layers,
                 'factor':args.factor,'dropout':args.dropout,'with_geo':cfg['geo'],
                 'with_station_attention':cfg['geo']},
                'station_ids':series.station_ids,'geo':series.geo,
                'dynamic_mean':series.dynamic_mean,'dynamic_std':series.dynamic_std,
                'p95':series.p95,'dynamic_names':DYNAMIC,
                'best_epoch':best_epoch},model_dir/'checkpoint.pt')
    result={'variant':variant,'horizon_days':horizon,'best_epoch':best_epoch,
            'validation_NSE':best_nse,'test':test_scores,
            'num_training_windows':len(train),'num_validation_windows':len(val),
            'num_test_windows':len(test),'evaluation_stride_days':horizon,
            'seed':args.seed, 'dataset_note':'actual observations; not paper values'}
    (model_dir/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    dates=[date for chunk in test.target_dates() for date in chunk]
    records=[]
    for j,date in enumerate(dates):
        for s,sid in enumerate(series.station_ids):
            records.append((date.strftime('%Y-%m-%d'),sid,float(truth[j,s]),float(pred[j,s])))
    pd.DataFrame(records,columns=['date','station_id','observed_mm','predicted_mm'])\
      .to_csv(model_dir/'test_predictions.csv',index=False)
    return result


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--observations',required=True,help='Long-form daily meteorological CSV')
    p.add_argument('--stations',required=True,help='23-station latitude/longitude/elevation CSV')
    p.add_argument('--out',default='results')
    p.add_argument('--history',type=int,choices=(96,192,336),default=192)
    p.add_argument('--horizons',type=int,nargs='+',choices=(7,15,30),default=[7,15,30])
    p.add_argument('--variant',choices=[*VARIANTS,'all'],default='M5')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--early-stop',type=int,default=10)
    p.add_argument('--train-stride',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--d-model',type=int,default=512)
    p.add_argument('--heads',type=int,default=8)
    p.add_argument('--encoder-layers',type=int,default=3)
    p.add_argument('--decoder-layers',type=int,default=2)
    p.add_argument('--factor',type=int,default=5)
    p.add_argument('--dropout',type=float,default=0.05)
    p.add_argument('--alpha',type=float,default=3.)
    p.add_argument('--beta',type=float,default=0.75)
    p.add_argument('--seed',type=int,default=2026,help='Reproduction seed (not stated in paper)')
    p.add_argument('--allow-rain-gap-fill',action='store_true')
    p.add_argument('--cpu',action='store_true')
    return p.parse_args()


def main():
    args=parse_args()
    set_seed(args.seed)
    series=load_series(args.observations,args.stations,
                       allow_rain_gap_fill=args.allow_rain_gap_fill)
    if series.dates.max().year < 2023:
        raise ValueError('Daily CSV must include 2020-2023 for paper-defined test period')
    results=[]
    for variant in (list(VARIANTS) if args.variant=='all' else [args.variant]):
        for horizon in args.horizons:
            # Fair comparisons reinitialize each training model from the same seed.
            set_seed(args.seed)
            results.append(fit_one(args,series,horizon,variant))
    Path(args.out).mkdir(parents=True,exist_ok=True)
    (Path(args.out)/'summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    for result in results:
        print(f'COMPLETE {result["variant"]} {result["horizon_days"]}d:',result['test'])


if __name__=='__main__':
    main()
