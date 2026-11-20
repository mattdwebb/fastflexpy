from pathlib import Path
import os
import os,sys,json
for n in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[n]='1'
import numpy as np,pandas as pd
p=Path(__file__).resolve().parents[1];sys.path.insert(0,str(p/'src'))
from fastflexpy import Design
h=Path(os.environ['FLEX_HISTORY_RESULTS'])
panel=pd.read_csv(h/'prepared/source_panel.csv');eps=pd.read_csv(h/'prepared/disturbances.csv',nrows=2).set_index('replication_id');old=pd.read_csv(p/'docs/historical_replay_results.csv');out=[]
for _,row in old.iterrows():
    data=panel.copy();base='base_dgp1_no_x' if row.DGP=='dgp1_no_x' else 'base_dgp5_state_time_ipt';data['y']=data[base]+eps.loc[row.replication].to_numpy()
    if row.DGP=='dgp7_treatment_effect_x':data['y']+=data.tau_dgp6*data.treated_post
    d=Design(data,outcome='y',cluster='state',time='year',assignment=dict(data.groupby('state').cohort.first()),covariates=['age','educ','ofcolor','marstat'])
    f=d.point();r=d.point(engine='reference');out.append(dict(replication=int(row.replication),DGP=row.DGP,fast=f,reference=r,stored=float(row.historical_estimate),fast_stored_error=abs(f-row.historical_estimate),reference_stored_error=abs(r-row.historical_estimate)))
print(json.dumps(out,indent=2));(p/'docs/historical_fast_comparison.json').write_text(json.dumps(out,indent=2))
