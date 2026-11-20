from pathlib import Path
import os
import os,sys,json
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
p=Path(__file__).resolve().parents[1];sys.path.insert(0,str(p/'src'))
from fastflexpy import Design
work=p/'work/external';work.mkdir(parents=True,exist_ok=True)
rows=[];do=['clear all','set more off','set type double',f'adopath ++ "{Path(os.environ['FLEX_OFFICIAL_ADO'])}"',f'log using "{work.as_posix()}/official.log", text replace',f'postfile out str30 fixture double estimate double se double stat double p double df using "{work.as_posix()}/official.dta", replace']
for i in range(8):
    rng=np.random.default_rng(713+i);G=12;T=8
    s=np.repeat(np.arange(1,G+1),T*4);t=np.tile(np.repeat(np.arange(1,T+1),4),G);x=rng.normal(size=len(s));w=np.ones(len(s)) if i<6 else rng.uniform(.2,3,len(s))
    a=np.array([5]*3+[7]*3+[0]*6);d=(a[s-1]>0)&(t>=a[s-1]);cov=bool(i%2)
    y=.1*s+.2*t+x*(.3+.01*s+.02*t)+d*(1+.1*s+.2*x)+rng.normal(size=len(s))
    frame=pd.DataFrame(dict(y=y,state=s,year=t,x=x,w=w,D=d.astype(int),cohort=a[s-1]))
    if i in [2,3]:frame=frame.loc[~((frame.state==12)&(frame.year==3))]
    if i in [4,5]:frame=frame.loc[(frame.state%3==0)|(np.arange(len(frame))%2==0)]
    name=f'fixture_{i}';frame.to_stata(work/f'{name}.dta',write_index=False,version=118)
    design=Design(frame,outcome='y',cluster='state',time='year',assignment=dict(zip(range(1,G+1),a)),covariates=['x'] if cov else [],weights='w' if i>=6 else None)
    f=design.fit();r=design.fit(engine='reference');nat=design.native_cluster() if i<6 else None;nr=design.native_cluster(engine='reference') if i<6 else None
    rows.append(dict(fixture=name,point_fast=f['estimate'],point_reference=r['estimate'],component_error=float(np.max(abs(f['components']-r['components']))),native=nat,native_reference=nr))
    wt='[aw=w]' if i>=6 else ''
    do += [f'use "{(work/f"{name}.dta").as_posix()}", clear',f'flexdid y {"x" if cov else ""} {wt}, tx(D) group(state) time(year) specification(lagsonly) usercohort(cohort) vce(cluster state)', 'matrix tab=r(table)',f'post out ("{name}") (tab[1,1]) (tab[2,1]) (tab[3,1]) (tab[4,1]) (tab[7,1])']
do+=['postclose out',f'use "{work.as_posix()}/official.dta", clear','format estimate se stat p df %24.17g',f'export delimited using "{work.as_posix()}/official.csv", replace','log close','exit, clear']
(work/'official.do').write_text('\n'.join(do))
(work/'python.json').write_text(json.dumps(rows,indent=2,default=lambda v:v.tolist()))
print([(r['fixture'],abs(r['point_fast']-r['point_reference']),r['component_error']) for r in rows])

