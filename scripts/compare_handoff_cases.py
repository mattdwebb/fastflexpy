"""Compare fastflexpy with the handoff simulation estimator on nine datasets."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE_ROOT/'src'))
from fastflexpy import Design


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.handoff_root.resolve()
    sys.path.insert(0,str(root))
    from combined_simulation.test_flex_grouping import fixture,SCHEDULE,cache_for
    from combined_simulation.estimators import estimate_flex

    cases=[]
    for name,covs,unequal,weighted in [('no_cov',[],False,False),('one_cov',['x'],False,False),
            ('two_cov',['x','z'],False,False),('unequal',['x','z'],True,False),
            ('weighted',['x','z'],True,True)]:
        cases.append((name,fixture(unequal=unequal),SCHEDULE,covs,weighted))
    data=fixture(unequal=True)
    cases.append(('delete_state',data.loc[data.state.ne(2)].copy(),
        {s:g for s,g in SCHEDULE.items() if s!=2},['x','z'],False))
    cases.append(('random_schedule',data.copy(),{1:5,2:0,3:3,4:0,5:3,6:5},['x','z'],False))
    source=root/'results/didint_smooth_allperiod_ipt_10000rep/prepared/source_panel.csv'
    if not source.is_file():raise FileNotFoundError(source)
    original=pd.read_csv(source);schedule=original.groupby('state').cohort.first().to_dict()
    for variant in ('cohort_dynamics','state_dynamics'):
        frame=original.copy();frame['y']=frame['base_smooth_allperiod_ipt']
        if variant=='state_dynamics':
            k=frame.state.map({15:1.,21:0.,22:3.,23:2.,54:6.})
            frame['y']+=(k-frame.k)*(frame.year-2000)*(frame.training-.3)
        cases.append((variant,frame,schedule,['union','training'],False))
    if len(cases)!=9:raise RuntimeError(f'Expected 9 cases, found {len(cases)}')

    rows=[]
    for name,data,schedule,covs,weighted in cases:
        for grouping in ('state','cohort'):
            fit=Design(data,outcome='y',cluster='state',time='year',assignment=schedule,
                covariates=covs,weights='weight' if weighted else None,grouping=grouping).fit()
            oracle=estimate_flex(cache_for(data,covs,weighted),schedule,grouping=grouping)
            rows.append(dict(case=name,grouping=grouping,fastflexpy=fit['estimate'],
                simulation=oracle.native_att,absolute_difference=abs(fit['estimate']-oracle.native_att),
                components=len(fit['components'])))
    result=pd.DataFrame(rows);args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    cohort=result.loc[result.grouping.eq('cohort'),'absolute_difference'].max()
    state_unweighted=result.loc[result.grouping.eq('state')&result.case.ne('weighted'),'absolute_difference'].max()
    state_weighted=result.loc[result.grouping.eq('state')&result.case.eq('weighted'),'absolute_difference'].iloc[0]
    print(json.dumps(dict(cases=len(result),max_cohort_difference=float(cohort),
        max_unweighted_state_difference=float(state_unweighted),
        historical_weighted_state_difference=float(state_weighted)),indent=2))


if __name__=='__main__':main()
