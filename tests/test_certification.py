import os
for k in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[k]='1'
import sys,json,unittest,concurrent.futures,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastflexpy import Design,InferenceError,assignment_draws
P=Path(__file__).resolve().parents[1]
OUTPUT=Path(os.environ.get('FASTFLEX_TEST_OUTPUT', P/'output/tests'))
OUTPUT.mkdir(parents=True,exist_ok=True)

def fixture(i):
    rng=np.random.default_rng(12000+i);G=50 if i==24 else 10;T=8
    rows=[]
    for s in range(G):
        for t in range(1,T+1):
            if i%7==2 and s==G-1 and t==3:continue
            for j in range(2+(s%5 if i%3==1 else 0)):rows.append((s,t))
    st=np.asarray(rows);s,t=st.T;a=np.array([5]*2+[7]*2+[0]*(G-4),float)
    if i%6==1:a=np.array([5]*4+[0]*(G-4),float)
    if i==25:a=np.array([5]+[0]*(G-1),float)
    x=rng.normal(size=len(s));d=(a[s]>0)&(t>=a[s]);tau=np.ones(len(s))
    if i%6==0:tau*=0
    if i%6==2:tau=.3*(s-2)
    if i%6==3:tau=.3*(t-5)
    if i%6==4:tau=np.where(s<2,1.,-2.) # zero average for balanced 4/2 posts
    y=.1*s+.2*t+.4*x+d*tau+rng.normal(size=len(s))
    data=dict(y=y,state=s*7+11,year=t,x=x,z=x+1e-13*rng.normal(size=len(s)),w=rng.uniform(.2,5,len(s)))
    cov=['x'] if i%2 else []
    if i==26:cov=['x','z']
    weights='w' if i in (27,28) else None
    if i==29:a*=0
    if i%5==3:
        order=rng.permutation(len(s));data={k:v[order] for k,v in data.items()}
    schedule=dict(zip(np.arange(G)*7+11,a))
    if i%5==4:
        data['state']=np.array([f'S{v:03d}' for v in data['state']]);schedule={f'S{k:03d}':v for k,v in schedule.items()}
    return data,schedule,cov,weights

def exercise(i):
    data,a,cov,w=fixture(i)
    des=Design(data,outcome='y',cluster='state',time='year',assignment=a,covariates=cov,weights=w)
    draws=np.asarray(list(assignment_draws(tuple(des.assignment),reps=2,seed=987)))
    nat=des.native_cluster() if w is None else None
    jk=des.jackknife();ri=des.randomization_inference(assignments=draws)
    repl=des.placebo_replace(assignments=draws);sub=des.placebo_subtract(assignments=draws)
    return [des.point(),None if nat is None else nat['native_cluster_se'],None if nat is None else nat['native_cluster_p'],jk['standard_error'],jk['p_value'],jk['delete_estimates'].tolist(),ri['statistics'].tolist(),ri['hits'],ri['p_value'],repl['procedural_p_values'].tolist(),repl['hits'],sub['procedural_p_values'].tolist(),sub['hits']]

class Certification(unittest.TestCase):
    def test_stress(self):
        records=[]
        for i in range(30):
            data,a,cov,w=fixture(i)
            if i==29:
                with self.assertRaises(InferenceError):Design(data,outcome='y',cluster='state',time='year',assignment=a)
                records.append(dict(case=i,status='expected failure',reason='no treated states'));continue
            des=Design(data,outcome='y',cluster='state',time='year',assignment=a,covariates=cov,weights=w)
            if i==26:
                simpler=Design(data,outcome='y',cluster='state',time='year',assignment=a,covariates=['x'])
                self.assertAlmostEqual(des.point(),simpler.point(),places=9)
                records.append(dict(case=i,status='identified margin with redundant slope'));continue
            f=des.fit();r=des.fit(engine='reference')
            ep=abs(f['estimate']-r['estimate']);ec=float(np.max(abs(f['components']-r['components'])));ew=float(np.max(abs(f['weights']-r['weights'])))
            self.assertLess(ep,1e-10);self.assertLess(ec,1e-9);self.assertLess(ew,1e-13)
            native_error=None
            if w is None:
                nf=des.native_cluster();nr=des.native_cluster(engine='reference');native_error=abs(nf['native_cluster_se']-nr['native_cluster_se']);self.assertLess(native_error,1e-9)
            else:
                with self.assertRaises(InferenceError):des.native_cluster()
            jf=des.jackknife();jr=des.jackknife(engine='reference')
            self.assertEqual(jf['jackknife_df'],len(des.states)-1)
            if i==25:
                self.assertEqual(jf['diagnostics']['status'],'singular_jackknife_replication');self.assertEqual(jr['diagnostics']['status'],jf['diagnostics']['status'])
                records.append(dict(case=i,status='expected failure',reason='last treated school deletion',point_error=ep));continue
            self.assertEqual(jf['diagnostics']['status'],'ok');self.assertEqual(jr['diagnostics']['status'],'ok')
            ej=float(np.max(abs(jf['delete_estimates']-jr['delete_estimates'])));self.assertLess(ej,1e-9);self.assertAlmostEqual(jf['p_value'],jr['p_value'],places=9)
            draws=list(assignment_draws(tuple(des.assignment),reps=2,seed=21))
            rf=des.randomization_inference(assignments=draws);rr=des.randomization_inference(assignments=draws,engine='reference')
            self.assertEqual(rf['hits'],rr['hits']);np.testing.assert_allclose(rf['statistics'],rr['statistics'],atol=1e-10,rtol=1e-10)
            eplace=0.
            for proc in ['replace','subtract']:
                pf=des.placebo(proc,assignments=draws);pr=des.placebo(proc,assignments=draws,engine='reference')
                self.assertEqual(pf['invalid_draws'],0);self.assertEqual(pf['hits'],pr['hits'])
                eplace=max(eplace,float(np.max(abs(pf['procedural_p_values']-pr['procedural_p_values']))));self.assertLess(eplace,1e-9)
            records.append(dict(case=i,status='pass',point_error=ep,component_error=ec,weight_error=ew,native_se_error=native_error,jackknife_deletion_error=ej,placebo_inner_p_error=eplace,native_weighted='N/A' if w else 'pass'))
        (OUTPUT/'stress_test_results.json').write_text(json.dumps(records,indent=2))
        import csv
        keys=sorted(set().union(*(r.keys() for r in records)))
        with (OUTPUT/'stress_test_results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)

    def test_processes(self):
        indices=list(range(8));serial=[exercise(i) for i in indices]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:parallel=list(pool.map(exercise,indices))
        self.assertEqual(serial,parallel)
        (OUTPUT/'process_isolation.json').write_text(json.dumps(dict(processes=8,designs=8,exact_serial_parallel=True,procedures=['point','native','JK','RI','Replace','Subtract']),indent=2))

    def test_assignments(self):
        a=(5,5,0,0);draws=list(assignment_draws(a,exact=True));self.assertEqual(len(draws),5);self.assertEqual(len(set(draws)),5);self.assertNotIn(a,draws)
        a=(5,5,7,7)+tuple([0]*46);b=list(assignment_draws(a,reps=999,seed=5));self.assertEqual(b,list(assignment_draws(a,reps=999,seed=5)));self.assertEqual(len(set(b)),999)
        for row in b:self.assertEqual((row.count(5),row.count(7),row.count(0)),(2,2,46))

    def test_add_scope(self):
        d,a,c,w=fixture(0);des=Design(d,outcome='y',cluster='state',time='year',assignment=a)
        with self.assertRaises(NotImplementedError):des.placebo_add(reps=2)

    def test_invalid_weights(self):
        d,a,c,w=fixture(0);d['w'][0]=0
        with self.assertRaises(InferenceError):Design(d,outcome='y',cluster='state',time='year',assignment=a,weights='w')

    def test_cohort_grouping(self):
        for i in (1,2,3,27,28):
            data,a,cov,w=fixture(i)
            des=Design(data,outcome='y',cluster='state',time='year',assignment=a,
                       covariates=cov,weights=w,grouping='cohort')
            fast=des.fit();reference=des.fit(engine='reference')
            self.assertLess(abs(fast['estimate']-reference['estimate']),1e-9)
            np.testing.assert_allclose(fast['components'],reference['components'],atol=1e-8,rtol=1e-10)
            np.testing.assert_allclose(fast['weights'],reference['weights'],atol=1e-13,rtol=1e-13)
            self.assertEqual(fast['cell_labels'],reference['cell_labels'])
        data,a,cov,w=fixture(2)
        state=Design(data,outcome='y',cluster='state',time='year',assignment=a,covariates=cov)
        cohort=Design(data,outcome='y',cluster='state',time='year',assignment=a,covariates=cov,grouping='cohort')
        self.assertGreater(abs(state.point()-cohort.point()),1e-6)
        jf=cohort.jackknife();jr=cohort.jackknife(engine='reference')
        np.testing.assert_allclose(jf['delete_estimates'],jr['delete_estimates'],atol=1e-8,rtol=1e-10)
        draws=list(assignment_draws(tuple(cohort.assignment),reps=2,seed=44))
        rf=cohort.randomization_inference(assignments=draws);rr=cohort.randomization_inference(assignments=draws,engine='reference')
        np.testing.assert_allclose(rf['statistics'],rr['statistics'],atol=1e-8,rtol=1e-10)
        with self.assertRaises(InferenceError):cohort.native_cluster()
        with self.assertRaises(ValueError):Design(data,outcome='y',cluster='state',time='year',assignment=a,grouping='bad')
        no_never={s:(5 if g==0 else g) for s,g in a.items()}
        with self.assertRaises(InferenceError):Design(data,outcome='y',cluster='state',time='year',assignment=no_never,grouping='cohort')

if __name__=='__main__':unittest.main(verbosity=2)
