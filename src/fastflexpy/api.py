"""Independent FLEX: pooled cell-interacted reference and cell-Schur fast path.

No observation-by-high-dimensional-dummy matrix is built in the fast path.
The explicit-dummy reference is permanently retained for bounded validation.
"""
import copy
import math
import numpy as np
from ._inference import InferenceMixin, InferenceError
from ._probability import student_t

class Design(InferenceMixin):
    kind='flex'
    def __init__(self,data,*,outcome,cluster,time,assignment,covariates=(),weights=None,grouping='state'):
        if grouping not in ('state','cohort'):raise ValueError("grouping must be 'state' or 'cohort'")
        self.grouping=grouping
        self.weighted=weights is not None
        self.y=np.asarray(data[outcome],float).copy()
        labels=np.asarray(data[cluster]);years=np.asarray(data[time],float)
        self.x=np.column_stack([np.asarray(data[c],float) for c in covariates]) if covariates else np.empty((len(self.y),0))
        self.w=np.ones(len(self.y)) if weights is None else np.asarray(data[weights],float).copy()
        if not len(self.y) or any(len(z)!=len(self.y) for z in [labels,years,self.w]):raise InferenceError('invalid_sample')
        if any(v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v) for v in labels):raise InferenceError('invalid_sample')
        if not all(np.isfinite(z).all() for z in [self.y,years,self.x,self.w]) or np.any(self.w<=0):raise InferenceError('invalid_sample','finite complete data and positive weights required')
        # Scaling is an invertible change of nuisance/effect basis.
        self.x_center=self.x.mean(0);self.x_scale=self.x.std(0)
        self.x_scale=np.where(self.x_scale>0,self.x_scale,1.)
        self.x=(self.x-self.x_center)/self.x_scale
        self.states=sorted(set(labels.tolist()));self.times=np.unique(years)
        codes={s:i for i,s in enumerate(self.states)}
        self._rowstate=np.array([codes[s] for s in labels]);self._rowtime=np.searchsorted(self.times,years)
        self.assignment=self._assignment(assignment)
        if self.grouping=='cohort' and not np.any(self.assignment==0):
            raise InferenceError('cohort_requires_never_treated','cohort grouping currently requires never-treated states')
        self._compile()

    def _compile(self):
        G,T=len(self.states),len(self.times);q=self.x.shape[1]+1
        self.v=np.column_stack([np.ones(len(self.y)),self.x]);self._cell=self._rowstate*T+self._rowtime
        def sums(z):return np.bincount(self._cell,weights=z,minlength=G*T).reshape(G,T)
        self.n=sums(np.ones(len(self.y)));self.mass=sums(self.w)
        self.vsum=np.stack([sums(self.w*self.v[:,i]) for i in range(q)],axis=-1)
        self.vmean=np.stack([sums(self.v[:,i]) for i in range(q)],axis=-1)/np.maximum(self.n[:,:,None],1)
        self.vy=np.stack([sums(self.w*self.v[:,i]*self.y) for i in range(q)],axis=-1)
        self.vv=np.empty((G,T,q,q))
        for i in range(q):
            for j in range(q):self.vv[:,:,i,j]=sums(self.w*self.v[:,i]*self.v[:,j])
        # Only cell-sized inverses; missing cells have no target mass.
        self.cell_ranks=np.linalg.matrix_rank(self.vv)
        self.cell_beta=np.linalg.pinv(self.vv,rcond=1e-13)@self.vy[:,:,:,None]
        self.cell_beta=self.cell_beta[:,:,:,0]
        self._cache=None
        self.__dict__.pop('_qr_cells',None)

    @staticmethod
    def _solve(A,b):
        scale=np.sqrt(np.maximum(np.diag(A),0))
        if np.any(scale<=0):raise InferenceError('rank_deficient','unidentified nuisance dimension')
        Z=A/scale[:,None]/scale[None,:]
        if np.linalg.eigvalsh(Z)[0]<1e-11:raise InferenceError('rank_deficient','ill-conditioned nuisance; use explicit reference for diagnosis')
        return np.linalg.solve(Z,b/scale)/scale

    def _blocks(self,a):
        if self._cache is not None and self._cache[0]==tuple(a):return self._cache[1]
        G,T=len(self.states),len(self.times);q=self.v.shape[1];P=(T-1)*q
        d=(a[:,None]>0)&(self.times[None,:]>=a[:,None]);u=~d
        A=(self.vv*u[:,:,None,None]).sum(1);r=(self.vy*u[:,:,None]).sum(1)
        if np.any(np.linalg.matrix_rank(A,tol=1e-10)<q):raise InferenceError('rank_deficient','state untreated nuisance is unidentified')
        Ai=np.linalg.inv(A)
        B=np.transpose(self.vv[:,1:]*u[:,1:,None,None],(0,2,1,3)).reshape(G,q,P)
        C=np.zeros((G,P,P))
        for t in range(T-1):C[:,t*q:(t+1)*q,t*q:(t+1)*q]=self.vv[:,t+1]*u[:,t+1,None,None]
        z=(self.vy[:,1:]*u[:,1:,None]).reshape(G,P)
        S=C-np.transpose(B,(0,2,1))@Ai@B
        h=z-(np.transpose(B,(0,2,1))@Ai@r[:,:,None])[:,:,0]
        m=self.mass*d;target=self.vmean*m[:,:,None]
        # Historical weighted target uses unweighted within-cell covariate means.
        result=(d,Ai,r,B,S,h,target,m)
        self._cache=(tuple(a),result)
        return result

    def _identified_margins(self, a, omit=None, details=False):
        """Reduce untreated cells by QR and test each requested mean for estimability."""
        G, T, q = len(self.states), len(self.times), self.v.shape[1]
        k = (G+T-1)*q
        if not hasattr(self, "_qr_cells"):
            self._qr_cells = []
            order = np.argsort(self._cell, kind="stable")
            for rows in np.split(order, np.flatnonzero(np.diff(self._cell[order]))+1):
                s, t = self._rowstate[rows[0]], self._rowtime[rows[0]]
                xy = np.column_stack((self.v[rows], self.y[rows]))*np.sqrt(self.w[rows,None])
                self._qr_cells.append((s,t,np.linalg.qr(xy,mode="r")))
        blocks, outcomes, owners, targets, treated = [], [], [], [], []
        for s, t, qr in self._qr_cells:
            if s == omit:
                continue
            cols = s*q + np.arange(q)
            if t:
                cols = np.r_[cols, (G+t-1)*q + np.arange(q)]
            if a[s] > 0 and self.times[t] >= a[s]:
                target = np.zeros(k)
                target[cols] = np.tile(self.vmean[s,t], len(cols)//q)
                targets.append(target)
                treated.append((s,t))
            else:
                block = np.zeros((len(qr),k))
                block[:,cols] = np.tile(qr[:,:q],(1,len(cols)//q))
                blocks.append(block)
                outcomes.extend(qr[:,q])
                owners.extend([s]*len(qr))
        if not blocks or not treated:
            raise InferenceError("undefined_estimand","no treated or untreated cells")
        z = np.vstack(blocks)
        u, sigma, vt = np.linalg.svd(z,full_matrices=False)
        # QR compression preserves singular values, but reduces the row count.
        # Apply the tolerance of the original individual-level regression.
        nobs = len(self.y) if omit is None else int(np.sum(self._rowstate != omit))
        use = sigma > sigma[0]*max(1e-12,max(nobs,k)*np.finfo(float).eps)
        basis = vt[use]
        inv = (basis.T/sigma[use])@u[:,use].T
        b = inv@outcomes
        target = np.array(targets)
        error = target - (target@basis.T)@basis
        ok = np.max(abs(error),axis=1) <= 1e-8*np.maximum(1,np.max(abs(target),axis=1))
        if not ok.any():
            raise InferenceError("undefined_estimand","no estimable treated-cell FLEX margin")
        st, tt = np.array(treated)[ok].T
        h = target[ok]
        weights = self.mass[st,tt]/self.mass[st,tt].sum()
        effects = (self.vmean[st,tt]*self.cell_beta[st,tt]).sum(1)-h@b
        theta = float(weights@effects)
        if not details:
            return theta
        regression = np.bincount(owners,weights=-(weights@h@inv)*(outcomes-z@b),minlength=G)
        composition = np.bincount(st,weights=weights*(effects-theta),minlength=G)
        rank = int(use.sum()+sum(np.linalg.matrix_rank(qr[:,:q]) for s,t,qr in self._qr_cells
                                 if s!=omit and a[s]>0 and self.times[t]>=a[s]))
        return dict(estimate=theta,components=effects,weights=weights,solver="cell_qr",
                    regression=regression,composition=composition,rank=rank)

    def _fast(self,a,omit=None,details=False):
        if self.grouping=='cohort':return self._cohort_fast(a,omit,details)
        try:
            d,Ai,r,B,S,h,target,m=self._blocks(a)
        except InferenceError:
            return self._identified_margins(a,omit,details)
        G,T=len(self.states),len(self.times);q=self.v.shape[1]
        keep=np.ones(G,bool)
        if omit is not None:keep[omit]=False
        mass=m[keep].sum()
        if mass<=0:raise InferenceError('undefined_estimand')
        lhs=S[keep].sum(0);rhs=h[keep].sum(0)
        try:
            gamma=self._solve(lhs,rhs)
        except InferenceError:
            return self._identified_margins(a,omit,details)
        alpha=(Ai@(r-(B@gamma))[:,:,None])[:,:,0]
        tau=self.cell_beta-alpha[:,None,:]-np.concatenate([np.zeros((1,q)),gamma.reshape(T-1,q)])[None,:,:]
        effect=np.sum(self.vmean*tau,axis=2)
        theta=float(np.sum(m[keep]*effect[keep])/mass)
        if not details:return theta
        return dict(estimate=theta,components=effect[d&keep[:,None]],weights=m[d&keep[:,None]]/mass,cell_effects=effect,tau=tau,alpha=alpha,gamma=gamma,lhs=lhs,keep=keep,solver='cell_schur')

    def _reference(self,a,omit=None,details=False):
        if self.grouping=='cohort':return self._cohort_reference(a,omit,details)
        G,T=len(self.states),len(self.times);q=self.v.shape[1]
        keep=np.ones(len(self.y),bool) if omit is None else self._rowstate!=omit
        active=np.unique(self._rowstate[keep]);d=(a[:,None]>0)&(self.times[None,:]>=a[:,None]);cells=np.argwhere(d&(self.n>0))
        cells=np.array([(s,t) for s,t in cells if s!=omit],dtype=int)
        if len(cells)==0:raise InferenceError('undefined_estimand')
        width=(len(active)+T-1+len(cells))*q
        if keep.sum()*width>12_000_000:
            return self._identified_margins(a,omit,details)
        pieces=[(self._rowstate==s)[:,None]*self.v for s in active]
        pieces += [(self._rowtime==t)[:,None]*self.v for t in range(1,T)]
        offset=len(pieces)*q
        pieces += [((self._rowstate==s)&(self._rowtime==t))[:,None]*self.v for s,t in cells]
        Z=np.column_stack(pieces)[keep];y=self.y[keep];w=self.w[keep]
        beta,_,rank,_=np.linalg.lstsq(Z*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)
        if rank<width:
            return self._identified_margins(a,omit,details)
        tau=beta[offset:].reshape(-1,q);means=self.vmean[cells[:,0],cells[:,1]]
        effects=np.sum(means*tau,axis=1);mass=self.mass[cells[:,0],cells[:,1]];weights=mass/mass.sum()
        theta=float(weights@effects)
        if not details:return theta
        l=np.zeros(width);l[offset:]=(weights[:,None]*means).ravel()
        return dict(estimate=theta,components=effects,weights=weights,z=Z,coefficients=beta,contrast=l,rank=rank,keep=keep,cells=cells,tau=tau,solver='pooled_dummy_lstsq')

    def _cohort_blocks(self,a,omit=None):
        """Aggregate cached state-period cross-products to assignment cohorts."""
        G,T=len(self.states),len(self.times);q=self.v.shape[1];P=(T-1)*q
        keep=np.ones(G,bool)
        if omit is not None:keep[omit]=False
        if not np.any(a[keep]==0):
            raise InferenceError('cohort_requires_never_treated','cohort grouping currently requires never-treated states')
        groups=np.unique(a[keep]);K=len(groups)
        group_of_state=np.full(G,-1,int)
        group_of_state[keep]=np.searchsorted(groups,a[keep])
        vv=np.zeros((K,T,q,q));vy=np.zeros((K,T,q));mass=np.zeros((K,T));n=np.zeros((K,T))
        rawsum=np.zeros((K,T,q));vsum=np.zeros((K,T,q))
        state_raw=self.vmean*self.n[:,:,None]
        for k,g in enumerate(groups):
            ix=keep&(a==g)
            vv[k]=self.vv[ix].sum(0);vy[k]=self.vy[ix].sum(0)
            mass[k]=self.mass[ix].sum(0);n[k]=self.n[ix].sum(0)
            rawsum[k]=state_raw[ix].sum(0);vsum[k]=self.vsum[ix].sum(0)
        means=(vsum/np.maximum(mass[:,:,None],1)) if self.weighted else (rawsum/np.maximum(n[:,:,None],1))
        cell_ranks=np.linalg.matrix_rank(vv)
        cell_beta=(np.linalg.pinv(vv,rcond=1e-13)@vy[:,:,:,None])[:,:,:,0]
        d=(groups[:,None]>0)&(self.times[None,:]>=groups[:,None]);u=~d
        A=(vv*u[:,:,None,None]).sum(1);r=(vy*u[:,:,None]).sum(1)
        if np.any(np.linalg.matrix_rank(A,tol=1e-10)<q):
            raise InferenceError('rank_deficient','cohort untreated nuisance is unidentified')
        Ai=np.linalg.inv(A)
        B=np.transpose(vv[:,1:]*u[:,1:,None,None],(0,2,1,3)).reshape(K,q,P)
        C=np.zeros((K,P,P))
        for t in range(T-1):C[:,t*q:(t+1)*q,t*q:(t+1)*q]=vv[:,t+1]*u[:,t+1,None,None]
        z=(vy[:,1:]*u[:,1:,None]).reshape(K,P)
        S=C-np.transpose(B,(0,2,1))@Ai@B
        h=z-(np.transpose(B,(0,2,1))@Ai@r[:,:,None])[:,:,0]
        m=mass*d;target=means*m[:,:,None]
        return dict(d=d,Ai=Ai,r=r,B=B,S=S,h=h,target=target,m=m,means=means,
                    cell_beta=cell_beta,cell_ranks=cell_ranks,groups=groups,
                    group_of_state=group_of_state,keep_states=keep,vv=vv,vy=vy)

    def _cohort_fast(self,a,omit=None,details=False):
        b=self._cohort_blocks(a,omit);q=self.v.shape[1];T=len(self.times)
        mass=b['m'].sum()
        if mass<=0:raise InferenceError('undefined_estimand')
        lhs=b['S'].sum(0);rhs=b['h'].sum(0);gamma=self._solve(lhs,rhs)
        alpha=(b['Ai']@(b['r']-(b['B']@gamma))[:,:,None])[:,:,0]
        tau=b['cell_beta']-alpha[:,None,:]-np.concatenate([np.zeros((1,q)),gamma.reshape(T-1,q)])[None,:,:]
        effect=np.sum(b['means']*tau,axis=2);observed=b['d']&(b['m']>0)
        theta=float(np.sum(b['m']*effect)/mass)
        if not details:return theta
        cells=np.argwhere(observed)
        return dict(estimate=theta,components=effect[observed],weights=b['m'][observed]/mass,
                    cell_effects=effect,tau=tau,alpha=alpha,gamma=gamma,lhs=lhs,
                    keep=b['keep_states'],solver='cohort_cell_schur',group_values=b['groups'],
                    cells=cells,cell_labels=[(float(b['groups'][g]),float(self.times[t])) for g,t in cells])

    def _cohort_reference(self,a,omit=None,details=False):
        G,T=len(self.states),len(self.times);q=self.v.shape[1]
        keep_states=np.ones(G,bool)
        if omit is not None:keep_states[omit]=False
        if not np.any(a[keep_states]==0):
            raise InferenceError('cohort_requires_never_treated','cohort grouping currently requires never-treated states')
        keep=keep_states[self._rowstate];groups=np.unique(a[keep_states]);K=len(groups)
        group_of_state=np.full(G,-1,int);group_of_state[keep_states]=np.searchsorted(groups,a[keep_states])
        rowgroup=group_of_state[self._rowstate]
        d=(groups[:,None]>0)&(self.times[None,:]>=groups[:,None])
        cells=[]
        for g,t in np.argwhere(d):
            if np.any(keep&(rowgroup==g)&(self._rowtime==t)):cells.append((g,t))
        cells=np.asarray(cells,dtype=int)
        if len(cells)==0:raise InferenceError('undefined_estimand')
        width=(K+T-1+len(cells))*q
        if keep.sum()*width>12_000_000:raise InferenceError('oracle_size_limit')
        pieces=[(rowgroup==g)[:,None]*self.v for g in range(K)]
        pieces += [(self._rowtime==t)[:,None]*self.v for t in range(1,T)]
        offset=len(pieces)*q
        pieces += [((rowgroup==g)&(self._rowtime==t))[:,None]*self.v for g,t in cells]
        Z=np.column_stack(pieces)[keep];y=self.y[keep];w=self.w[keep]
        beta,_,rank,_=np.linalg.lstsq(Z*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)
        tau=beta[offset:].reshape(-1,q);means=[];mass=[]
        for g,t in cells:
            ix=keep&(rowgroup==g)&(self._rowtime==t)
            means.append(np.average(self.v[ix],axis=0,weights=self.w[ix]) if self.weighted else self.v[ix].mean(0))
            mass.append(self.w[ix].sum())
        means=np.asarray(means);mass=np.asarray(mass);effects=np.sum(means*tau,axis=1);weights=mass/mass.sum()
        theta=float(weights@effects)
        if not details:return theta
        l=np.zeros(width);l[offset:]=(weights[:,None]*means).ravel()
        return dict(estimate=theta,components=effects,weights=weights,z=Z,coefficients=beta,
                    contrast=l,rank=rank,keep=keep,cells=cells,tau=tau,solver='cohort_pooled_dummy_lstsq',
                    group_values=groups,cell_labels=[(float(groups[g]),float(self.times[t])) for g,t in cells])

    def _estimate(self,a,engine,omit,procedure,observed):
        if procedure!='replace':raise NotImplementedError('Add is intentionally unsupported for FLEX')
        return self._reference(a,omit) if engine=='reference' else self._fast(a,omit)
    def fit(self,*,engine='fast'):
        if engine not in ('fast','reference'):raise ValueError('invalid engine')
        return self._reference(self.assignment,details=True) if engine=='reference' else self._fast(self.assignment,details=True)
    def _subtract(self,value):
        out=copy.copy(self);d=(self.assignment[self._rowstate]>0)&(self.times[self._rowtime]>=self.assignment[self._rowstate])
        out.y=self.y-value*d;out._compile();return out
    def native_cluster(self,*,null=0.,engine='fast'):
        # Official unconditional margins treats covariate means as sampled.
        if self.grouping=='cohort':
            raise InferenceError('native_cohort_grouping_unsupported','cohort point estimates are supported; native cohort standard errors are not certified')
        if not np.all(self.w==self.w[0]):
            raise InferenceError('native_nonuniform_weights_unsupported','official aweight unconditional margins requires separate certification; point and resampling weights remain supported')
        G=len(self.states);N=len(self.y);a=self.assignment
        if G<2:raise InferenceError("insufficient_clusters")
        f=self._reference(a,details=True) if engine=='reference' else self._fast(a,details=True)
        theta=f['estimate']
        if f['solver']=='cell_qr':
            score=f['regression']
            rank=f['rank']
        elif engine=='reference':
            Z=f['z'];b=f['coefficients']
            influence=f['contrast']@np.linalg.pinv(Z*np.sqrt(self.w[:,None]),rcond=1e-13)
            scores=np.sqrt(self.w)*(self.y-Z@b)*influence
            effect=np.zeros(N)
            for (s,t),tau in zip(f['cells'],f['tau']):
                ix=(self._rowstate==s)&(self._rowtime==t);effect[ix]=self.v[ix]@tau
            rank=f['rank']
            score=np.bincount(self._rowstate,weights=scores,minlength=G)
        else:
            d,Ai,r,B,S,h,target,m=self._blocks(a)
            mass=m.sum();ha=target.sum(1)/mass;hg=target[:,1:].sum(0).ravel()/mass
            delta=self._solve(f['lhs'],hg-(np.transpose(B,(0,2,1))@Ai@ha[:,:,None])[:,:,0].sum(0))
            first_global=h-(S@f['gamma'])
            score=-first_global@delta # local normal equations are zero.
            effect=np.sum(self.v*f['tau'][self._rowstate,self._rowtime],axis=1)
            rank=(G+len(self.times)-1)*self.v.shape[1]+int(self.cell_ranks[d].sum())
        if f['solver']=='cell_qr':
            composition=f['composition']
        else:
            target_rows=(a[self._rowstate]>0)&(self.times[self._rowtime]>=a[self._rowstate])
            composition=np.bincount(self._rowstate,weights=self.w*target_rows*(effect-theta),minlength=G)/np.sum(self.w*target_rows)
        # Official regress + margins unconditional finite-sample multiplier.
        total=score+composition
        if N<=rank:raise InferenceError("insufficient_residual_degrees_of_freedom")
        multiplier=G/(G-1)*(N-1)/(N-rank)
        se=float(np.linalg.norm(total)*np.sqrt(multiplier));stat=(theta-null)/se if se else float('nan')
        return dict(estimate=theta,native_cluster_method='flexdid_unconditional_cluster',native_cluster_se=se,native_cluster_stat=stat,native_cluster_df=G-1,native_cluster_p=float(2*student_t.sf(abs(stat),G-1)),native_cluster_distribution='t',cluster_scores=total,regression_scores=score,composition_scores=composition,finite_sample_multiplier=multiplier,regression_rank=rank,status='ok' if se>0 else 'zero_standard_error')

