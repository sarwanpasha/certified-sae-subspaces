import numpy as np
rng=np.random.default_rng(0)
viol=dict(t1lo=0,t1hi=0,t2lo=0,t2hi=0,t1eq=0,t2eq=0,greedy=0)
for trial in range(300):
    d=rng.integers(6,30); c=rng.integers(2,5)
    A=rng.standard_normal((d,d)); S=A@A.T+0.1*np.eye(d)
    Sxy=rng.standard_normal((d,c)); Syy=np.eye(c)*(np.linalg.norm(Sxy)**2/ c)*5  # ensure valid joint cov (approx)
    # make joint cov valid: Syy >= Syx S^-1 Sxy
    Syy=Sxy.T@np.linalg.solve(S,Sxy)+np.eye(c)
    beta=np.linalg.solve(S,Sxy); tr=np.trace(Syy)
    R2full=np.trace(Sxy.T@beta)/tr
    k=rng.integers(1,d); Q,_=np.linalg.qr(rng.standard_normal((d,k))); P=Q@Q.T
    R2in=np.trace(Sxy.T@Q@np.linalg.solve(Q.T@S@Q,Q.T@Sxy))/tr
    gap=(R2full-R2in)*tr; r=np.linalg.norm((np.eye(d)-P)@beta)**2
    ev=np.linalg.eigvalsh(S)
    viol['t1lo']+= gap < ev[0]*r-1e-8; viol['t1hi']+= gap > ev[-1]*r+1e-8
    # out
    Qo=np.linalg.svd(np.eye(d)-P)[0][:,:d-k]
    R2out=np.trace(Sxy.T@Qo@np.linalg.solve(Qo.T@S@Qo,Qo.T@Sxy))/tr
    ro=np.linalg.norm((np.eye(d)-P)@Sxy)**2
    lmin=np.linalg.eigvalsh(Qo.T@S@Qo)[0]
    viol['t2hi']+= R2out*tr > ro/lmin+1e-8; viol['t2lo']+= R2out*tr < ro/ev[-1]-1e-8
    # equality cases
    Qb,_=np.linalg.qr(np.hstack([beta,rng.standard_normal((d,1))]))
    R2b=np.trace(Sxy.T@Qb@np.linalg.solve(Qb.T@S@Qb,Qb.T@Sxy))/tr
    viol['t1eq']+= abs(R2b-R2full)>1e-8
    Qc,_=np.linalg.qr(Sxy); Qco=np.linalg.svd(np.eye(d)-Qc@Qc.T)[0][:,:d-Qc.shape[1]]
    R2c=np.trace(Sxy.T@Qco@np.linalg.solve(Qco.T@S@Qco,Qco.T@Sxy))/tr
    viol['t2eq']+= abs(R2c)>1e-8
    # greedy (OLS / forward regression) vs exhaustive best-k on a small dictionary
    m=10; D=rng.standard_normal((d,m)); D/=np.linalg.norm(D,axis=0); T=Sxy/np.linalg.norm(Sxy)
    kk=3
    def g(Sset):
        if not Sset: return 0
        B=D[:,Sset]; Pq=B@np.linalg.pinv(B); return np.linalg.norm(Pq@T)**2
    import itertools
    opt=max(g(list(s)) for s in itertools.combinations(range(m),kk))
    Sg=[]
    for _ in range(kk):
        Sg.append(max([j for j in range(m) if j not in Sg],key=lambda j:g(Sg+[j])))
    C=D.T@D; gam=min(np.linalg.eigvalsh(C[np.ix_(s,s)])[0] for s in itertools.combinations(range(m),2*kk))
    viol['greedy']+= g(Sg) < (1-np.exp(-gam))*opt-1e-9
print(viol)
