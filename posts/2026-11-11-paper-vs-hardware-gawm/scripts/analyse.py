import json,glob,numpy as np,os
R=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','expected','runs')
def load(m,s):
    o=[]
    for k in range(3):
        d=json.load(open(f'{R}/{m}_{s}_s{k}/heldout_l1.json')); o.append(d)
    eps=sorted(o[0]); return eps,np.array([[x[e]['l1'] for e in eps] for x in o]).mean(0),np.array([o[0][e]['identity_null'] for e in eps])
res={}
for s in ['train172','quarter_seed0']:
    eps,g,nl=load('gawm',s); _,n,_=load('nowm',s); res[s]=(g,n,nl)
rng=np.random.default_rng(0); E=len(eps); idx=rng.integers(0,E,(2000,E))
def ci(x): return np.percentile(x,[2.5,97.5])
out={}
for s,(g,n,nl) in res.items():
    d=g-n; b=d[idx].mean(1)
    out[s]=dict(gawm=g.mean(),nowm=n.mean(),null=nl.mean(),diff=d.mean(),ci=list(ci(b)))
    print(s,{k:(np.round(v,4).tolist()) for k,v in out[s].items()})
gap=lambda s:(res[s][1]-res[s][0])
gd=gap('quarter_seed0')-gap('train172'); b=gd[idx].mean(1)
out['B2_gap_q']=gap('quarter_seed0').mean();out['B2_gap_full']=gap('train172').mean()
out['B2_gapdiff']=gd.mean();out['B2_ci']=list(ci(b))
print('gap quarter',out['B2_gap_q'],'gap full',out['B2_gap_full'],'diff',gd.mean(),ci(b))
# seed spread
for m in ['gawm','nowm']:
  for s in ['train172','quarter_seed0']:
    out[f'seedmeans_{m}_{s}']=[float(np.mean([v['l1'] for v in json.load(open(f'{R}/{m}_{s}_s{k}/heldout_l1.json')).values()])) for k in range(3)]
    print(m,s,out[f'seedmeans_{m}_{s}'])
json.dump(out,open(os.path.join(os.path.dirname(R),'analysis.json'),'w'),indent=1,default=float)
