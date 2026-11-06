import sys,json,gzip,re,random,collections
sys.path.insert(0,'/work/code'); sys.path.insert(0,'/authors/exp8_math/code')
import mathqa
gold={}
for l in open('/work/data/math500.jsonl'):
    d=json.loads(l); gold[d['id']]=d
arms="A_base B_astra C_crd D_random E_nothink F_astra_nothink G_crd_nothink H_crdloc I_clc J_clc_nothink".split()
claimed=dict(zip("ABCDEFGHIJ",[7.4,49.2,16.4,8.6,45.8,44.4,41.0,23.2,53.0,43.0]))
out={'arms':{}};C={}
for a in arms:
    recs=[json.loads(l) for l in gzip.open(f'/work/runs/m500_{a}/records.jsonl.gz','rt')]
    ids=[r['id'] for r in recs]
    mine={};dis=[]
    for r in recs:
        try:
            pred,_=mathqa.extract(r['text']); ok=bool(mathqa.is_correct(pred,gold[r['id']]['answer']))
        except Exception as e: ok=False; print('ERR',repr(e)[:80])
        mine[r['id']]=int(ok)
        if ok!=bool(r['correct']): dis.append(r['id'])
    st={r['id']:int(bool(r['correct'])) for r in recs}
    acc=100*sum(mine.values())/len(mine)
    out['arms'][a]=dict(n=len(recs),unique=len(set(ids)),ids_match_gold=set(ids)==set(gold),finish=dict(collections.Counter(r['finish'] for r in recs)),acc_regraded=acc,acc_stored=100*sum(st.values())/len(st),disagree=len(dis),claimed=claimed[a[0]])
    C[a[0]]=mine
ids=sorted(gold);rng=random.Random(12345)
def boot(x,y):
    d=[C[x][i]-C[y][i] for i in ids];n=len(d);ms=[]
    for _ in range(10000):
        ms.append(sum(d[rng.randrange(n)] for _ in range(n))/n*100)
    ms.sort();return dict(diff=100*sum(d)/n,lo=ms[250],hi=ms[9749])
out['paired']={f'{x}-{y}':boot(x,y) for x,y in ["IA","IB","IH","IE","BE","CA","DA"]}
def norm(s): return re.sub(r'\s+',' ',s).strip()
m500=set(gold);mt={norm(d['problem']) for d in gold.values()}
lk={}
for f in ['extract','dev']:
    rs=[json.loads(l) for l in open(f'/work/data/{f}.jsonl')]
    lk[f]=dict(n=len(rs),id_overlap=sum(r['id'] in m500 for r in rs),text_overlap=sum(norm(r['problem']) in mt for r in rs))
out['leak']=lk
ex={json.loads(l)['id'] for l in open('/work/data/extract.jsonl')}
for p in ['clc','extract']:
    j=json.load(open(f'/work/runs/{p}/extract.json'))
    pid={x['id'] for x in j['pairs']} if 'pairs' in j else set()
    out['leak'][f'runs_{p}_pair_ids']=dict(n=len(pid),outside_extract=len(pid-ex),in_math500=len(pid&m500),keys=list(j)[:12])
json.dump(out,open('/work/audit_independent/audit_part.json','w'),indent=1)
print(json.dumps(out,indent=1))
