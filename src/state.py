"""Deterministic product state indexing, not psychological uniqueness."""
from math import comb
import json,hashlib,math

def subset_rank(ids,n=150):
    ids=tuple(sorted(ids))
    if len(set(ids))!=len(ids) or any(type(i)!=int or i<1 or i>n for i in ids):raise ValueError('Invalid dimension IDs')
    # Colexicographic combinatorial number system; separate cardinality required.
    return {'n':n,'r':len(ids),'rank':sum(comb(i-1,j+1) for j,i in enumerate(ids)),'active_ids':list(ids)}

def state(dimensions,thresholds,version,speaker):
    known={x['id']:x['strength'] for x in dimensions}
    if len(known)!=len(dimensions):raise ValueError('Duplicate IDs')
    if any(type(k)!=int or not 1<=k<=150 or type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for k,v in known.items()):raise ValueError('Invalid strengths')
    if any(k not in thresholds for k in known):raise ValueError('Missing threshold')
    a=[k for k,v in known.items() if v>=thresholds[k]]
    canonical=json.dumps({'taxonomy_version':version,'speaker':speaker,'known_dimensions':sorted(known.items())},sort_keys=True,separators=(',',':'))
    return {'subset':subset_rank(a),'canonical_state':canonical,'state_hash':hashlib.sha256(canonical.encode()).hexdigest(),'unknown_ids':[i for i in range(1,151) if i not in known]}
