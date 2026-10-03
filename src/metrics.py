"""Masked evaluation. Weak/synthetic labels must never be reported as human gold."""
import math,json

def parse(s):
 try:
  j=json.loads(s);assert set(j)=={'dimensions'} and isinstance(j['dimensions'],list)
  assert len(j['dimensions'])<=150
  ids=set()
  for x in j['dimensions']:
   assert set(x)=={'id','strength','evidence_turns'}
   assert type(x['id'])==int and 1<=x['id']<=150 and x['id'] not in ids;ids.add(x['id'])
   assert type(x['strength']) in (int,float) and math.isfinite(x['strength']) and 0<=x['strength']<=1
   assert isinstance(x['evidence_turns'],list) and all(type(t)==int and t>=0 for t in x['evidence_turns'])
  return j,True
 except (ValueError,TypeError,KeyError,AssertionError):return None,False

def evaluate(records,predictions,thresholds):
 if len(records)!=len(predictions):raise ValueError('Prediction/record count mismatch')
 counts={i:{'tp':0,'fp':0,'fn':0,'support':0,'known_count':0} for i in range(1,151)}
 valid=0;errors=[];similarities=[];sources={}
 for r,s in zip(records,predictions):
  if 'known_ids' not in r or r.get('dimensions') is None:raise ValueError('Missing gold mask/labels')
  gold={x['id']:x['strength'] for x in r['dimensions']};known=set(r['known_ids'])
  assert set(gold)<=known,'Unknown dimensions cannot be gold'
  j,ok=parse(s)
  if ok and any(t>r['target_turn'] for x in j['dimensions'] for t in x['evidence_turns']):ok=False
  valid+=int(ok);pred={x['id']:x['strength'] for x in j['dimensions']} if ok else {}
  source=r['label_source'];sources[source]=sources.get(source,0)+1
  gset=set();pset=set()
  for i in known:
   if i not in thresholds:raise ValueError('Threshold missing')
   g=gold.get(i,0);p=pred.get(i,0);gp=g>=thresholds[i];pp=p>=thresholds[i]
   c=counts[i];c['known_count']+=1;c['support']+=int(gp);c['tp']+=int(gp and pp);c['fp']+=int(not gp and pp);c['fn']+=int(gp and not pp)
   # Invalid schema is a failure, never silently dropped from denominator.
   errors.append(abs(g-p) if ok else 1.0)
   if gp:gset.add(i)
   if pp:pset.add(i)
  if known:similarities.append(len(gset&pset)/len(gset|pset) if gset|pset else 1.0 if ok else 0.0)
 def f1(tp,fp,fn):return 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
 measured={i:c for i,c in counts.items() if c['known_count']}
 per={i:{**c,'f1':f1(c['tp'],c['fp'],c['fn'])} for i,c in measured.items()}
 tp=sum(c['tp'] for c in measured.values());fp=sum(c['fp'] for c in measured.values());fn=sum(c['fn'] for c in measured.values())
 fs=[c['f1'] for c in per.values() if c['f1'] is not None]
 return {'n':len(records),'strict_schema_valid_count':valid,'strict_schema_valid_rate':valid/len(records) if records else None,'micro_f1_known':f1(tp,fp,fn),'macro_f1_measured':sum(fs)/len(fs) if fs else None,'masked_strength_mae_invalid_penalty_1':sum(errors)/len(errors) if errors else None,'mean_jaccard_known':sum(similarities)/len(similarities) if similarities else None,'per_dimension':per,'unmeasured_ids':[i for i in range(1,151) if i not in measured],'label_sources':sources,'caveat':'These metrics measure the supplied masked reference only. Synthetic/teacher references are not human emotional accuracy.'}
