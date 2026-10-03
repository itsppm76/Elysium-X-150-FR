"""Prepare candidate causal OASST branches. No invented emotional labels."""
import gzip,json,hashlib,re,collections
from pathlib import Path
from huggingface_hub import hf_hub_download
REPO='OpenAssistant/oasst1';NAME='2023-04-12_oasst_ready.trees.jsonl.gz'
# Pin actual resolved repository revision and preserve license at acquisition.
def ingest(root,revision):
 path=hf_hub_download(REPO,NAME,repo_type='dataset',revision=revision)
 license_path=hf_hub_download(REPO,'LICENSE',repo_type='dataset',revision=revision)
 root=Path(root);root.mkdir(parents=True,exist_ok=True)
 (root/'OASST_LICENSE').write_bytes(Path(license_path).read_bytes())
 stems=['feel','lonely','grief','relationship','friend','afraid','anxious','sad','love','family','stress','support','angry','hurt','emotion']
 accepted=[];seen=set();counts=collections.Counter();trees=0;messages=0
 def walk(node,branch):
  if node.get('deleted') or not node.get('text'):return
  branch=branch+[node]
  children=node.get('replies',[])
  if children:
   for child in children:yield from walk(child,branch)
  else:yield branch
 for line in gzip.open(path,'rt'):
  obj=json.loads(line);trees+=1
  prompt=obj.get('prompt',obj)
  for branch in walk(prompt,[]):
   messages+=len(branch)
   # Scope candidate to English/Hindi, preserving source lang, no fabricated translations.
   if len(branch)<4 or any(x.get('lang') not in ['en','hi'] for x in branch):continue
   text=' '.join(x['text'] for x in branch)
   if not any(w in text.lower() for w in stems):continue # transparent EN relevance heuristic; not Hindi eligibility judgment
   if re.search(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}',text):continue
   ids=[x['message_id'] for x in branch];h=hashlib.sha256('|'.join(ids).encode()).hexdigest()
   if h in seen:continue
   seen.add(h);tree_id=obj.get('message_tree_id') or branch[0].get('message_tree_id') or ids[0]
   # Split trees, never sibling branches or overlapping causal windows across splits.
   bucket=int(hashlib.sha256(str(tree_id).encode()).hexdigest()[:8],16)%100
   split='train' if bucket<80 else 'dev' if bucket<90 else 'test'
   r={'id':'oasst-'+h[:20],'source_repo':REPO,'source_revision':revision,'source_license':'Apache-2.0','source_tree':tree_id,'source_message_ids':ids,'synthetic':False,'human_emotion_gold':False,'label_source':'not_labelled','dimensions':None,'quality_review_passed':False,'language':[x['lang'] for x in branch],'split':split,'turns':[{'speaker':'A' if x['role']=='prompter' else 'B','text':x['text']} for x in branch],'target_turn':len(branch)-1,'target_speaker':'A' if branch[-1]['role']=='prompter' else 'B'}
   accepted.append(r);counts[split]+=1
 with open(root/'oasst_candidates.jsonl','w') as f:
  for r in accepted:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 result={'source_file':NAME,'source_revision':revision,'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'trees_read':trees,'branch_message_visits':messages,'accepted_candidate_branches':len(accepted),'splits':dict(counts),'emotion_labels':0,'status':'candidate only; quality review and annotation required; English keyword filter underselects Hindi'}
 json.dump(result,open(root/'oasst_acquisition.json','w'),indent=2)
 return result
