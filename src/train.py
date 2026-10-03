"""Train a candidate from reviewed records. No unsupported SOTA/accuracy claims."""
import os,json,time,random,hashlib,math
from pathlib import Path
import torch,numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig,get_cosine_schedule_with_warmup
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from window import window
BASE='Qwen/Qwen2.5-1.5B-Instruct';SEED=150;MAX_SEQ=2048;MAX_ANSWER=512
ROOT=Path(os.environ.get('X150_ROOT','/content/drive/MyDrive/Elysium-X-150-FR'))
DATA=ROOT/'data/train.jsonl';OUT=ROOT/'candidate';OUT.mkdir(parents=True,exist_ok=True)
if not torch.cuda.is_available():raise RuntimeError('No GPU; stop instead of buying compute')
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
tax=json.load(open(ROOT/'taxonomy.json'));rows=[json.loads(l) for l in open(DATA)]
assert len(rows)>=1000,'Scale/quality gate: require 1000 accepted reviewed train records, not a 30-dialogue pilot'
assert all(r.get('quality_review_passed') and r.get('split')=='train' for r in rows),'Only reviewed train records'
# Gold provenance remains synthetic/teacher unless independent human labels exist.
tok=AutoTokenizer.from_pretrained(BASE);tok.pad_token=tok.eos_token
prepared=[];reject=[]
for r in rows:
 try:
  p,meta=window(tok,r,tax,MAX_SEQ,MAX_ANSWER)
  a=tok.encode(json.dumps({'dimensions':r['dimensions']},separators=(',',':'))+tok.eos_token,add_special_tokens=False)
  assert len(a)<=MAX_ANSWER,'answer too long'
  # Do not train evidence referring to a removed history turn.
  assert all(not(set(d['evidence_turns']) & set(meta['removed_turn_ids'])) for d in r['dimensions']),'evidence trimmed'
  prepared.append({'input_ids':p+a,'labels':[-100]*len(p)+a,'meta':meta,'id':r['id']})
 except (ValueError,AssertionError) as e:reject.append({'id':r['id'],'reason':str(e)})
assert len(prepared)>=1000,'Post-window gate failed'
json.dump(reject,open(OUT/'window_rejections.json','w'),indent=2)
config={'base':BASE,'seed':SEED,'max_seq':MAX_SEQ,'answer_budget':MAX_ANSWER,'n_records':len(prepared),'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'taxonomy_sha256':hashlib.sha256((ROOT/'taxonomy.json').read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name(0),'epochs':1,'microbatch':1,'accumulation':16,'lr':1e-4,'lora_r':16,'label_caveat':'sparse weak supervision; omitted labels not independently observed negatives'}
json.dump(config,open(OUT/'run_config.json','w'),indent=2)
model=AutoModelForCausalLM.from_pretrained(BASE,quantization_config=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16),device_map='auto')
model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=True)
model=get_peft_model(model,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
model.config.use_cache=False
optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
random.shuffle(prepared);steps=math.ceil(len(prepared)/16);scheduler=get_cosine_schedule_with_warmup(optimizer,max(1,steps//20),steps);scaler=torch.amp.GradScaler('cuda');model.train();start=time.time();losses=[];global_step=0
for i,r in enumerate(prepared):
 ids=torch.tensor([r['input_ids']],device=model.device);labels=torch.tensor([r['labels']],device=model.device)
 with torch.autocast('cuda',dtype=torch.float16):raw=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss;loss=raw/16
 scaler.scale(loss).backward();losses.append(raw.item())
 if (i+1)%16==0 or i+1==len(prepared):
  scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(optimizer);scaler.update();optimizer.zero_grad();scheduler.step();global_step+=1
  if global_step%25==0 or i+1==len(prepared):
   checkpoint=OUT/f'checkpoint-{global_step}';checkpoint.mkdir(exist_ok=True);model.save_pretrained(checkpoint);tok.save_pretrained(checkpoint)
   torch.save({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'ordered_record_ids':[x['id'] for x in prepared],'record_index':i,'global_step':global_step},checkpoint/'resume_state.pt')
   json.dump(config,open(checkpoint/'run_config.json','w'),indent=2);print('CHECKPOINT',global_step,'mean_train_loss',sum(losses[-400:])/len(losses[-400:]),'seconds',round(time.time()-start),flush=True)
model.save_pretrained(OUT/'adapter');tok.save_pretrained(OUT/'adapter')
json.dump({'training_complete':True,'n_train':len(prepared),'optimizer_steps':global_step,'seconds':time.time()-start,'mean_train_loss':sum(losses)/len(losses),'evaluation':'not yet run'},open(OUT/'train_result.json','w'),indent=2)
