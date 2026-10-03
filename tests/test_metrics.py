import sys,unittest,json
import os;sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
from metrics import parse,evaluate
class Tests(unittest.TestCase):
 def r(self):return {'known_ids':[1,2],'dimensions':[{'id':1,'strength':1}],'label_source':'synthetic','target_turn':3}
 def test_perfect_mask(self):
  p=json.dumps({'dimensions':[{'id':1,'strength':1,'evidence_turns':[3]}]});m=evaluate([self.r()],[p],{1:.5,2:.5});self.assertEqual(m['micro_f1_known'],1);self.assertEqual(len(m['unmeasured_ids']),148)
 def test_invalid_counts_failure(self):
  m=evaluate([self.r()],['bad'],{1:.5,2:.5});self.assertEqual(m['strict_schema_valid_rate'],0);self.assertEqual(m['micro_f1_known'],0);self.assertEqual(m['masked_strength_mae_invalid_penalty_1'],1)
 def test_future_invalid(self):
  p=json.dumps({'dimensions':[{'id':1,'strength':1,'evidence_turns':[4]}]});m=evaluate([self.r()],[p],{1:.5,2:.5});self.assertEqual(m['strict_schema_valid_rate'],0)
 def test_duplicates_invalid(self):
  p=json.dumps({'dimensions':[{'id':1,'strength':1,'evidence_turns':[0]}]*2});self.assertFalse(parse(p)[1])
if __name__=='__main__':unittest.main()
