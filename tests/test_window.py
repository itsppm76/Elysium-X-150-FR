import sys,unittest,json
import os;sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
from window import window
class Tok:
 def apply_chat_template(self,messages,**kw):
  self.messages=messages;return list(range(len(messages[0]['content'])+len(messages[1]['content'])))
class Tests(unittest.TestCase):
 def record(self):return {'turns':[{'speaker':'A','text':'old'*100},{'speaker':'B','text':'target'},{'speaker':'A','text':'FUTURE MUST NOT LEAK'}],'target_turn':1,'target_speaker':'B'}
 def test_causal(self):
  t=Tok();p,m=window(t,self.record(),{'dimensions':[]},1000,10);self.assertNotIn('FUTURE',str(t.messages))
 def test_trim_preserves_target(self):
  t=Tok();p,m=window(t,self.record(),{'dimensions':[]},700,10);self.assertEqual(m['removed_turn_ids'],[0]);self.assertIn('target',t.messages[-1]['content']);self.assertIn('Appraise',t.messages[0]['content'])
 def test_oversize_rejected(self):
  with self.assertRaises(ValueError):window(Tok(),self.record(),{'dimensions':[]},20,10)
if __name__=='__main__':unittest.main()
