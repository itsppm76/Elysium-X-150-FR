import sys,itertools,unittest
import os;sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..','src'))
from state import subset_rank,state
class Tests(unittest.TestCase):
 def test_bijection_small(self):
  for r in range(6):
   ranks=[subset_rank(a,n=5)['rank'] for a in itertools.combinations(range(1,6),r)]
   self.assertEqual(len(ranks),len(set(ranks)))
   self.assertEqual(sorted(ranks),list(range(len(ranks))))
 def test_mask(self):
  s=state([{'id':1,'strength':.7}],{1:.5},'v0','A');self.assertEqual(len(s['unknown_ids']),149);self.assertEqual(s['subset']['active_ids'],[1])
 def test_strength_key(self):
  def f(v):return state([{'id':1,'strength':v}],{1:.5},'v0','A')
  self.assertEqual(f(.6)['subset'],f(.7)['subset']);self.assertNotEqual(f(.6)['canonical_state'],f(.7)['canonical_state'])
 def test_bad(self):
  for ids in [[0],[151],[1,1],[1.0]]:
   with self.assertRaises(ValueError):subset_rank(ids)
if __name__=='__main__':unittest.main()
