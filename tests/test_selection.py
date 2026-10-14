import json
import unittest
from pathlib import Path
from skillrae.cli import load_input
from skillrae.mechanism_policies_v2 import select_hq_community_exact
from skillrae.compiler import closed_requirements

class SelectionContract(unittest.TestCase):
    def setUp(self):
        self.raw=json.loads((Path(__file__).resolve().parents[1]/'examples/selection.json').read_text())
    def test_preserve_support_and_add_requirement_at_same_budget(self):
        data=load_input(self.raw);decision=select_hq_community_exact(data)
        self.assertEqual(decision.selected_skill_ids,('s1','s3'))
        self.assertEqual(set(closed_requirements(data,decision.selected_skill_ids)),{'r1','r2'})
        self.assertEqual(decision.audit['critical_community_quota'],2)
    def test_no_accepted_gain_retains_retrieval_baseline(self):
        self.raw['support_paths'][2]['status']='UNRESOLVED'
        decision=select_hq_community_exact(load_input(self.raw))
        self.assertEqual(decision.selected_skill_ids,('s1','s2'))
    def test_critical_support_cannot_be_traded_for_unsupported_quantity(self):
        self.raw['top_k']=1
        decision=select_hq_community_exact(load_input(self.raw))
        self.assertEqual(decision.selected_skill_ids,('s1',))

if __name__=='__main__':unittest.main()
