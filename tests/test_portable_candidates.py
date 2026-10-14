"""OCU candidate construction checks source identity."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from skillrae.graph_v2 import build_graph_package

class PortableCandidates(unittest.TestCase):
    def test_one_skill_and_source_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pool=root/'pool';skill=pool/'diagnose';skill.mkdir(parents=True)
            source=skill/'SKILL.md'
            source.write_text('# Diagnose\n\n## Procedure\n\nRead fault codes and write a diagnostic report.\n')
            graph=build_graph_package(pool,root)
            self.assertTrue(graph.subunits)
            anchor=graph.subunits[0]
            profiles={'status':'PASS','model':'gpt-5.5','graph_package_sha256':graph.package_sha256,
                'profiles':[{'status':'CERTIFIED','skill_id':anchor.skill_id,
                'support_subunit_ids':[anchor.subunit_id],'concept_id':'constructed-anchor',
                'canonical_label':'Diagnose'}]}
            profile_file=root/'profiles.json';profile_file.write_text(json.dumps(profiles))
            cmd=[sys.executable,'-m','skillrae.cli','ocu-candidates',str(pool),str(profile_file),'--root',str(root)]
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            candidates=json.loads(result.stdout)['candidates'];self.assertTrue(candidates)
            self.assertTrue(candidates[0]['support_spans'])
            # The same source parser accepts the configured construction model,
            # and rejects a profile from a different configuration.
            profiles['model']='gpt-5.5'
            profile_file.write_text(json.dumps(profiles))
            as_cmd=cmd[:3]+['--preset','agentskillos']+cmd[3:]
            configured=subprocess.run(as_cmd,capture_output=True,text=True)
            self.assertEqual(configured.returncode,0,configured.stderr)
            profiles['model']='different-construction-model'
            profile_file.write_text(json.dumps(profiles))
            mismatched=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(mismatched.returncode,0)
            self.assertIn('construction model',mismatched.stderr)
            profiles['model']='gpt-5.5'
            profiles['normalized_output_sha256']='fixture-profile-identity'
            profile_file.write_text(json.dumps(profiles))
            from skillrae.capability_units_v3 import freeze_capability_unit_protocol
            rankings=root/'rankings.json'
            rankings.write_text(json.dumps({'labels_rewards_or_trajectories_loaded':False,
                'task_rows':[{'task_id':'new-task','c0_ranking':[anchor.skill_id]}]}))
            protocol=freeze_capability_unit_protocol(root,rankings,profile_file,root/'protocol',pool_dir=pool)
            self.assertEqual(protocol['model'],'gpt-5.5')
            source.write_text(source.read_text()+'\nChanged source.\n')
            rejected=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn('graph hash',rejected.stderr)

    def test_example_provenance_resolves(self):
        root=Path(__file__).resolve().parents[1]
        raw=json.loads((root/'examples/selection.json').read_text())
        for row in raw['requirements']+raw['subunits']:
            p=row['provenance'];source=root/p['source_file']
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),p['source_sha256'])
            lines=source.read_text().splitlines()
            self.assertEqual('\n'.join(lines[p['line_start']-1:p['line_end']]),p['exact_text'])
