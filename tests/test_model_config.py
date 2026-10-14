"""Model routing and request identity checks; no external model calls."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from skillrae.model_config import model_config
from skillrae import semantic_runner

class ModelRouting(unittest.TestCase):
    def test_presets_and_independent_overrides(self):
        from skillrae.model_config import PRESETS
        for preset in PRESETS:
            with patch.dict(os.environ, {'SKILLRAE_MODEL_PRESET':preset}, clear=True):
                self.assertEqual(model_config().payload(), {'construction':'gpt-5.5','binding':'gpt-5.5'})
        with patch.dict(os.environ, {'SKILLRAE_MODEL_PRESET':'skillsbench'}, clear=True):
            self.assertEqual(model_config().payload(), {'construction':'gpt-5.5','binding':'gpt-5.5'})
        with patch.dict(os.environ, {'SKILLRAE_MODEL_PRESET':'agentskillos','SKILLRAE_BINDING_MODEL':'custom-model'}, clear=True):
            self.assertEqual(model_config().payload(), {'construction':'gpt-5.5','binding':'custom-model'})
        with patch.dict(os.environ, {'SKILLRAE_MODEL_PRESET':'typo'}, clear=True):
            with self.assertRaises(ValueError):model_config()

    def test_binding_dispatch_uses_binding_model(self):
        script='''
from unittest.mock import patch
from skillrae import binding_judge, semantic_protocol
assert binding_judge.CONSTRUCTION_MODEL == semantic_protocol.EXACT_MODEL == 'gpt-5.5'
with patch.object(binding_judge, '_semantic_call_json', return_value={}) as call:
    binding_judge._call_json('source', 'call', 'prompt', {}, 64, 'artifact')
    assert call.call_args.kwargs['model'] == 'binding-test-model'
'''
        env={k:v for k,v in os.environ.items() if not k.startswith('SKILLRAE_')}
        result=subprocess.run([sys.executable,'-c',script],env={**env,'SKILLRAE_MODEL_PRESET':'skillsbench','SKILLRAE_BINDING_MODEL':'binding-test-model'},capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_request_model_and_cache_validation(self):
        calls=[]
        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='{"status":"OK"}',model=kwargs['model'],id='test-response')
        client=SimpleNamespace(responses=SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory() as td, patch.object(semantic_runner,'api_client',return_value=(client,'test')):
            root=Path(td);path=root/'call.json'
            args=(root,'test-call','prompt',{'task':'test'},64,path)
            artifact=semantic_runner._call_json(*args,model='gpt-5.5')
            self.assertEqual(calls[0]['model'],'gpt-5.5')
            self.assertEqual(artifact['requested_model'],'gpt-5.5')
            semantic_runner._call_json(*args,model='gpt-5.5')
            self.assertEqual(len(calls),1)
            with self.assertRaises(semantic_runner.SemanticCallError):
                semantic_runner._call_json(*args,model='different-model')
            with self.assertRaises(semantic_runner.SemanticCallError):
                semantic_runner._call_json(root,'test-call','changed prompt',{'task':'test'},64,path,model='gpt-5.5')
            with self.assertRaises(semantic_runner.SemanticCallError):
                semantic_runner._call_json(root,'test-call','prompt',{'task':'other'},64,path,model='gpt-5.5')
            self.assertEqual(len(calls),1)

    def test_provider_model_mismatch_rejected(self):
        client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:SimpleNamespace(output_text='{}',model='wrong-model')))
        with tempfile.TemporaryDirectory() as td, patch.object(semantic_runner,'api_client',return_value=(client,'test')):
            root=Path(td)
            with self.assertRaises(semantic_runner.SemanticCallError):
                semantic_runner._call_json(root,'call','prompt',{},64,root/'call.json',model='gpt-5.5')
            self.assertFalse((root/'call.json').exists())
