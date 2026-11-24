import importlib.util
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import protocol as P
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('abc_test_tokenizer',ROOT/'experiments/tokenizers.py')
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)


class ProtocolTests(unittest.TestCase):
    def test_real_tokenizer_layout_and_independent_truth(self):
        tok=T.FactoredVocabTokenizer(128,['ADD'],23,3)
        for c in P.make_cases(2):
            for stage in (0,1,19):
                f=P.frame(c,stage);lay=P.render(f,c['names'],f['constants'])
                rows=tok.encode_tokens(lay['tokens']);vals,depth=P.evaluate(f['parents'],f['constants'])
                # Evaluate the actual serialized equations, independently of parent indices.
                solved={}
                for equation in ' '.join(lay['tokens']).split(' <EQ_SEP> '):
                    left,name=equation.split(' = ')
                    operands=left.split(' ADD ')
                    solved[name]=sum(int(x) if x.isdigit() else solved[x] for x in operands)%23
                self.assertEqual([solved[n] for n in c['names']],vals)
                self.assertLessEqual(len(rows),256)
                self.assertEqual(len(set(lay['nodes'])),32)
                self.assertFalse(set(lay['nodes']) & set(lay['task_rows']))
                for i,p in enumerate(lay['nodes']):
                    self.assertEqual(rows[p][1],tok.variable_tok2idx[c['names'][i]])
                    self.assertEqual(rows[p][3],tok.value_tok2idx['EMPTY'])

    def test_boundary_uses_model_commit_and_preserves_error(self):
        c=P.make_cases(1)[0];f=P.frame(c,1)
        actual=[(v+3)%23 for v in P.reference_program(c,1)[0]['output']]
        inputs=P.emitted_inputs(f,actual)
        self.assertEqual(inputs[:f['width']],actual)
        self.assertNotEqual(inputs[:f['width']],P.reference_program(c,2)[1]['inputs'][:f['width']])
        self.assertIsNone(P.emitted_inputs(f,[None]*f['width']))
        with self.assertRaises(ValueError):P.evaluate(f['parents'],[None]*6)

    def test_program_length_and_dependency_not_just_independent_problems(self):
        for c in P.make_cases(1):
            rr=P.reference_program(c,20)
            self.assertEqual(rr[-1]['cumulative_logical_nodes'],526)
            self.assertEqual(rr[-1]['executed_operations'],520)
            self.assertGreater(rr[-1]['logical_depth'],rr[0]['logical_depth'])
            self.assertGreater(rr[-1]['output_ancestor_operations'],26)
            for s in range(1,20):
                w=P.STRATA[c['stratum']][0]
                self.assertEqual(rr[s]['inputs'][:w],rr[s-1]['output'])

    def test_reassignment_returns_to_original_and_only_edits_inputs(self):
        c=P.make_cases(1)[0];f=P.frame(c,0);plans=P.update_inputs(f['constants'])
        self.assertEqual(plans[-1]['inputs'],f['constants'])
        old=P.render(f,c['names'],f['constants'])
        for plan in plans:
            new=P.render(f,c['names'],plan['inputs'])
            changed={i for i,(x,y) in enumerate(zip(old['tokens'],new['tokens'])) if x!=y}
            self.assertTrue(changed<=set(old['constants']))
            old=new

    def test_finite_stability_and_value_identity_are_separate(self):
        gold=[[0,1,1,4]]
        tr=[[[0,1,1,4]],[[0,1,1,5]],[[0,9,1,4]],[[0,9,1,4]]]
        m=P.measures(tr,gold,[0],[0])
        self.assertTrue(m['nodes_value_all']);self.assertFalse(m['nodes_first4_all'])
        self.assertEqual(m['nodes_value_stable'],3);self.assertTrue(m['nodes_value_reerror'])
        m=P.measures(tr[:3],gold,[0],[0])
        self.assertFalse(m['nodes_value_all']);self.assertIsNone(m['nodes_value_stable'])


if __name__=='__main__':unittest.main()
