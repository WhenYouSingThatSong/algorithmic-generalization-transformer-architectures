import importlib.util
import sys
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import core as P


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.nodes=[2,6,10,16,22]
        self.parents={2:[0],6:[4],10:[2],16:[2,6],22:[6,10]}
        self.constants={0:3,4:7}

    def test_repeated_and_alternating_return_to_original(self):
        old,_=P.evaluate(self.nodes,self.parents,self.constants)
        schedules=P.schedules(self.nodes,self.parents,self.constants,9)
        self.assertEqual(set(schedules),{'same_input','alternating_inputs'})
        for seq in schedules.values():
            plans=list(P.assignment_plans(self.nodes,self.parents,self.constants,9,seq))
            self.assertEqual(plans[-1]['new_truth'],old)
            self.assertEqual(plans[0]['old_truth'],old)
            for before,after in zip(plans,plans[1:]):
                self.assertEqual(before['new_truth'],after['old_truth'])
        self.assertEqual(self.constants,{0:3,4:7})

    def test_reuse_is_dead_and_consumer_reads_new_version(self):
        seq=P.reuse_plans(self.nodes,self.parents,self.constants,4)
        self.assertEqual(len(seq),2)
        a,b=seq
        self.assertIn(a['target'],b['new_parents'])
        used={q for ps in self.parents.values() for q in ps}
        self.assertNotIn(a['target'],used);self.assertNotIn(b['target'],used)
        self.assertEqual(a['logical_version'],f"{a['target']}:v1")
        self.assertEqual(b['cumulative_logical_nodes'],len(self.nodes)+2)
        self.assertEqual(b['new_truth'][b['target']],sum(b['new_truth'][q] for q in b['new_parents'])%23)

    def test_cycle_and_undefined_stop(self):
        with self.assertRaises(ValueError):P.evaluate([1,2],{1:[2],2:[1]}, {})
        with self.assertRaises(ValueError):P.evaluate([1],{1:[4]}, {})

    def test_modulo_cancellation_excluded_from_changed(self):
        nodes=[2,4];parents={2:[0],4:[2]*23}
        plan=next(P.assignment_plans(nodes,parents,{0:3},0,[(2,1)]))
        self.assertIn(4,plan['affected']);self.assertNotIn(4,plan['changed'])

    def test_expression_uses_left_operands_right_result(self):
        syntax={'VARIABLE':0,'=':3,'<EQ_SEP>':4}
        raw=[[0,2],[1,0],[0,6],[3,0],[0,10]]
        self.assertEqual(P.expression_positions(raw,4,[0,2],syntax),[0,2])
        with self.assertRaises(ValueError):P.expression_positions(raw,4,[2,0],syntax)

    def test_suffix_needs_two_outputs(self):
        self.assertIsNone(P.stable_round([False,False,True]))
        self.assertEqual(P.stable_round([True,False,True,True]),3)
        gold=[[0,0,0,4]];outs=[[[0,1,0,4]],[[0,0,0,4]]]
        self.assertEqual(P.trace_counts(outs,gold,[0],'last_two'),dict(n=1,value_n=1,first4_n=0))

    def test_c_readout_and_function_qualification_are_separate(self):
        gold=[[0,0,0,4],[0,1,0,13],[0,2,0,9]]
        state=[gold[0],[0,1,0,14],gold[2]]
        flags=P.c_context(state,gold,[0,1,0,24],gold[1],gold,[1,2,3],1,2,{2:[1,3]},24)
        self.assertTrue(flags['u_value_correct']);self.assertFalse(flags['q_value_correct'])
        self.assertTrue(flags['donor_pre_empty']);self.assertTrue(flags['other_parents_value_correct'])
        self.assertEqual(len(P.c_probe_grid()),36)


if __name__=='__main__':unittest.main()
