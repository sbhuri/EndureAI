import copy,json,unittest
from types import SimpleNamespace as NS
from endureai.core import ROOT
from endureai.llm import (CoordinatorAgent,ModelFailure,OpenAIBackend,RESEARCH,PLAN,AUDIT,DECISION,
                         check_schema,verify_claims,materialize)

class ScriptedBackend:
    """Test double; never represented as a live model."""
    live=False
    def __init__(self,bad_quote=False,bad_plan=False,accept_invalid=False,revise=False,fail=False):
        self.records=[];self.bad_quote=bad_quote;self.bad_plan=bad_plan
        self.accept_invalid=accept_invalid;self.revise=revise;self.fail=fail;self.round=0
    def ask(self,role,instruction,payload,schema):
        self.records.append({'agent':role,'test_double':True})
        if self.fail:raise ModelFailure('Simulated transport failure')
        if role=='Research':
            page=next(p for p in payload['pages'] if p['page']==19)
            self.citation={'page_id':page['id'],'quote':'fabricated quotation not in document' if self.bad_quote else page['text'][:150]}
            return {'summary':'Test synthesis','findings':[{'statement':'Test claim','citations':[self.citation]}],'limitations':['Test only']}
        if role=='Planning':
            self.round+=1
            bad=self.bad_plan or (self.revise and self.round==1)
            return {'alternatives':[{'label':'Test plan','rationale':'Test rationale','allocations':[
                {'shelter_id':'A','open':True,'assigned':400 if bad else 250},
                {'shelter_id':'B','open':not bad,'assigned':0 if bad else 150}]}],'limitations':[]}
        if role=='Auditor':
            return {'acceptable':True,'explanation':'Test reviewer','concerns':[],
                    'evidence_review':[{'statement':'Test evidence review','citations':[self.citation]}]}
        return {'action':'accept_for_review' if payload['passing_candidate_indices'] or self.accept_invalid else ('revise' if self.revise else 'escalate'),'explanation':'Test decision'}

class Tests(unittest.TestCase):
    def setUp(self):self.s=json.loads((ROOT/'data/scenarios/winter_shortage.json').read_text())
    def test_full_four_agent_flow(self):
        b=ScriptedBackend();r=CoordinatorAgent(b).run(self.s)
        self.assertEqual(r['mode'],'llm-mocked-test');self.assertEqual(r['status'],'REVISION_FOR_REVIEW')
        self.assertEqual([x['agent'] for x in b.records],['Research','Planning','Auditor','Coordinator'])
        self.assertEqual(sum(x['assigned'] for x in r['selected_plan']['shelters']),400)
    def test_fabricated_citation_blocks_workflow(self):
        r=CoordinatorAgent(ScriptedBackend(bad_quote=True)).run(self.s)
        self.assertIsNone(r['selected_plan']);self.assertIn('Citation',r['error'])
    def test_llm_approval_cannot_override_constraints(self):
        r=CoordinatorAgent(ScriptedBackend(bad_plan=True,accept_invalid=True)).run(self.s)
        self.assertIsNone(r['selected_plan']);self.assertIn('without a passing audit',r['error'])
    def test_revision_loop(self):
        r=CoordinatorAgent(ScriptedBackend(revise=True)).run(self.s)
        self.assertEqual(len(r['planning_rounds']),2);self.assertIsNotNone(r['selected_plan'])
    def test_round_bound(self):
        r=CoordinatorAgent(ScriptedBackend(revise=True),max_rounds=1).run(self.s)
        self.assertEqual(len(r['planning_rounds']),1);self.assertIsNone(r['selected_plan'])
    def test_failure_is_recorded_without_fallback(self):
        r=CoordinatorAgent(ScriptedBackend(fail=True)).run(self.s)
        self.assertEqual(r['status'],'REVIEW_REQUIRED');self.assertIn('error',r)
    def test_no_mutation(self):
        before=copy.deepcopy(self.s);CoordinatorAgent(ScriptedBackend()).run(self.s);self.assertEqual(self.s,before)
    def test_unknown_shelter(self):
        with self.assertRaises(ModelFailure):materialize(self.s,{'allocations':[{'shelter_id':'invented','open':True,'assigned':400}]})
    def test_negative_assignment(self):
        with self.assertRaises(ModelFailure):materialize(self.s,{'allocations':[{'shelter_id':'A','open':True,'assigned':-1},{'shelter_id':'B','open':False,'assigned':0}]})
    def test_strict_schema(self):
        with self.assertRaises(ModelFailure):check_schema({'action':'approve','explanation':'x'},DECISION)
        with self.assertRaises(ModelFailure):check_schema({'action':'revise','explanation':'x','extra':1},DECISION)
        with self.assertRaises(ModelFailure):check_schema(True,{'type':'integer'})
    def test_unknown_citation(self):
        with self.assertRaises(ModelFailure):verify_claims([{'statement':'x','citations':[{'page_id':'fake','quote':'a long enough quote'}]}],[])
    def test_response_adapter(self):
        captured={}
        def create(**kw):
            captured.update(kw)
            return NS(id='test-response',model='test-model',status='completed',usage=NS(model_dump=lambda:{'input_tokens':5,'output_tokens':8}),output_text=json.dumps({'action':'escalate','explanation':'test'}),output=[])
        b=OpenAIBackend('test-model',client=NS(responses=NS(create=create)),max_calls=1)
        self.assertEqual(b.ask('Coordinator','test',{},DECISION)['action'],'escalate')
        self.assertTrue(captured['text']['format']['strict']);self.assertFalse(captured['store'])
        self.assertEqual(b.records[0]['usage']['input_tokens'],5)
        with self.assertRaises(ModelFailure):b.ask('Coordinator','test',{},DECISION)
    def test_incomplete_api_response(self):
        def create(**kw):return NS(id='r',model='m',status='incomplete',usage=None,output_text='',output=[])
        b=OpenAIBackend('test',client=NS(responses=NS(create=create)))
        with self.assertRaises(ModelFailure):b.ask('Coordinator','test',{},DECISION)
        self.assertEqual(b.records[0]['status'],'incomplete')
    def test_refusal(self):
        def create(**kw):return NS(id='r',model='m',status='completed',usage=None,output_text='',output=[NS(type='message',content=[NS(type='refusal')])])
        b=OpenAIBackend('test',client=NS(responses=NS(create=create)))
        with self.assertRaises(ModelFailure):b.ask('Coordinator','test',{},DECISION)
    def test_credentials_not_in_error_log(self):
        def create(**kw):raise RuntimeError('secret-test-token')
        b=OpenAIBackend('test',client=NS(responses=NS(create=create)))
        with self.assertRaises(ModelFailure):b.ask('Coordinator','test',{},DECISION)
        self.assertNotIn('secret-test-token',json.dumps(b.records))
    def test_real_sdk_request_with_mock_http_transport(self):
        import httpx
        from openai import OpenAI
        captured={}
        def handler(request):
            captured.update(json.loads(request.content))
            return httpx.Response(200,json={
                'id':'resp_test','object':'response','created_at':1,'status':'completed','model':'test-model',
                'output':[{'type':'message','id':'msg_test','role':'assistant','status':'completed',
                    'content':[{'type':'output_text','text':json.dumps({'action':'escalate','explanation':'test'}),'annotations':[]}]}],
                'usage':{'input_tokens':10,'output_tokens':5,'total_tokens':15,
                         'input_tokens_details':{'cached_tokens':0},'output_tokens_details':{'reasoning_tokens':0}}})
        client=OpenAI(api_key='test-only-placeholder',http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        try:
            b=OpenAIBackend('test-model',client=client)
            self.assertEqual(b.ask('Coordinator','test',{},DECISION)['action'],'escalate')
            self.assertEqual(captured['text']['format']['type'],'json_schema')
            self.assertEqual(b.records[0]['usage']['total_tokens'],15)
        finally:client.close()
if __name__=='__main__':unittest.main()
