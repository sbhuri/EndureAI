import datetime,hashlib,json,platform,time
from pathlib import Path
from .core import ROOT,CoordinatorAgent,render_report

def main():
 cases=json.loads((ROOT/'data/evaluation_cases.json').read_text());agent=CoordinatorAgent();rows=[];tp=fp=fn=0
 out=ROOT/'evidence';out.mkdir(exist_ok=True)
 for c in cases:
  start=time.perf_counter();r=agent.run(c['scenario']);elapsed=time.perf_counter()-start
  actual={f['code'] for f in r['original']['findings']};expected=set(c['expected_codes'])
  tp+=len(actual&expected);fp+=len(actual-expected);fn+=len(expected-actual)
  ok=actual==expected and r['status']==c['expected_status'] and r['original']['shortage']==c['expected_shortage']
  rows.append({'id':r['scenario_id'],'passed':ok,'expected_codes':sorted(expected),'actual_codes':sorted(actual),'expected_status':c['expected_status'],'actual_status':r['status'],'shortage':r['original']['shortage'],'seconds':round(elapsed,6),'candidates':len(r['candidates'])})
  (out/f"{r['scenario_id']}.json").write_text(json.dumps(r,indent=2));(out/f"{r['scenario_id']}.md").write_text(render_report(r))
 retrieval=[]
 for q,pages in [('power generation redundant outage alternate source repairs',{19,21}),('Planning Shelter Capacity',{17}),('facility inaccessible features accessible before use',{19})]:
  hits=agent.research.retrieve(q,5);retrieval.append({'query':q,'relevant_pages':sorted(pages),'retrieved_pages':[p['page'] for p in hits],'hit_at_5':bool(pages&{p['page'] for p in hits})})
 result={'run_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':platform.python_version(),'mode':'deterministic-extractive','cases':rows,'passed':sum(x['passed'] for x in rows),'total':len(rows),'finding_code_precision':tp/(tp+fp) if tp+fp else None,'finding_code_recall':tp/(tp+fn) if tp+fn else None,'retrieval':retrieval,'corpus_sha256':agent.research.hash,'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'endureai').glob('*.py'))},'limitations':'Authored synthetic regression cases, not independent real-world validation. No LLM, live retrieval, or operational accuracy evaluated.'}
 (out/'evaluation.json').write_text(json.dumps(result,indent=2))
 lines=['# Measured evaluation','',f"Run: {result['run_at']}",f"Mode: {result['mode']}; Python {result['python']}",'',f"Scenario checks: {result['passed']}/{result['total']}",f"Finding-code precision: {result['finding_code_precision']}; recall: {result['finding_code_recall']}",'','| Scenario | Match | Status | Original shortage | Candidates | Seconds |','|---|---|---|---:|---:|---:|']
 lines += [f"| {x['id']} | {x['passed']} | {x['actual_status']} | {x['shortage']} | {x['candidates']} | {x['seconds']} |" for x in rows]
 lines += ['',f"Known-page retrieval hit@5: {sum(x['hit_at_5'] for x in retrieval)}/{len(retrieval)}",'',result['limitations']]
 (out/'EVALUATION.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
 if result['passed']!=len(rows) or not all(x['hit_at_5'] for x in retrieval):raise SystemExit(1)
if __name__=='__main__':main()
