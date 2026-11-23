"""Run real API evaluation; persist every case even if an agent fails."""
import argparse,json,time
from datetime import datetime,timezone
from pathlib import Path
from .core import ROOT
from .llm import CoordinatorAgent,OpenAIBackend,render_report

def main():
    p=argparse.ArgumentParser(description='Paid live evaluation, up to 12 model calls per scenario')
    p.add_argument('--model');p.add_argument('--limit',type=int,default=1)
    p.add_argument('--output',type=Path,default=ROOT/'runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    a=p.parse_args()
    if not 1<=a.limit<=10:p.error('limit must be 1–10')
    cases=json.loads((ROOT/'data/evaluation_cases.json').read_text())[:a.limit]
    try:backend=OpenAIBackend(a.model,max_calls=a.limit*12)
    except ValueError as e:p.error(str(e))
    a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for c in cases:
        start=time.perf_counter();r=CoordinatorAgent(backend).run(c['scenario'])
        name=r['scenario_id'];(a.output/f'{name}.json').write_text(json.dumps(r,indent=2))
        (a.output/f'{name}.md').write_text(render_report(r))
        # These labels describe deterministic policy outcomes, not independent semantic correctness.
        expected=c['expected_status'];status_match=r['status']==expected
        execution_ok=not r.get('error') and not any(call.get('error_type') for call in r['calls'])
        audits=[x.get('audit',{}) for x in r['candidates']]
        rows.append({'id':name,'expected_status':expected,'actual_status':r['status'],'status_match':status_match,
          'execution_succeeded':execution_ok,'error':r.get('error'),'seconds':round(time.perf_counter()-start,4),
          'calls':len(r['calls']),'rejected_candidates':sum(not x.get('passes',False) for x in audits),
          'selected_plan_present':r['selected_plan'] is not None,
          'input_tokens':sum((x.get('usage') or {}).get('input_tokens',0) for x in r['calls']),
          'output_tokens':sum((x.get('usage') or {}).get('output_tokens',0) for x in r['calls'])})
        summary={'mode':'llm-live','requested_model':backend.model,'cases':rows,
            'completed_at':datetime.now(timezone.utc).isoformat(),'status_matches':sum(x['status_match'] for x in rows),
            'execution_successes':sum(x['execution_succeeded'] for x in rows),'total':len(rows),
            'semantic_review':'Not independently assessed. Quote membership is not proof of entailment.',
            'limits':'Synthetic development fixtures; no operational validation. Status matches on failed runs are not successful evaluations.'}
        (a.output/'evaluation.json').write_text(json.dumps(summary,indent=2))
        print(name,r['status'],'execution_succeeded=',execution_ok,flush=True)
    print('Evidence:',a.output)
    if any(not row['execution_succeeded'] for row in rows):raise SystemExit(2)
if __name__=='__main__':main()
