import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description='EndureAI LLM-powered synthetic shelter auditing')
    p.add_argument('scenario',type=Path)
    p.add_argument('--output',type=Path,default=Path('runs/audit'))
    p.add_argument('--mode',choices=['llm','baseline'],default='llm')
    p.add_argument('--model',help='Responses API model; otherwise use OPENAI_MODEL')
    p.add_argument('--max-rounds',type=int,default=2)
    p.add_argument('--max-candidates',type=int,default=32,help='Baseline only')
    a=p.parse_args()
    try:
        scenario=json.loads(a.scenario.read_text())
        if a.mode=='llm':
            from .llm import OpenAIBackend,CoordinatorAgent,render_report
            backend=OpenAIBackend(model=a.model,max_calls=1+a.max_rounds*5)
            r=CoordinatorAgent(backend,max_rounds=a.max_rounds).run(scenario)
        else:
            from .core import CoordinatorAgent,render_report
            r=CoordinatorAgent(max_candidates=a.max_candidates).run(scenario)
    except (ValueError,KeyError,OSError,ImportError) as e:p.error(str(e))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.with_suffix('.json').write_text(json.dumps(r,indent=2))
    a.output.with_suffix('.md').write_text(render_report(r))
    print(r['status'],f"shortage={r['original']['shortage']}",a.output.with_suffix('.md'))
    if r.get('error'):raise SystemExit(2)
if __name__=='__main__':main()
