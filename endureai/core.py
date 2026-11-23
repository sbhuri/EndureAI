from __future__ import annotations
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
from .summarizer import _tokenize, summarize_text

ROOT = Path(__file__).resolve().parents[1]
QUERIES = {'power': 'power generation redundant outage alternate source repairs',
           'access': 'accessible shelter facility barriers accessibility',
           'capacity': 'planning shelter capacity space needs population'}

def validate(s, *, allow_unavailable_open=False):
    if not isinstance(s, dict) or s.get('synthetic') is not True:
        raise ValueError('Only explicitly synthetic scenarios are accepted.')
    for key in ('id', 'hazard'):
        if not isinstance(s.get(key), str) or not s[key].strip(): raise ValueError(f'Missing {key}')
    def number(v): return type(v) is int and 0 <= v <= 1000000
    if not number(s.get('demand')): raise ValueError('Demand must be a nonnegative integer.')
    if type(s.get('power_required')) is not bool: raise ValueError('power_required must be boolean')
    if not isinstance(s.get('shelters'), list) or not 1 <= len(s['shelters']) <= 12:
        raise ValueError('Supply 1–12 shelters.')
    ids=set()
    for x in s['shelters']:
        if not isinstance(x,dict): raise ValueError('Invalid shelter')
        if not isinstance(x.get('id'),str) or not x['id'] or x['id'] in ids: raise ValueError('Unique shelter IDs required')
        ids.add(x['id'])
        if not all(number(x.get(k)) for k in ('capacity','assigned')): raise ValueError('Invalid shelter counts')
        for k in ('open','available','backup_power','accessible'):
            if type(x.get(k)) is not bool: raise ValueError(f'Missing boolean {k}')
        if x['open'] and not x['available'] and not allow_unavailable_open: raise ValueError('Open shelter cannot be unavailable')
    return copy.deepcopy(s)

def rank_bm25(passages, tokens, query):
    """Score tokens using the prototype's BM25 weighting and deterministic ties."""
    terms = set(_tokenize(query))
    total = len(passages)
    average = sum(map(len, tokens)) / max(total, 1)
    idf = {term: math.log(1 + (total - df + .5) / (df + .5))
           for term in sorted(terms) if (df := sum(term in group for group in tokens))}
    rankings = []
    for page, words in zip(passages, tokens, strict=True):
        normalizer = 1.2 * (.25 + .75 * len(words) / max(average, 1))
        score = 0.0
        for term, weight in idf.items():
            frequency = words.count(term)
            if frequency:
                score += weight * frequency * 2.2 / (frequency + normalizer)
        if score > 0:
            rankings.append((round(score, 5), page))
    return sorted(rankings, key=lambda item: (-item[0], item[1]["page"], item[1]["id"]))


class ResearchAgent:
    """BM25 retrieval over immutable page-labelled FEMA text; extractive generation."""
    def __init__(self, corpus=None):
        self.path=Path(corpus) if corpus is not None else ROOT/'data/corpus.json'
        self.data=json.loads(self.path.read_text())
        self.hash=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.pages=self.data['pages']
        self.tokens=[_tokenize(p['text']) for p in self.pages]
    def retrieve(self, query, k=3):
        ranked = rank_bm25(self.pages, self.tokens, query)[:k]
        return [{**page, "score": score, "url": self.data["retrieved_url"] + f"#page={page['page']}",
                 "summary": summarize_text(page["text"], query, 2)} for score, page in ranked]
    def run(self):
        return {topic:self.retrieve(query) for topic,query in QUERIES.items()}

class AuditorAgent:
    """Recompute constraints independently; never trust planner's totals."""
    def run(self,s,evidence,baseline=None, *, require_topic_evidence=True):
        findings=[]
        def add(code,detail,topic=None):
            findings.append({'code':code,'detail':detail,'basis':'synthetic constraint' if topic is None else 'prototype rule informed by FEMA guidance',
                             'citations':[p['id'] for p in evidence.get(topic,[])[:1]] if topic else []})
        if baseline is not None:
            fixed=('id','synthetic','hazard','demand','power_required')
            inventory=lambda plan: {x['id']:{k:v for k,v in x.items() if k not in ('open','assigned')} for x in plan['shelters']}
            if any(s.get(k)!=baseline.get(k) for k in fixed) or inventory(s)!=inventory(baseline):
                add('INVENTORY_CHANGED','Planner changed scenario constraints or inventory')
        usable=0
        for x in s['shelters']:
            if x['open'] and x['available'] and x['accessible'] and (not s['power_required'] or x['backup_power']): usable+=x['capacity']
            if x['open'] and not x['available']: add('UNAVAILABLE_OPEN',x['id'])
            if x['assigned'] and not x['open']: add('CLOSED_ALLOCATION',x['id'])
            if x['assigned']>x['capacity']: add('OVER_CAPACITY',x['id'])
            if x['open'] and not x['accessible']: add('ACCESS_GAP',x['id'],'access')
            if x['open'] and s['power_required'] and not x['backup_power']: add('POWER_GAP',x['id'],'power')
        shortage=max(0,s['demand']-usable)
        if shortage: add('CAPACITY_SHORTAGE',f'{shortage} spaces short')
        assigned=sum(x['assigned'] for x in s['shelters'])
        if assigned!=s['demand']: add('DEMAND_MISMATCH',f'{assigned} assigned versus {s["demand"]} demand')
        for topic in QUERIES if require_topic_evidence else ():
            if not evidence.get(topic): add('EVIDENCE_MISSING',topic)
        return {'findings':findings,'usable_capacity':usable,'shortage':shortage,'passes':not findings}

class PlanningAgent:
    """Bounded combinatorial search, not LLM Tree-of-Thought."""
    def propose(self,s,limit=32):
        eligible=[x for x in s['shelters'] if x['available'] and x['accessible'] and (not s['power_required'] or x['backup_power'])]
        count=0
        for size in range(len(eligible)+1):
            for selected in itertools.combinations(eligible,size):
                if count>=limit: return
                candidate=copy.deepcopy(s); ids={x['id'] for x in selected}; remaining=s['demand']
                for x in candidate['shelters']:
                    x['open']=x['id'] in ids
                    x['assigned']=min(remaining,x['capacity']) if x['open'] else 0
                    remaining-=x['assigned']
                count+=1
                yield candidate

class CoordinatorAgent:
    def __init__(self,corpus=None,max_candidates=32):
        if type(max_candidates) is not int or not 1<=max_candidates<=4096: raise ValueError('Invalid search bound')
        self.research=ResearchAgent(corpus);self.planning=PlanningAgent();self.auditor=AuditorAgent();self.bound=max_candidates
    def run(self,scenario):
        s=validate(scenario); evidence=self.research.run(); original=self.auditor.run(s,evidence)
        trace=[{'agent':'Coordinator','event':'input validated'}, {'agent':'Research','event':'retrieved FEMA pages'},
               {'agent':'Auditor','event':'original audited','codes':[f['code'] for f in original['findings']]}]
        candidates=[]; selected=None
        if not original['passes']:
            for candidate in self.planning.propose(s,self.bound):
                audit=self.auditor.run(candidate,evidence,baseline=s)
                candidates.append({'scenario':candidate,'audit':audit})
                trace.append({'agent':'Planning/Auditor','candidate':len(candidates),'passes':audit['passes'],'shortage':audit['shortage']})
                if audit['passes']:
                    selected=candidate;break
        else: selected=s
        result={'scenario_id':s['id'],'mode':'deterministic-extractive','status':'REVIEW_REQUIRED' if selected is None else ('REVISION_FOR_REVIEW' if candidates else 'CHECKS_PASSED'),
                'human_review_required':True,'original':original,'selected_plan':selected,'candidates':candidates,'trace':trace,
                'evidence':evidence,'corpus_sha256':self.research.hash,'max_candidates':self.bound,
                'scope':'Synthetic shelter allocation only. No operational approval or FEMA compliance certification.'}
        return result

def render_report(r):
    lines=['# EndureAI audit', '',r['scope'],'',f"Scenario: {r['scenario_id']} | Status: {r['status']}",
           f"Original usable capacity: {r['original']['usable_capacity']}; shortage: {r['original']['shortage']}.",'','## Original findings']
    lines += [f"- {f['code']}: {f['detail']} ({f['basis']}; {', '.join(f['citations']) or 'arithmetic/input data'})" for f in r['original']['findings']] or ['- No findings within the implemented checks.']
    lines+=['','## Proposed allocation']
    if r['selected_plan']:
        lines += [f"- {x['id']}: {x['assigned']} people; open={x['open']}" for x in r['selected_plan']['shelters']]
    else: lines+=['No passing candidate found within the search bound. Escalate to a human planner.']
    lines+=['','## Retrieved guidance (extractive; historical 2010 source)']
    for topic,pages in r['evidence'].items():
        for p in pages[:1]: lines += [f"- {topic}: [{p['id']}]({p['url']}) — {p['summary']}"]
    lines+=['','All proposed changes require human review. Capacity/access/power flags are synthetic inputs, not facility assessments.']
    return '\n'.join(lines)+'\n'
