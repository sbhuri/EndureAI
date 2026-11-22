from pathlib import Path
import copy,json
root=Path(__file__).resolve().parents[1]
def shelter(i,capacity,assigned,**kw):return dict(id=i,capacity=capacity,assigned=assigned,open=True,available=True,backup_power=True,accessible=True,**kw)
base={'id':'winter_shortage','synthetic':True,'hazard':'winter storm','demand':400,'power_required':True,'shelters':[shelter('A',250,250),dict(id='B',capacity=150,assigned=0,open=False,available=True,backup_power=True,accessible=True)]}
cases=[]
def add(name,mutate,codes,status,shortage):
 s=copy.deepcopy(base);s['id']=name;mutate(s);cases.append({'scenario':s,'expected_codes':sorted(codes),'expected_status':status,'expected_shortage':shortage})
add('winter_shortage',lambda s:None,['CAPACITY_SHORTAGE','DEMAND_MISMATCH'],'REVISION_FOR_REVIEW',150)
add('insufficient_inventory',lambda s:s['shelters'][1].update(available=False),['CAPACITY_SHORTAGE','DEMAND_MISMATCH'],'REVIEW_REQUIRED',150)
add('balanced',lambda s:s.update(demand=250),[],'CHECKS_PASSED',0)
add('over_allocation',lambda s:s['shelters'][0].update(assigned=400),['OVER_CAPACITY','CAPACITY_SHORTAGE'],'REVISION_FOR_REVIEW',150)
add('power_gap',lambda s:s['shelters'][0].update(backup_power=False),['POWER_GAP','CAPACITY_SHORTAGE','DEMAND_MISMATCH'],'REVIEW_REQUIRED',400)
add('access_gap',lambda s:s['shelters'][0].update(accessible=False),['ACCESS_GAP','CAPACITY_SHORTAGE','DEMAND_MISMATCH'],'REVIEW_REQUIRED',400)
add('closed_allocation',lambda s:s['shelters'][0].update(open=False),['CLOSED_ALLOCATION','CAPACITY_SHORTAGE','DEMAND_MISMATCH'],'REVISION_FOR_REVIEW',400)
add('zero_demand',lambda s:(s.update(demand=0),s['shelters'][0].update(assigned=0)),[],'CHECKS_PASSED',0)
add('surplus_assignment',lambda s:s.update(demand=200),['DEMAND_MISMATCH'],'REVISION_FOR_REVIEW',0)
add('power_not_required',lambda s:(s.update(demand=250,power_required=False),s['shelters'][0].update(backup_power=False)),[],'CHECKS_PASSED',0)
(root/'data/scenarios').mkdir(exist_ok=True)
for c in cases:(root/'data/scenarios'/f"{c['scenario']['id']}.json").write_text(json.dumps(c['scenario'],indent=2))
(root/'data/evaluation_cases.json').write_text(json.dumps(cases,indent=2))
