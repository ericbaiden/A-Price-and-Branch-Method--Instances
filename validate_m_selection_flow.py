"""Analytical small networks distinguish cost-driven rejection and capacity."""
from types import SimpleNamespace
import networkx as nx
import run_alns_benchmark as bench  # establishes the existing local license
from m_selection_flow import MSelectionFlow


def instance(count):
    return SimpleNamespace(tasks=[SimpleNamespace(task_id=i, data_volume=1) for i in range(count)],
                           sources={i:f's{i}' for i in range(count)},
                           sinks={i:'sink' for i in range(count)})


def arc(graph, u, v, cost, cap=10, resource=False):
    graph.add_edge(u,v,weight=cost,cap=cap,
                   arc_type='downlink_arc' if resource else 'test_nonresource')


def validate():
    reports = []
    # Capacity can serve all tasks, but M=200 economically rejects a cost-300 path.
    graph = nx.DiGraph(); arc(graph,'s0','sink',300)
    flow = MSelectionFlow(instance(1),graph,threads=1)
    p,_ = flow.service_frontier_endpoint()
    assert (p['maximum_completed'],p['critical_penalty'],p['smallest_strict_integer_penalty']) == (1,300,301)
    assert flow.solve('penalty',200)['completed'] == 0
    assert flow.solve('penalty',301)['completed'] == 1
    reports.append({'case':'economic_rejection',**p}); flow.close()
    # Competition for one cheap link: F(1)=50; F(2)=230; threshold is 180.
    graph=nx.DiGraph()
    for i in range(2): arc(graph,f's{i}','u',0)
    arc(graph,'u','sink',50,1,True); arc(graph,'s0','sink',180)
    flow=MSelectionFlow(instance(2),graph,threads=1)
    p,_=flow.service_frontier_endpoint()
    assert (p['maximum_completed'],p['minimum_cost_at_maximum_service'],p['critical_penalty']) == (2,230,180)
    assert flow.solve('penalty',179)['completed']==1
    assert flow.solve('penalty',181)['completed']==2
    reports.append({'case':'rerouting_marginal_cost',**p}); flow.close()
    # Three tasks compete for capacity 1.9 -> one integral task; M cannot fix it.
    graph=nx.DiGraph()
    for i in range(3): arc(graph,f's{i}','u',0)
    arc(graph,'u','sink',40,1.9,True)
    flow=MSelectionFlow(instance(3),graph,threads=1)
    p,_=flow.service_frontier_endpoint()
    assert (p['maximum_completed'],p['critical_penalty']) == (1,40)
    assert flow.solve('penalty',10000)['completed']==1
    reports.append({'case':'capacity_ceiling',**p}); flow.close()
    # Disconnected sources: no fabricated completion or invalid K-1 stage.
    graph=nx.DiGraph(); graph.add_nodes_from(['s0','sink'])
    flow=MSelectionFlow(instance(1),graph,threads=1)
    p,_=flow.service_frontier_endpoint()
    assert p['maximum_completed']==0
    assert flow.solve('penalty',200)['completed']==0
    reports.append({'case':'no_feasible_task',**p}); flow.close()
    return {'status':'passed','analytical_cases':reports}


if __name__=='__main__':
    import json
    print(json.dumps(validate(),indent=2))
