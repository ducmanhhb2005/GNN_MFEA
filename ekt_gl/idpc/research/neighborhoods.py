"""Local-search interventions charged to the V2 evaluation tracker."""

from collections import Counter
from dataclasses import replace
from time import perf_counter

import numpy as np

from ekt_gl.idpc.v2.local_search import DirectedPairs, _step, improve as directed_2opt
from .beam import BeamConfig, decode_beam, pair_costs, shortest


def inspect_rebuild(indexes, prefix, suffix, codes, config, counts):
    """Mirror the V2 reconnection and split its combined truncation counter."""
    active, closed = None, frozenset()
    for edge in prefix:
        active, closed = _step(codes[edge.domain], active, closed)
    beam = [(0.0, (), active, closed)]
    remaining = config.max_edge_expansions_per_cut
    beam_pruned = expansion_cap = False
    for ids, weights, domains in indexes:
        children = []
        for partial, path, active, closed in beam:
            visits = min(len(ids), remaining)
            counts["ls_edge_expansions"] += visits
            expansion_cap |= visits < len(ids)
            if not visits:
                break
            remaining -= visits
            blocked = np.zeros(len(codes), dtype=bool)
            if closed:
                blocked[list(closed)] = True
            valid = np.flatnonzero(~blocked[domains[:visits]])
            counts["ls_du_rejects"] += visits - len(valid)
            counts["ls_partial_cost_updates"] += min(len(valid), config.beam_width)
            beam_pruned |= len(valid) > config.beam_width
            for index in valid[:config.beam_width]:
                d, left = _step(int(domains[index]), active, closed)
                children.append((partial+float(weights[index]), path+(int(ids[index]),), d,left))
        children.sort(key=lambda row: row[:2])
        beam_pruned |= len(children) > config.beam_width
        beam = children[:config.beam_width]
        if not beam:
            break
    left_ids,right_ids = tuple(e.id for e in prefix),tuple(e.id for e in suffix)
    complete = []
    for _,path,active,closed in beam:
        if len(path) != len(indexes):
            continue
        for edge in suffix:
            update = _step(codes[edge.domain], active, closed)
            if update is None:
                counts["ls_du_rejects"] += 1
                break
            active,closed = update
        else:
            complete.append(left_ids + path + right_ids)
    complete_cap=len(complete)>config.max_complete_paths_per_cut
    counts["ls_truncated"] += bool(beam_pruned or expansion_cap or complete_cap)
    counts["ls_beam_pruned_cuts"] += beam_pruned
    counts["ls_expansion_capped_cuts"] += expansion_cap
    counts["ls_complete_capped_cuts"] += complete_cap
    counts["ls_no_feasible_reconnection_cuts"] += not complete
    return complete[:config.max_complete_paths_per_cut]


class ObservedPairs(DirectedPairs):
    def reconstruct(self, key, indexes, prefix, suffix, config, counts):
        key=(key, config)
        cached=self.beam_cache.get(key)
        if cached is None:
            work=Counter()
            paths=inspect_rebuild(indexes,prefix,suffix,self.codes,config,work)
            if self.cache_size:
                if len(self.beam_cache)>=self.cache_size:
                    del self.beam_cache[next(iter(self.beam_cache))]
                self.beam_cache[key]=(paths,work)
            counts["ls_actual_edge_expansions"]+=work["ls_edge_expansions"]
        else:
            paths,work=cached
            counts["ls_beam_cache_hits"]+=1
        counts.update(work)
        return paths


def shortcuts(tracker, task, candidate, limits, config, phase, mode, unified=False):
    """Legacy first-improvement order, with explicit V2 scoring/encoding."""
    graph=tracker.graphs[task]
    counts=Counter(ls_calls=1)
    started=perf_counter()
    result=candidate
    before=tracker.per_task[task]["ls_evaluations"]
    for _ in range(config.max_passes):
        baseline=result
        path=baseline.solution.path
        edges=[graph.by_id[e] for e in path]
        vertices=[graph.source]+[e.v for e in edges]
        prefix=np.cumsum([0.0]+[e.weight for e in edges])
        for i,u in enumerate(vertices[:-1]):
            for j in range(i+1,len(vertices)):
                if (mode=="parallel" and j!=i+1) or (mode=="shortcut" and j==i+1):
                    continue
                if result is not baseline or tracker.exhausted:
                    break
                view=graph.by_neighbor.get(u,{}).get(vertices[j])
                if view is None:
                    continue
                upper=prefix[j]-prefix[i]
                counts["ls_edge_expansions"] += len(view)
                failed=set()
                for edge in view.lighter_than(upper):
                    if edge.domain in failed:
                        continue
                    if tracker.exhausted:
                        break
                    ids=path[:i]+(edge.id,)+path[j:]
                    proposed=tracker.path(task,ids,"ls")
                    counts["local_search_checks"]+=1
                    counts["ls_complete_paths"]+=1
                    if not proposed.feasible:
                        failed.add(edge.domain)
                        counts["ls_du_rejects"]+=1
                    else:
                        counts["ls_feasible_neighbors"]+=1
                        if proposed.cost<baseline.solution.cost:
                            encoded=tracker.roundtrip(task,proposed,limits,"ls",
                                unified=baseline.chromosome if unified else None,phase=phase)
                            if encoded is not None:
                                result=encoded
                                break
            if result is not baseline or tracker.exhausted:
                break
        if result is baseline or tracker.exhausted:
            break
    finish_stats(tracker,task,phase,candidate,result,before,counts,started)
    return result


def finish_stats(tracker,task,phase,original,result,before,counts,started):
    counts["ls_no_improvement"] += result is original
    if result is not original:
        counts["ls_accepted"] += 1
        counts["ls_cost_reduction"] += original.solution.cost-result.solution.cost
    counts["ls_time_s"]=perf_counter()-started
    work=tracker.per_task[task]
    work[phase+"_ls_evaluations"]+=work["ls_evaluations"]-before
    work.update(counts)
    work.update({phase+"_"+k:v for k,v in counts.items()})


def replace_segment(tracker, task, candidate, limits, pairs, config, rng, phase, unified=False):
    """Search a bounded replacement inside the two retained path sections."""
    graph=tracker.graphs[task]
    counts=Counter(ls_calls=1)
    started=perf_counter()
    before=tracker.per_task[task]["ls_evaluations"]
    path=candidate.solution.path
    edges=[graph.by_id[e] for e in path]
    nodes=[graph.source]+[edge.v for edge in edges]
    cuts=[(i,j) for i in range(len(edges)) for j in range(i+1,len(nodes))]
    if len(cuts)>config.max_cut_pairs_per_candidate:
        counts["ls_cut_sampling_truncated"]+=1
        cuts=sorted(rng.sample(cuts,config.max_cut_pairs_per_candidate))
    if not hasattr(pairs,"lower_by_target"):
        pairs.lower_by_target={}
        pairs.vertex_costs=pair_costs(graph)
    best=candidate
    for i,j in cuts:
        if tracker.exhausted:
            break
        counts["ls_cut_attempts"]+=1
        counts["local_search_checks"]+=1
        target=nodes[j]
        if target not in pairs.lower_by_target:
            pairs.lower_by_target[target]=shortest(pairs.vertex_costs.T,target)
        active,closed=-1,0
        for e in edges[:i]:
            domain=pairs.codes[e.domain]
            if active>=0 and active!=domain:
                closed |= 1<<active
            active=domain
        visited=sum(1<<v for v in nodes[:i]+nodes[j+1:])
        result=decode_beam(graph,pairs.lower_by_target[target],
                           BeamConfig(config.beam_width,0,config.max_edge_expansions_per_cut,
                                      config.max_complete_paths_per_cut),
                           start=nodes[i],target=target,visited=visited,active=active,closed=closed)
        counts["ls_edge_expansions"]+=result.expansions
        counts["ls_ranking_edge_reads"]+=result.ranking_reads
        counts["ls_truncated"]+=result.cap_hit
        counts["ls_complete_paths"]+=len(result.paths)
        for subpath in result.paths:
            if tracker.exhausted:
                break
            proposed=tracker.path(task,path[:i]+subpath+path[j:],"ls")
            if not proposed.feasible:
                counts["ls_du_rejects"]+=1
                continue
            counts["ls_feasible_neighbors"]+=1
            if proposed.cost>=best.solution.cost:
                continue
            encoded=tracker.roundtrip(task,proposed,limits,"ls",
                                       unified=candidate.chromosome if unified else None,phase=phase)
            if encoded is not None:
                best=encoded
    finish_stats(tracker,task,phase,candidate,best,before,counts,started)
    return best


def improve(tracker, task, candidate, limits, pairs, config, rng, phase, mode, unified=False):
    if not candidate.solution.feasible:
        return candidate
    if mode=="2opt":
        return directed_2opt(tracker,task,candidate,limits,pairs,config,rng,phase,unified)
    if mode=="subpath":
        return replace_segment(tracker,task,candidate,limits,pairs,config,rng,phase,unified)
    if mode in ("legacy","parallel","shortcut"):
        return shortcuts(tracker,task,candidate,limits,config,phase,mode,unified)
    raise ValueError(f"Unknown local search {mode}")
