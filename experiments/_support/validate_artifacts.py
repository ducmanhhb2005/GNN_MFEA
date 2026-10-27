"""Independently verify saved paths, traces, statistics, source and data hashes."""
import argparse,csv,gc,hashlib,json,math
from datetime import date
from collections import defaultdict
from pathlib import Path
from idpc_du import Graph,validate_path
ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def audit_outputs(directories):
    by_file=defaultdict(list)
    dataset_hashes={}
    reports={}
    for directory in directories:
        output=Path(directory).resolve()
        manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
        versions={}
        for relative,expected in manifest['code_sha256'].items():
            current=ROOT/relative
            if not current.is_file() or digest(current)!=expected:
                raise AssertionError(f'Source changed since run: {output}: {relative}')
            versions[relative]='current'
        for filename,expected in manifest['dataset_sha256'].items():
            if filename in dataset_hashes:assert dataset_hashes[filename]==expected
            dataset_hashes[filename]=expected
        files={Path(f).name:Path(f) for f in manifest['config']['files']}
        assert len(files)==len(manifest['config']['files']), 'Ambiguous duplicate input basenames'
        records=json.loads((output/'paths.json').read_text(encoding='utf-8'))
        expected_runs=manifest['config']['runs']
        assert len(records)==len(files)*expected_runs*len(manifest['config']['methods'])
        buckets=defaultdict(list)
        seen=set()
        for record in records:
            key=(record['algorithm'],record['instance'],record['seed'])
            assert key not in seen;seen.add(key)
            buckets[(record['algorithm'],record['instance'])].append(record)
            by_file[files[record['instance']]].append(record)
            assert record['evaluations'] >= manifest['config']['population']
            assert 0 <= record['infeasible_evaluations'] <= record['evaluations']
            if record['cost'] is not None:
                assert record['objective_min'] == record['cost']
                assert record['objective_max'] >= record['cost']
        expected_seeds=set(range(manifest['config']['seed'],manifest['config']['seed']+expected_runs))
        for bucket in buckets.values():assert {r['seed'] for r in bucket}==expected_seeds
        trace_count=0
        for checkpoint in (output/'checkpoints').glob('*.json'):
            content=json.loads(checkpoint.read_text(encoding='utf-8'))
            assert content['fingerprint']==manifest['fingerprint']
            assert sum(r['evaluations'] for r in content['rows']) == manifest['config']['population'] * (len(content['rows']) + manifest['config']['generations'])
            for row in content['rows']:
                values=[math.inf if x is None else x for x in row['convergence']]
                assert len(values)==manifest['config']['generations']+1
                assert all(a>=b for a,b in zip(values,values[1:])), 'Non-monotone best-so-far trace'
                assert values[-1]==(math.inf if row['cost'] is None else row['cost'])
                assert (row['algorithm'],row['instance'],row['seed']) in seen
                trace_count+=1
        assert trace_count==len(records)
        with (output/'summary.csv').open(encoding='utf-8') as stream:
            summaries=list(csv.DictReader(stream))
        assert len(summaries)==len(buckets)
        for summary in summaries:
            bucket=buckets[(summary['algorithm'],summary['instance'])]
            costs=[math.inf if r['cost'] is None else r['cost'] for r in bucket]
            assert len(costs)==int(summary['runs'])
            assert min(costs)==float(summary['BF'])
            avg=math.fsum(costs)/len(costs) if all(math.isfinite(c) for c in costs) else math.inf
            assert math.isclose(avg,float(summary['AVG']),rel_tol=1e-10,abs_tol=1e-10)
            sd=(math.sqrt(math.fsum((v-avg)**2 for v in costs)/(len(costs)-1)) if len(costs)>1 else 0.) if math.isfinite(avg) else math.inf
            assert math.isclose(sd,float(summary['STD_sample']),rel_tol=1e-9,abs_tol=1e-9)
            assert sum(bool(r['valid']) for r in bucket)==int(summary['feasible'])
        reports[output.name]=dict(path_records=len(records),convergence_traces=trace_count,
            instances=len(files),runs_per_instance=expected_runs,
            source_versions=versions,
            dataset_hashes_verified=True,paths_verified=True,statistics_verified=True)
    for file,records in sorted(by_file.items()):
        assert digest(file)==dataset_hashes[str(file)]
        graph=Graph.from_file(str(file))
        for record in records:
            value=validate_path(graph,record['edge_ids'])
            expected=math.inf if record['cost'] is None else record['cost']
            assert math.isclose(value,expected), (file,record['algorithm'],record['seed'])
            assert math.isfinite(value)==record['valid']
            assert record['nodes']==[graph.source]+[graph.by_id[i].v for i in record['edge_ids']]
            assert record['domains']==[graph.by_id[i].domain for i in record['edge_ids']]
        del graph
        gc.collect()
    return dict(date=date.today().isoformat(),unique_original_instances=len(by_file),
        path_records_checked=sum(r['path_records'] for r in reports.values()),
        convergence_traces_checked=sum(r['convergence_traces'] for r in reports.values()),experiments=reports)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directories',nargs='+',type=Path)
    p.add_argument('--out',type=Path,default=ROOT/'experiments/paper_run/validation.json')
    a=p.parse_args()
    result=audit_outputs(a.directories)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Verified',result['path_records_checked'],'paths and',result['convergence_traces_checked'],'traces on',result['unique_original_instances'],'instances.',flush=True)
    print('Audit:',a.out,flush=True)
if __name__=='__main__':main()
