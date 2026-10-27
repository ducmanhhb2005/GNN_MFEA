"""Data model and official .idpc / JSON graph parsers."""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class Edge:
    id: int
    u: int
    v: int
    weight: float
    domain: str


@dataclass
class Graph:
    n: int
    source: int
    target: int
    edges: list[Edge]

    def __post_init__(self):
        assert self.n >= 2 and 0 <= self.source < self.n and 0 <= self.target < self.n
        assert self.source != self.target
        assert len({e.id for e in self.edges}) == len(self.edges), "Edge ids must be unique"
        for e in self.edges:
            assert 0 <= e.u < self.n and 0 <= e.v < self.n and e.weight >= 0
        self.adj: dict[int,list[Edge]] = defaultdict(list)
        self.by_id = {e.id: e for e in self.edges}
        for e in self.edges:
            self.adj[e.u].append(e)
        # Preserve deterministic parallel-edge ordering, but index by (source, destination).
        # Dense original instances can contain millions of parallel edges. GPA only
        # needs to visit neighboring VERTICES and the parallel edges for one vertex.
        self.by_neighbor: dict[int, dict[int, list[Edge]]] = defaultdict(lambda: defaultdict(list))
        for e in self.edges:
            self.by_neighbor[e.u][e.v].append(e)
        # Edge-index genes refer to parallel edges in original file order.
        # The article does not specify their order; this is a fixed assumption.
        self.domains = sorted({e.domain for e in self.edges})

    @classmethod
    def from_json(cls, filename: str) -> "Graph":
        with open(filename, encoding="utf-8") as f:
            x = json.load(f)
        edges = [Edge(i, int(e[0]), int(e[1]), float(e[2]), str(e[3])) for i,e in enumerate(x["edges"])]
        return cls(n=int(x["n"]), source=int(x["source"]), target=int(x["target"]), edges=edges)


    @classmethod
    def from_idpc(cls, filename: str) -> "Graph":
        """Read official Mendeley IDPC-DU .idpc format.

        N D
        s t
        u v w d   (one edge per line, 1-based labels in source file)

        Edges keep their input ordering and unique IDs. Node/domain labels are
        normalized to zero-based node indices and string domain labels.
        """
        from pathlib import Path
        import re
        with open(filename, encoding="utf-8-sig") as f:
            try:
                header=f.readline().split()
                n,d=map(int,header)
                s,t=map(int,f.readline().split())
            except (ValueError,TypeError) as e:
                raise ValueError(f"Invalid IDPC header: {filename}") from e
            if n<2 or d<1 or not 1<=s<=n or not 1<=t<=n or s==t:
                raise ValueError(f"Invalid node/domain count or source-target in {filename}")
            edges=[]
            for line_num,line in enumerate(f,start=3):
                if not line.strip():
                    continue
                fields=line.split()
                if len(fields)!=4:
                    raise ValueError(f"{filename}:{line_num}: expected u v w d")
                try:
                    u,v,w,domain=map(int,fields)
                except ValueError as exc:
                    raise ValueError(f"{filename}:{line_num}: noninteger value") from exc
                if not 1<=u<=n or not 1<=v<=n or not 1<=domain<=d or w<0:
                    raise ValueError(f"{filename}:{line_num}: out-of-range edge/domain")
                edges.append(Edge(len(edges),u-1,v-1,float(w),str(domain)))
        match=re.search(r"idpc_(\d+)x(\d+)x(\d+)",Path(filename).stem,re.I)
        if match and (int(match.group(1))!=n or int(match.group(2))!=d or int(match.group(3))!=len(edges)):
            raise ValueError(f"{filename}: filename counts do not match file content")
        return cls(n=n,source=s-1,target=t-1,edges=edges)

    @classmethod
    def from_file(cls, filename: str) -> "Graph":
        """Auto-detect JSON demos and official .idpc data."""
        return (cls.from_idpc(filename)
                if str(filename).lower().endswith('.idpc')
                else cls.from_json(filename))
