"""Controlled feature, loss and knowledge-fusion variants of the V2 learner."""

from collections import Counter, defaultdict
from dataclasses import fields
import math
import random
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ekt_gl.idpc.v2.core import seed_for
from ekt_gl.idpc.v2.learning import EdgeNetwork, Features, OnlineLearning
from .beam import pair_costs, shortest


def feature_set(graph, device, policy):
    if policy == "v2":
        return Features.build(graph, device)
    f=Features.build(graph,"cpu")
    raw=graph.edges.table.rows
    if policy=="legacy":
        count=max(len(raw),1)
        scale=math.log1p(float(count))
        weight=np.log1p(raw["weight"])
        weight/=max(float(weight.max(initial=0.0)),1.0 if not len(weight) else 1e-12)
        inverse=f.edge_pair.numpy()
        pair_count=np.bincount(inverse,minlength=len(f.pair_u))
        pair_min=np.full(len(pair_count),np.inf)
        np.minimum.at(pair_min,inverse,weight)
        domains=np.bincount(raw["domain"],minlength=len(graph.domains))
        dom_weight=np.bincount(raw["domain"],weights=weight,minlength=len(graph.domains))/np.maximum(domains,1)
        f.nodes[:,2]=torch.tensor(np.log1p(np.bincount(raw["v"],minlength=graph.n))/scale)
        f.nodes[:,3]=torch.tensor(np.log1p(np.bincount(raw["u"],minlength=graph.n))/scale)
        f.nodes[:,4:]=0
        f.pair_features=torch.tensor(np.column_stack((np.log1p(pair_count)/scale,pair_min,
                                                      np.zeros(len(pair_count)))),dtype=torch.float32)
        f.multiplicity=torch.ones_like(f.multiplicity)
        f.edge_features=torch.tensor(np.column_stack((weight,domains[raw["domain"]]/count,
                                                       dom_weight[raw["domain"]])),dtype=torch.float32)
    elif policy=="distances":
        matrix=pair_costs(graph)
        forward=shortest(matrix,graph.source)
        reverse=shortest(matrix.T,graph.target)
        # Log distances and a reachability flag avoid sharing arbitrary domain IDs.
        def normalize(values):
            valid=np.isfinite(values)
            logged=np.zeros_like(values)
            logged[valid]=np.log1p(values[valid])
            logged/=max(1.0,float(logged.max(initial=0)))
            return logged,valid
        fwd,fwd_ok=normalize(forward)
        rev,rev_ok=normalize(reverse)
        f.nodes=torch.cat((f.nodes,torch.tensor(np.column_stack((fwd,rev,fwd_ok,rev_ok)),
                                              dtype=torch.float32)),dim=1)
        endpoints=np.column_stack((fwd[raw["u"]],rev[raw["v"]]))
        f.edge_features=torch.cat((f.edge_features,torch.tensor(endpoints,dtype=torch.float32)),dim=1)
        f.pair_features=torch.cat((f.pair_features,torch.tensor(
            np.column_stack((fwd[f.pair_u.numpy()],rev[f.pair_v.numpy()])),dtype=torch.float32)),dim=1)
    elif policy!="zero":
        raise ValueError(f"Unknown feature intervention {policy}")
    return Features(*(getattr(f,field.name).to(device) for field in fields(f)))


class NoCrossTaskAttention(nn.Module):
    def __init__(self,value):
        super().__init__()
        self.value=value

    def forward(self,pool):
        return self.value(pool),torch.eye(len(pool),device=pool.device,dtype=pool.dtype)


class ResearchLearning(OnlineLearning):
    def __init__(self,graphs,features,config,seed,variant,diagnostics=False):
        super().__init__(graphs,features,config)
        self.seed,self.variant=seed,variant
        self.event=0
        self.gradients={}
        self.examples=(0,0)
        self.soft=[]
        if variant.features=="distances":
            self.model.nodes=nn.Linear(12,config.hidden_dim).to(config.device)
            self.model.pairs=nn.Linear(5,config.hidden_dim).to(config.device)
            self.model.exact_edge=nn.Linear(5,config.hidden_dim,bias=False).to(config.device)
        if variant.fusion=="identity":
            self.model.fusion=NoCrossTaskAttention(self.model.fusion.value)
        if variant.features=="distances" or variant.fusion=="identity":
            self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config.learning_rate)
        if diagnostics:
            self.optimizer.register_step_pre_hook(self._record_gradients)

    def _record_gradients(self,optimizer,args,kwargs):
        groups=defaultdict(list)
        for name,param in self.model.named_parameters():
            if param.grad is not None:
                group="attention" if name.startswith("fusion") else ("decoder" if name.startswith(
                    ("first","last","exact_edge")) else "encoder")
                groups[group].append(param.grad.detach().square().sum())
        self.gradients={name:float(torch.stack(v).sum().sqrt().cpu()) for name,v in groups.items()}

    def labels(self,populations):
        labels=super().labels(populations)
        self.soft=[]
        if self.variant.loss in ("frequency","elite_cost"):
            for pop in populations:
                elites=[m.solution for m in pop if m.solution.feasible][:self.config.elite_count]
                mass=defaultdict(float)
                if elites:
                    weights=([1.0]*len(elites) if self.variant.loss=="frequency" else
                             [1.0/max(1.0,e.cost) for e in elites])
                    scale=sum(weights)
                    for elite,w in zip(elites,weights):
                        for edge in elite.path:
                            mass[edge]+=w/scale
                self.soft.append(mass)
        return labels

    def fit(self,labels):
        if self.variant.loss=="full":
            self.examples=(sum(len(p) for p,count,_ in labels if count),
                           sum(len(g.edges)-len(p) for g,(p,count,_) in zip(self.graphs,labels) if count))
            return super().fit(labels)
        graph_batches=[]
        for i,(g,f,(p,count,_)) in enumerate(zip(self.graphs,self.features,labels)):
            if not count:
                continue
            n=len(g.edges)
            mapping=g.by_id.positions
            positive=sorted(p if mapping is None else (mapping[x] for x in p))
            soft=self.soft[i] if self.soft else {x:1.0 for x in p}
            lookup=sorted((x if mapping is None else mapping[x],v) for x,v in soft.items())
            if self.variant.loss=="sampled":
                rng=random.Random(seed_for(self.seed,"negative",self.event,i))
                k=min(512,n-len(p))
                blocked=set(positive)
                neg=[x for x in rng.sample(range(n),k+len(p)) if x not in blocked][:k]
                positions=torch.tensor(positive+neg,device=self.config.device,dtype=torch.long)
                weights=torch.cat((torch.ones(len(positive),device=self.config.device),
                                   torch.zeros(k,device=self.config.device)))
            else:
                positions=None
                weights=torch.zeros(n,device=self.config.device)
                ids,values=zip(*lookup)
                weights[list(ids)]=torch.tensor(values,device=weights.device).clamp(0,1)
            graph_batches.append((i,f,positions,weights,len(positive)))
        if not graph_batches:
            self.examples=(0,0)
            return None,0.0
        self.examples=(sum(n for *_,n in graph_batches),
                       sum(len(w)-n for _,_,_,w,n in graph_batches))
        self._sync()
        start=perf_counter()
        self.model.train()
        for _ in range(self.config.train_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            bases,weight,_=self.model.contexts(self.features)
            original=[*bases,weight,self.model.last.weight,self.model.last.bias]
            leaves=[x.detach().requires_grad_() for x in original]
            fw,lw,lb=leaves[-3:]
            loss_value=torch.zeros((),device=fw.device)
            for i,f,positions,targets,n in graph_batches:
                size=len(targets)
                balanced=self.variant.loss in ("sampled","balanced")
                positive_factor=(0.5/n if size>n else 1/n) if balanced else 1/size
                negative_factor=0.5/(size-n) if balanced and size>n else 1/size
                for offset in range(0,size,self.config.edge_batch_size):
                    batch=slice(offset,offset+self.config.edge_batch_size)
                    selected=batch if positions is None else positions[batch]
                    logits=EdgeNetwork.logits(leaves[i],fw,f.edge_pair[selected],
                                              f.edge_features[selected],lw,lb)
                    logit=logits[:,1]-logits[:,0]
                    target=targets[batch]
                    loss=(positive_factor*target*F.softplus(-logit) +
                          negative_factor*(1-target)*F.softplus(logit)).sum()/len(graph_batches)
                    loss.backward()
                    loss_value+=loss.detach()
            pair=[(value,leaf.grad) for value,leaf in zip(original,leaves) if leaf.grad is not None]
            torch.autograd.backward(*(list(v) for v in zip(*pair)))
            self.optimizer.step()
            self.steps+=1
        self._sync()
        return float(loss_value),perf_counter()-start


class IndependentLearning(nn.Module):
    """A8 independently fitted per-task parameters and optimizer states."""
    def __init__(self,graphs,features,config,seed,variant):
        super().__init__()
        self.config=config
        self.learners=[ResearchLearning([g],[f],config,seed_for(seed,i),variant)
                       for i,(g,f) in enumerate(zip(graphs,features))]
        self.model=nn.ModuleList(learner.model for learner in self.learners)
        self.optimizer=OptimizerCollection([learner.optimizer for learner in self.learners])
        self.steps=0
        self.event=0
        self.examples=(0,0)
        self.gradients={}

    def labels(self,populations):
        return [learner.labels([population])[0] for learner,population in zip(self.learners,populations)]

    def fit(self,labels):
        results=[]
        sizes=Counter()
        for learner,label in zip(self.learners,labels):
            learner.event=self.event
            results.append(learner.fit([label]))
            sizes.update({i:n for i,n in enumerate(learner.examples)})
        self.examples=(sizes[0],sizes[1])
        losses=[loss for loss,_ in results if loss is not None]
        self.steps+=bool(losses)
        return (sum(losses)/len(losses) if losses else None),sum(t for _,t in results)

    def scores(self):
        result=[learner.scores() for learner in self.learners]
        return [scores[0] for scores,_,_ in result],np.eye(len(result)).tolist(),sum(t for _,_,t in result)


class OptimizerCollection:
    def __init__(self,optimizers):
        self.optimizers=optimizers

    def state_dict(self):
        return [x.state_dict() for x in self.optimizers]

    def load_state_dict(self,state):
        for optimizer,entry in zip(self.optimizers,state,strict=True):
            optimizer.load_state_dict(entry)
