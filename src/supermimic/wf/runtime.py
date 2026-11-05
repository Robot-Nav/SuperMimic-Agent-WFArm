"""Optional pure-FK kernel compilation; no model/optimizer/data mutation."""
import time

import torch


def compile_fk(task,state,batch_sizes):
    original=task.fk;started=time.monotonic()
    sizes=sorted(set(int(value) for value in batch_sizes))
    device=state.device
    devices=([device.index if device.index is not None else torch.cuda.current_device()]
        if device.type=='cuda' else [])
    try:
        compiled=torch.compile(original,fullgraph=True,dynamic=True,mode='default')
        # Compilation/warm-up must not consume the resumed policy's noise RNG.
        with torch.random.fork_rng(devices=devices):
            for size in sizes:
                sample=state[:1].detach().expand(size,-1).clone()
                with torch.no_grad():compiled(sample)
                with torch.enable_grad():
                    compiled(sample)  # Initial imagined state has no gradients.
                    leaf=sample.requires_grad_(True)
                    bodies=compiled(leaf)
                    loss=sum(value.square().mean() for value in bodies.values())
                    gradient=torch.autograd.grad(loss,leaf)[0]
                    if not torch.isfinite(gradient).all():raise FloatingPointError('compiled FK warm-up has a nonfinite state gradient')
        if device.type=='cuda':torch.cuda.synchronize(device)
        task.fk=compiled
        return dict(stage='compiled',wall_s=time.monotonic()-started,batch_sizes=sizes,
            device=str(device),rng_preserved=True,task_or_parameters_changed=False)
    except Exception as error:
        task.fk=original
        return dict(stage='eager_fallback',wall_s=time.monotonic()-started,error=repr(error),
            rng_preserved=True,task_or_parameters_changed=False)


def compile_clearance(task,state,batch_sizes):
    """Warm the unchanged surface-distance computation, preserving all RNG."""
    original=task.surface_clearance;started=time.monotonic()
    if original is None:return dict(stage='not_applicable',rng_preserved=True)
    sizes=sorted(set(int(value) for value in batch_sizes));device=state.device
    devices=([device.index if device.index is not None else torch.cuda.current_device()]
        if device.type=='cuda' else [])
    try:
        with torch.random.fork_rng(devices=devices):
            compiled=torch.compile(original,fullgraph=True,mode='default')
            for size in sizes:
                sample=state[:1].detach().expand(size,-1).clone()
                with torch.no_grad():compiled(task.fk(sample))
                with torch.enable_grad():
                    compiled(task.fk(sample))
                    leaf=sample.requires_grad_(True)
                    result=compiled(task.fk(leaf))
                    gradient=torch.autograd.grad(sum(v.square().mean() for v in result.values()),leaf)[0]
                    if not torch.isfinite(gradient).all():raise FloatingPointError('compiled clearance has a nonfinite gradient')
        if device.type=='cuda':torch.cuda.synchronize(device)
        task.surface_clearance=compiled
        return dict(stage='compiled',wall_s=time.monotonic()-started,batch_sizes=sizes,
            device=str(device),rng_preserved=True,task_or_parameters_changed=False)
    except Exception as error:
        task.surface_clearance=original
        return dict(stage='eager_fallback',wall_s=time.monotonic()-started,error=repr(error),
            rng_preserved=True,task_or_parameters_changed=False)
