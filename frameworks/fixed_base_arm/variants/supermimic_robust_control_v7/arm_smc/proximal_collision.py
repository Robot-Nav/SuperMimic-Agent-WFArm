"""Differentiable separating-face margin for the observed arm-base collision.

Uses convex hull vertices from the compiled MuJoCo meshes. A subset of face
axes gives a conservative separation margin: a real intersection is never
declared separated, but skew disjoint bodies can get a conservative penalty.
This learning aid does not change physical contacts or replace collision tests.
"""
import numpy as np
import torch
from torch import nn
from scipy.spatial import ConvexHull

from .rotations import quat_apply,quat_mul


def select_normals(normals,maximum=64):
    normals=np.asarray(normals).copy()
    index=np.abs(normals).argmax(-1)
    normals*=np.where(normals[np.arange(len(normals)),index]<0,-1,1)[:,None]
    normals=np.unique(np.round(normals,6),axis=0)
    normals/=np.linalg.norm(normals,axis=-1,keepdims=True)
    selected=[int(np.argmax(normals[:,0]))]
    nearest=np.ones(len(normals))
    for _ in range(min(maximum,len(normals))-1):
        nearest=np.minimum(nearest,1-np.abs(normals@normals[selected[-1]]))
        selected.append(int(np.argmax(nearest)))
    return normals[selected]


class ProximalClearance(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.geoms=[model.geom(name).id for name in
                    ("piper_base_link_collision_0","piper_link2_collision_0")]
        self.bodies=[int(model.geom_bodyid[g])-1 for g in self.geoms]
        self.register_buffer("world_axes",torch.eye(3))
        def buf(name,value,dtype=torch.float32):
            self.register_buffer(name,torch.as_tensor(np.asarray(value).copy(),dtype=dtype))
        for k,g in enumerate(self.geoms):
            mesh=int(model.geom_dataid[g]);start=int(model.mesh_vertadr[mesh])
            vertices=np.asarray(model.mesh_vert[start:start+int(model.mesh_vertnum[mesh])],dtype=float)
            hull=ConvexHull(vertices)
            buf(f"vertices{k}",vertices[hull.vertices]);buf(f"normals{k}",select_normals(hull.equations[:,:3]))
            buf(f"pos{k}",model.geom_pos[g]);buf(f"quat{k}",model.geom_quat[g])

    def forward(self,bodies):
        shape=bodies["pos"].shape[:-2]
        p=bodies["pos"].reshape(-1,bodies["pos"].shape[-2],3)
        q=bodies["quat"].reshape(-1,bodies["quat"].shape[-2],4)
        vertices=[];normals=[]
        for k,body in enumerate(self.bodies):
            rotation=quat_mul(q[:,body],getattr(self,f"quat{k}")[None].expand(len(p),-1))
            position=p[:,body]+quat_apply(q[:,body],getattr(self,f"pos{k}")[None].expand(len(p),-1))
            v=getattr(self,f"vertices{k}");n=getattr(self,f"normals{k}")
            vertices.append(position[:,None]+quat_apply(rotation[:,None].expand(-1,len(v),-1),v[None].expand(len(p),-1,-1)))
            normals.append(quat_apply(rotation[:,None].expand(-1,len(n),-1),n[None].expand(len(p),-1,-1)))
        axes=torch.cat((self.world_axes[None].expand(len(p),-1,-1),*normals),1)
        with torch.no_grad():
            a=axes@vertices[0].transpose(-1,-2);b=axes@vertices[1].transpose(-1,-2)
            forward=b.amin(-1)-a.amax(-1);reverse=a.amin(-1)-b.amax(-1)
            which=torch.maximum(forward,reverse).argmax(-1)
            rows=torch.arange(len(p),device=p.device)
            sign=torch.where(reverse[rows,which]>forward[rows,which],-1.,1.)
            axis=axes[rows,which]*sign[:,None]
            ia=(vertices[0]*axis[:,None]).sum(-1).argmax(-1)
            ib=(vertices[1]*axis[:,None]).sum(-1).argmin(-1)
        axis=axes[rows,which]*sign[:,None]
        gap=((vertices[1][rows,ib]-vertices[0][rows,ia])*axis).sum(-1)
        return gap.reshape(shape)
