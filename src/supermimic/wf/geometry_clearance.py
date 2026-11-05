"""Differentiable sampled arm surface clearance to actual robot primitives.

Includes fixed fingers: excluding their actions does not remove their geometry.
MuJoCo collision meshes are convex; deterministic samples come from their hull
in compiled geom coordinates. The primitive SDF is analytic; surface sampling
is approximate and does not replace physical collision checks.

https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .math import quat_apply,quat_apply_inverse,quat_mul


def primitive_sdf(point,kind,size):
    """Local point SDF for MuJoCo sphere=2, capsule=3, cylinder=5, box=6."""
    positive=lambda x:torch.sqrt(x.square().sum(-1)+1e-16)
    box=point.abs()-size
    box_distance=positive(box.clamp_min(0))+box.amax(-1).clamp_max(0)
    radial=torch.sqrt(point[...,:2].square().sum(-1)+1e-16)
    cylinder=torch.stack((radial-size[...,0],point[...,2].abs()-size[...,1]),-1)
    cylinder_distance=positive(cylinder.clamp_min(0))+cylinder.amax(-1).clamp_max(0)
    capsule=torch.cat((point[...,:2],(point[...,2]-point[...,2].clamp(-size[...,1],size[...,1]))[...,None]),-1)
    capsule_distance=positive(capsule)-size[...,0]
    sphere_distance=positive(point)-size[...,0]
    return torch.where(kind==6,box_distance,torch.where(kind==5,cylinder_distance,
        torch.where(kind==3,capsule_distance,sphere_distance)))


def _samples(model,geom,spacing=.006,maximum=128):
    """Deterministic hull samples; report finite-pool covering error explicitly."""
    from scipy.spatial import ConvexHull
    kind=int(model.geom_type[geom])
    if kind==7:
        mesh=int(model.geom_dataid[geom]);start=int(model.mesh_vertadr[mesh]);count=int(model.mesh_vertnum[mesh])
        vertices=np.asarray(model.mesh_vert[start:start+count],dtype=np.float64)
    elif kind==6:
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        vertices=signs*model.geom_size[geom]
    else:raise ValueError('arm surface sampler requires mesh or box geometry')
    hull=ConvexHull(vertices);pool=[]
    for triangle in hull.simplices:
        a,b,c=vertices[triangle]
        intervals=max(1,int(np.ceil(max(np.linalg.norm(b-a),np.linalg.norm(c-a),np.linalg.norm(c-b))/spacing)))
        weights=np.array([(i/intervals,j/intervals) for i in range(intervals+1) for j in range(intervals+1-i)])
        pool.append(a+weights[:,:1]*(b-a)+weights[:,1:]*(c-a))
    pool=np.unique(np.round(np.concatenate(pool),8),axis=0)
    selected=[int(np.argmax(np.linalg.norm(pool-pool.mean(0),axis=-1)))]
    nearest=np.full(len(pool),np.inf)
    for _ in range(min(maximum,len(pool))-1):
        nearest=np.minimum(nearest,((pool-pool[selected[-1]])**2).sum(-1))
        selected.append(int(nearest.argmax()))
    nearest=np.minimum(nearest,((pool-pool[selected[-1]])**2).sum(-1))
    # The extra spacing bounds a triangle cell's distance to its sampled grid.
    cover=float(np.sqrt(nearest.max())+spacing)
    return pool[selected],cover


class SurfaceClearance(nn.Module):
    """Robot-specific learning aid, not an exact mesh-mesh collision solver.

    Closest primitive selection is detached, then its analytic SDF is evaluated
    with gradients. This retains the minimum's local gradient without retaining
    an autograd graph for every point/obstacle pair in every imagined step.
    """
    def __init__(self,model,*,wheel_geom_ids,spacing=.006,maximum=128,obstacle_chunk=6):
        super().__init__();self.obstacle_chunk=obstacle_chunk
        names=[model.body(i).name for i in range(model.nbody)]
        arm_names={'piper_link3','piper_link4','piper_link5','piper_link6',
            'piper_gripper_base','piper_gripper_link1','piper_gripper_link2'}
        self.arm_geoms=[i for i in range(model.ngeom) if names[int(model.geom_bodyid[i])] in arm_names
            and int(model.geom_contype[i])!=0]
        obstacles=[i for i in range(model.ngeom) if int(model.geom_bodyid[i])>0
            and not names[int(model.geom_bodyid[i])].startswith('piper_')
            and int(model.geom_type[i]) in (2,3,5,6) and int(model.geom_contype[i])!=0]
        if not self.arm_geoms or not obstacles:raise ValueError('robot has no supported clearance pairs')
        points=[];point_bodies=[];point_geom=[];covers=[];self.geometry_names=[];self.sample_slices=[]
        for index,geom in enumerate(self.arm_geoms):
            count=maximum*2 if float(np.max(model.geom_size[geom]))>.15 else maximum
            local,cover=_samples(model,geom,spacing,count)
            q=torch.tensor(model.geom_quat[geom],dtype=torch.float64).expand(len(local),-1)
            body_local=quat_apply(q,torch.tensor(local))+torch.tensor(model.geom_pos[geom])
            self.sample_slices.append((len(point_bodies),len(point_bodies)+len(local)))
            points.append(body_local);point_bodies.extend([int(model.geom_bodyid[geom])-1]*len(local))
            point_geom.extend([index]*len(local));covers.append(cover);self.geometry_names.append(model.geom(geom).name)
        self.register_buffer('points',torch.cat(points).float())
        self.register_buffer('point_bodies',torch.tensor(point_bodies,dtype=torch.long))
        self.register_buffer('point_geom',torch.tensor(point_geom,dtype=torch.long))
        self.register_buffer('cover',torch.tensor(covers,dtype=torch.float32))
        for name,values,dtype in [('obstacle_bodies',model.geom_bodyid[obstacles]-1,torch.long),
                ('obstacle_position',model.geom_pos[obstacles],torch.float32),
                ('obstacle_quaternion',model.geom_quat[obstacles],torch.float32),
                ('obstacle_kind',model.geom_type[obstacles],torch.long),
                ('obstacle_size',model.geom_size[obstacles],torch.float32)]:
            self.register_buffer(name,torch.as_tensor(values.copy(),dtype=dtype))
        supported={int(i) for i in wheel_geom_ids}
        floor=[i for i,geom in enumerate(obstacles) if geom not in supported]
        self.register_buffer('floor_primitives',torch.tensor(floor,dtype=torch.long))

    def forward(self,bodies):
        shape=bodies['pos'].shape[:-2]
        positions=bodies['pos'].reshape(-1,bodies['pos'].shape[-2],3)
        quaternions=bodies['quat'].reshape(-1,bodies['quat'].shape[-2],4)
        with torch.no_grad():
            points=positions[:,self.point_bodies]+quat_apply(quaternions[:,self.point_bodies],self.points)
            obs_quat=quat_mul(quaternions[:,self.obstacle_bodies],
                self.obstacle_quaternion[None].expand_as(quaternions[:,self.obstacle_bodies]))
            obs_pos=positions[:,self.obstacle_bodies]+quat_apply(quaternions[:,self.obstacle_bodies],self.obstacle_position)
            nearest=torch.full(points.shape[:2],float('inf'),device=points.device,dtype=points.dtype)
            indices=torch.zeros(points.shape[:2],device=points.device,dtype=torch.long)
            for begin in range(0,len(self.obstacle_kind),self.obstacle_chunk):
                end=min(begin+self.obstacle_chunk,len(self.obstacle_kind))
                delta=points[:,:,None]-obs_pos[:,None,begin:end]
                local=quat_apply_inverse(obs_quat[:,None,begin:end].expand(*delta.shape[:-1],4),delta)
                distance=primitive_sdf(local,self.obstacle_kind[None,None,begin:end],self.obstacle_size[None,None,begin:end])
                value,index=distance.min(-1);better=value<nearest
                indices=torch.where(better,index+begin,indices);nearest=torch.minimum(nearest,value)
            selected=[];ground_selected=[]
            for begin,end in self.sample_slices:
                selected.append(nearest[:,begin:end].argmin(-1)+begin)
                ground_selected.append(points[:,begin:end,2].argmin(-1)+begin)
            selected=torch.stack(selected,-1);ground_selected=torch.stack(ground_selected,-1)
            obstacle=torch.gather(indices,1,selected)
        # Rebuild only one closest surface point per geom with gradients. The
        # remaining sampled points affect selection, not the local minimum VJP.
        rows=torch.arange(positions.shape[0],device=positions.device)[:,None]
        def transform(sample):
            body=self.point_bodies[sample]
            return positions[rows,body]+quat_apply(quaternions[rows,body],self.points[sample])
        point=transform(selected)
        body=self.obstacle_bodies[obstacle]
        selected_pos=positions[rows,body]+quat_apply(quaternions[rows,body],self.obstacle_position[obstacle])
        selected_q=quat_mul(quaternions[rows,body],self.obstacle_quaternion[obstacle])
        local=quat_apply_inverse(selected_q,point-selected_pos)
        per_geom=primitive_sdf(local,self.obstacle_kind[obstacle],self.obstacle_size[obstacle])
        ground=transform(ground_selected)[...,2]
        # Exact lowest height of the base/leg primitives over the flat floor.
        # Wheel-floor contacts are supported and deliberately excluded.
        floor=self.floor_primitives;body=self.obstacle_bodies[floor]
        floor_pos=positions[:,body]+quat_apply(quaternions[:,body],self.obstacle_position[floor])
        floor_q=quat_mul(quaternions[:,body],self.obstacle_quaternion[floor][None].expand_as(quaternions[:,body]))
        up=torch.zeros_like(floor_pos);up[...,2]=1.
        axis=quat_apply_inverse(floor_q,up);size=self.obstacle_size[floor];kind=self.obstacle_kind[floor]
        box_support=(axis.abs()*size).sum(-1)
        cylinder_support=size[...,0]*torch.sqrt(axis[...,:2].square().sum(-1)+1e-16)+size[...,1]*axis[...,2].abs()
        capsule_support=size[...,0]+size[...,1]*axis[...,2].abs()
        support=torch.where(kind==6,box_support,torch.where(kind==5,cylinder_support,
            torch.where(kind==3,capsule_support,size[...,0])))
        ground=torch.cat((ground,floor_pos[...,2]-support),-1)
        return {'robot':per_geom.reshape(*shape,len(self.arm_geoms)),
            'ground':ground.reshape(*shape,len(self.arm_geoms)+len(self.floor_primitives))}
