"""Offline rolling-contact whole-body velocity QP teacher.

One optimization includes passive floating-base velocity and every actuated
leg/wheel/arm velocity. Only22 motor targets are executed; base state is never
assigned. This is kinematic WBC with real PD execution/screening, not a force-
feasible inverse-dynamics WBC or an MPC implementation. Never used by Actors.

APIs: https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html
https://osqp.org/docs/interfaces/python.html
"""
import copy

import mujoco
import numpy as np
import osqp
from scipy import sparse

from .gamepad import quat_mul


def heading_rate_row(rotation):
    """Map free-joint body angular velocity to world heading rate."""
    x,y=rotation[0,0],rotation[1,0]
    denominator=max(x*x+y*y,1e-8)
    return np.array([0.,(-x*rotation[1,2]+y*rotation[0,2])/denominator,
                     (x*rotation[1,1]-y*rotation[0,1])/denominator])


def capped(vector,maximum):
    return vector*min(1.,maximum/max(np.linalg.norm(vector),1e-12))


class KinematicWholeBodyTeacher:
    def __init__(self,sim,bounds,clearance=.008):
        if sim.num_envs!=1:raise ValueError('offline single-robot teacher')
        self.sim,self.spec=sim,sim.spec
        self.bounds=tuple(np.asarray(x,dtype=float) for x in bounds)
        self.dofs=np.r_[np.arange(self.spec.base_dof_index,self.spec.base_dof_index+6),self.spec.dof_indices]
        self.clearance=clearance
        # Expanded proximity queries belong to a separate planning model.
        self.query_model=copy.copy(sim.model);self.query_data=mujoco.MjData(self.query_model)
        self.query_model.geom_margin[:]=np.maximum(self.query_model.geom_margin,.02)
        self.wheels=set(map(int,self.spec.wheel_geom_ids))
        self.initial_clearance={}
        self.query_data.qpos[:]=sim.data[0].qpos
        mujoco.mj_forward(self.query_model,self.query_data)
        for c in self.query_data.contact[:self.query_data.ncon]:
            if self.allowed_support(c):continue
            if 0<c.dist<clearance:
                key=tuple(sorted((int(c.geom1),int(c.geom2))))
                self.initial_clearance[key]=min(self.initial_clearance.get(key,clearance),float(c.dist)*.5)

    def allowed_support(self,contact):
        pair={int(contact.geom1),int(contact.geom2)}
        return self.spec.ground_geom in pair and (pair-{self.spec.ground_geom}).issubset(self.wheels)

    def jacobian(self,model,data,point,body):
        result=np.zeros((3,model.nv))
        if body:mujoco.mj_jac(model,data,result,None,np.asarray(point),int(body))
        return result[:,self.dofs]

    def rolling_constraints(self):
        model,data,s=self.sim.model,self.sim.data[0],self.spec
        rows=[];rhs=[]
        for geom in s.wheel_geom_ids:
            points=[]
            for c in data.contact[:data.ncon]:
                if s.ground_geom in (int(c.geom1),int(c.geom2)) and int(geom) in (int(c.geom1),int(c.geom2)):
                    points.append(c.pos.copy())
            axis=data.geom_xmat[geom].reshape(3,3)[:,2]
            normal=np.array([0.,0.,1.])
            forward=np.cross(axis,normal);forward/=max(np.linalg.norm(forward),1e-8)
            lateral=np.cross(normal,forward)
            if points:point=np.mean(points,axis=0)
            else:
                radial=normal-axis*axis[2];radial/=max(np.linalg.norm(radial),1e-8)
                size=model.geom_size[geom]
                point=data.geom_xpos[geom]-size[0]*radial-size[1]*np.sign(axis[2])*axis
            basis=np.stack((forward,lateral,normal))
            rows.append(basis@self.jacobian(model,data,point,model.geom_bodyid[geom]))
            rhs.extend([0.,0.,-3.*float(point[2])])
        return np.vstack(rows),np.asarray(rhs)

    def command(self,goal,previous):
        sim,s=self.sim,self.spec;model,data=sim.model,sim.data[0]
        state=sim.read_state()[0];goal=np.asarray(goal);previous=np.asarray(previous)
        rotation=data.xmat[s.base_body].reshape(3,3)
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,jp,jr,s.tcp_site)
        task_jac=np.vstack((jp[:,self.dofs],jr[:,self.dofs]))
        current_q=np.empty(4);mujoco.mju_mat2Quat(current_q,data.site_xmat[s.tcp_site])
        error_q=quat_mul(goal[3:7],current_q*np.array([1.,-1.,-1.,-1.]))
        if error_q[0]<0:error_q=-error_q
        length=np.linalg.norm(error_q[1:])
        orientation_error=error_q[1:]*(2*np.arctan2(length,error_q[0])/max(length,1e-12))
        error=np.r_[capped(1.4*(goal[:3]-data.site_xpos[s.tcp_site]),.25),capped(1.8*orientation_error,.8)]
        following=len(goal)>=11 and goal[10]>.5
        C,rhs=self.rolling_constraints()
        if following:
            rate=heading_rate_row(rotation)
            lever=goal[:3]-np.r_[state[:2],0.]
            task_jac[:3,:3]-=np.diag([1.,1.,0.])
            task_jac[:3,3:6]-=np.outer(np.cross([0.,0.,1.],lever),rate)
            task_jac[3:,3:6]-=np.outer([0.,0.,1.],rate)
        # Weighted EE task, joint centering and gentle body posture preference.
        matrices=[task_jac*np.array([12.]*3+[5.]*3)[:,None]]
        targets=[error*np.array([12.]*3+[5.]*3)]
        regular=np.diag(np.r_[[.1,.1,.25,.35,.35,.15],[.18]*12,[.01]*4,[.08]*6])
        preferred=np.zeros(28)
        preferred[2]=-.4*(state[2]-.495)
        preferred[3:5]=-.4*np.array([np.arctan2(rotation[2,1],rotation[2,2]),
                                    np.arctan2(-rotation[2,0],np.hypot(rotation[0,0],rotation[1,0]))])
        preferred[6:]=np.clip(.1*(s.q0-state[7:29]),-.2,.2)
        preferred[18:22]=0.
        matrices.append(regular);targets.append(regular@preferred)
        # Cambered four-wheel motion can require small lateral contact motion;
        # bound it explicitly and penalize it instead of imposing an impossible
        # exact side-slip equality throughout changing body Roll/Pitch.
        matrices.append(C[1::3]*12.);targets.append(rhs[1::3]*12.)
        if following:
            base=np.zeros((3,28));base[:2,:3]=rotation.T[:2];base[2,5]=1.
            actual=np.r_[rotation.T@state[29:32],state[32:35]]
            # Rate-limited P reference uses observed previous wheel commands;
            # there is no hidden integral or trajectory clock in this teacher.
            reference_velocity=state[35:57].copy()
            reference_velocity[12:16]=previous[12:16]*s.action_scale[12:16]
            reference_root=np.linalg.lstsq(C[:,:6],rhs-C[:,6:]@reference_velocity,rcond=None)[0]
            previous_vx=float((rotation.T@reference_root[:3])[0])
            vx=np.clip(goal[7]+.8*(goal[7]-actual[0]),previous_vx-.007,previous_vx+.007)
            reference=np.array([vx,goal[8]-.8*actual[1],goal[9]-.6*(actual[5]-goal[9])])
            matrices.append(base*np.array([35.,35.,20.])[:,None]);targets.append(reference*np.array([35.,35.,20.]))
        A=np.vstack(matrices);b=np.concatenate(targets)
        H=A.T@A+np.eye(28)*1e-6;g=-A.T@b
        lower=np.r_[[-1.3,-.5,-.15,-.35,-.35,-.4],np.full(22,-np.inf)]
        upper=-lower
        position=np.r_[np.arange(12),np.arange(16,22)]
        for joint in position:
            j=joint+6;scale=s.action_scale[joint]
            ratio=s.kd[joint]/s.kp[joint]
            actual_velocity=state[35+joint]
            delta_low=max(-.008*scale,(self.bounds[0][joint]-previous[joint])*scale)
            delta_high=min(.008*scale,(self.bounds[1][joint]-previous[joint])*scale)
            lo=max((delta_low+ratio*actual_velocity)/(sim.dt+ratio),
                   (s.low[joint]-state[7+joint])/.15,-s.velocity[joint])
            hi=min((delta_high+ratio*actual_velocity)/(sim.dt+ratio),
                   (s.high[joint]-state[7+joint])/.15,s.velocity[joint])
            lower[j],upper[j]=lo,hi
        for joint in range(12,16):
            lower[6+joint]=max(self.bounds[0][joint],previous[joint]-.003)*s.action_scale[joint]
            upper[6+joint]=min(self.bounds[1][joint],previous[joint]+.003)*s.action_scale[joint]
        lower[2]=max(lower[2],(.35-state[2])/.15)
        upper[2]=min(upper[2],(.65-state[2])/.15)
        # Proximity constraints use exact engine contacts on the planning copy.
        self.query_data.qpos[:]=data.qpos;self.query_data.qvel[:]=data.qvel
        mujoco.mj_forward(self.query_model,self.query_data)
        barriers=[];barrier_rhs=[];minimum=.04
        for c in self.query_data.contact[:self.query_data.ncon]:
            if self.allowed_support(c):continue
            key=tuple(sorted((int(c.geom1),int(c.geom2))))
            margin=self.initial_clearance.get(key,self.clearance)
            minimum=min(minimum,float(c.dist))
            if c.dist>=margin+.015:continue
            body1=self.query_model.geom_bodyid[c.geom1];body2=self.query_model.geom_bodyid[c.geom2]
            normal=c.frame[:3]
            row=normal@(self.jacobian(self.query_model,self.query_data,c.pos,body2)
                        -self.jacobian(self.query_model,self.query_data,c.pos,body1))
            barriers.append(row);barrier_rhs.append(2.*(margin-float(c.dist)))
        contact_tolerance=np.tile([.003,.04,.0001],4)
        constraint=[np.eye(28),C];low=[lower,rhs-contact_tolerance];high=[upper,rhs+contact_tolerance]
        if barriers:
            constraint.append(np.asarray(barriers));low.append(np.asarray(barrier_rhs));high.append(np.full(len(barriers),np.inf))
        solver=osqp.OSQP()
        solver.setup(P=sparse.csc_matrix(np.triu(H)),q=g,A=sparse.csc_matrix(np.vstack(constraint)),
            l=np.concatenate(low),u=np.concatenate(high),verbose=False,polish=False,
            eps_abs=1e-5,eps_rel=1e-5,max_iter=6000)
        solution=solver.solve()
        if solution.info.status_val not in (1,2) or not np.isfinite(solution.x).all():
            raise RuntimeError('whole-body kinematic QP:'+solution.info.status)
        velocity=solution.x
        command=previous.copy()
        ratio=s.kd[position]/s.kp[position]
        command[position]+=((sim.dt+ratio)*velocity[6+position]-ratio*state[35+position])/s.action_scale[position]
        command[12:16]=velocity[18:22]/s.action_scale[12:16]
        command=np.clip(command,*self.bounds)
        return command,dict(qp_status=solution.info.status,qp_iterations=int(solution.info.iter),
            contact_velocity_residual=float(np.max(np.abs(C@velocity-rhs))),
            contact_bound_violation=float(np.maximum(np.abs(C@velocity-rhs)-contact_tolerance,0).max()),
            lateral_contact_velocity=float(np.abs((C@velocity-rhs)[1::3]).max()),
            rolling_contact_velocity=float(np.abs((C@velocity-rhs)[0::3]).max()),
            normal_contact_velocity_error=float(np.abs((C@velocity-rhs)[2::3]).max()),
            task_velocity_residual=float(np.linalg.norm(task_jac@velocity-error)),
            minimum_forbidden_query_clearance_m=minimum,passive_base_velocity=velocity[:6].tolist(),
            collision_constraints=len(barriers))
