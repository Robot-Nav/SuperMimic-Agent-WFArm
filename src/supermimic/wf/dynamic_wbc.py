"""Offline inverse-dynamics whole-body QP with motor-target conversion.

Optimization variables:28 generalized accelerations,22 motor torques,12 local
wheel contact forces. Uses mass/bias terms, unilateral friction pyramids and
rolling acceleration constraints. Real simulation screening is still required;
the finite contact model is not a proof of global feasibility. No MPC horizon,
reference field or deployed IK controller is introduced.
"""
import mujoco
import numpy as np
import quadprog

from .gamepad import quat_mul
from .kinematic_wbc import KinematicWholeBodyTeacher, capped, heading_rate_row


class DynamicWholeBodyTeacher(KinematicWholeBodyTeacher):
    def __init__(self,sim,bounds,clearance=.008,soft_contact_acceleration=False):
        super().__init__(sim,bounds,clearance)
        self.future_data=mujoco.MjData(self.query_model)
        self.soft_contact_acceleration=bool(soft_contact_acceleration)

    def task_jacobian(self,model,data,goal):
        s=self.spec;jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jacSite(model,data,jp,jr,s.tcp_site)
        result=np.vstack((jp[:,self.dofs],jr[:,self.dofs]))
        if len(goal)>=11 and goal[10]>.5:
            R=data.xmat[s.base_body].reshape(3,3);rate=heading_rate_row(R)
            p=data.qpos[s.base_qpos_index:s.base_qpos_index+3]
            lever=goal[:3]-np.r_[p[:2],0.]
            result[:3,:3]-=np.diag([1.,1.,0.])
            result[:3,3:6]-=np.outer(np.cross([0.,0.,1.],lever),rate)
            result[3:,3:6]-=np.outer([0.,0.,1.],rate)
        return result

    def contacts(self,model,data):
        rows=[];heights=[]
        for geom in self.spec.wheel_geom_ids:
            axis=data.geom_xmat[geom].reshape(3,3)[:,2];normal=np.array([0.,0.,1.])
            forward=np.cross(axis,normal);forward/=max(np.linalg.norm(forward),1e-8)
            lateral=np.cross(normal,forward)
            radial=normal-axis*axis[2];radial/=max(np.linalg.norm(radial),1e-8)
            size=model.geom_size[geom]
            point=data.geom_xpos[geom]-size[0]*radial-size[1]*np.tanh(axis[2]/.02)*axis
            rows.append(np.stack((forward,lateral,normal))@self.jacobian(model,data,point,model.geom_bodyid[geom]))
            heights.append(float(point[2]))
        return np.vstack(rows),np.asarray(heights)

    def command(self,goal,previous):
        sim,s=self.sim,self.spec;model,data=sim.model,sim.data[0]
        goal=np.asarray(goal);previous=np.asarray(previous);state=sim.read_state()[0]
        velocity=data.qvel[self.dofs].copy();R=data.xmat[s.base_body].reshape(3,3)
        J=self.task_jacobian(model,data,goal);C,heights=self.contacts(model,data)
        eps=1e-4
        future=self.future_data;future.qpos[:]=data.qpos;future.qvel[:]=data.qvel
        mujoco.mj_integratePos(self.query_model,future.qpos,future.qvel,eps)
        mujoco.mj_forward(self.query_model,future)
        moved_goal=goal.copy()
        if len(goal)>=11 and goal[10]>.5:
            r1=future.xmat[s.base_body].reshape(3,3)
            yaw=np.arctan2(r1[1,0],r1[0,0])-np.arctan2(R[1,0],R[0,0])
            c,si=np.cos(yaw),np.sin(yaw);rz=np.array([[c,-si,0.],[si,c,0.],[0.,0.,1.]])
            p1=future.qpos[s.base_qpos_index:s.base_qpos_index+3]
            moved_goal[:3]=np.r_[p1[:2],0.]+rz@(goal[:3]-np.r_[state[:2],0.])
            moved_goal[3:7]=quat_mul(np.array([np.cos(yaw/2),0.,0.,np.sin(yaw/2)]),goal[3:7])
        Jdotv=((self.task_jacobian(self.query_model,future,moved_goal)-J)/eps)@velocity
        Cdotv=((self.contacts(self.query_model,future)[0]-C)/eps)@velocity
        current=np.empty(4);mujoco.mju_mat2Quat(current,data.site_xmat[s.tcp_site])
        dq=quat_mul(goal[3:7],current*np.array([1.,-1.,-1.,-1.]))
        if dq[0]<0:dq=-dq
        length=np.linalg.norm(dq[1:]);angle=dq[1:]*(2*np.arctan2(length,dq[0])/max(length,1e-12))
        task_velocity=J@velocity
        desired=np.r_[capped(12.*(goal[:3]-data.site_xpos[s.tcp_site])-6.*task_velocity[:3],1.5),
                      capped(10.*angle-5.*task_velocity[3:],3.)]-Jdotv
        n=62;matrices=[];targets=[]
        def add(block,target,weight,columns=slice(0,28)):
            full=np.zeros((len(target),n));full[:,columns]=block
            weights=np.broadcast_to(np.asarray(weight),(len(target),))
            matrices.append(full*weights[:,None]);targets.append(np.asarray(target)*weights)
        add(J,desired,np.r_[[10.]*3,[4.]*3])
        # Gentle posture preferences leave RP/height assistance available.
        posture=np.zeros(28)
        roll=np.arctan2(R[2,1],R[2,2]);pitch=np.arctan2(-R[2,0],np.hypot(R[0,0],R[1,0]))
        posture[:3]=-.8*velocity[:3]
        posture[2]=4.*(.495-state[2])-3.*velocity[2]
        posture[3:6]=-2.*velocity[3:6];posture[3:5]-=3.*np.array([roll,pitch])
        posture[6:]=2.*(s.q0-state[7:29])-2.*velocity[6:]
        posture[18:22]=0.
        add(np.eye(28),posture,np.r_[[.3,.3,2.,1.,1.,.3],[.12]*12,[.01]*4,[.08]*6])
        if len(goal)>=11 and goal[10]>.5:
            base=np.zeros((3,28));base[:2,:3]=R.T[:2];base[2,5]=1.
            actual=R.T@velocity[:3]
            target=np.r_[np.clip(2.*(goal[7:9]-actual[:2]),-.35,.35),np.clip(2.*(goal[9]-velocity[5]),-.5,.5)]
            add(base,target,[15.,15.,10.])
        contact_velocity=C@velocity
        contact_desired=-Cdotv-8.*contact_velocity
        contact_desired[2::3]-=80.*heights
        add(C[1::3],contact_desired[1::3],3.)
        active=heights<.006
        for wheel in range(4):
            if not active[wheel]:add(C[3*wheel+2:3*wheel+3],contact_desired[3*wheel+2:3*wheel+3],15.)
            elif self.soft_contact_acceleration:
                # MuJoCo's contact is compliant. Keep physical unilateral force
                # and friction bounds hard, but penalize acceleration mismatch
                # when finite-rate motor targets cannot realize a rigid contact.
                indices=[3*wheel,3*wheel+2]
                add(C[indices],contact_desired[indices],[20.,100.])
        add(np.eye(22),np.zeros(22),.05/s.effort,columns=slice(28,50))
        add(np.eye(12),np.tile([0.,0.,float(model.body_mass.sum())*9.81/4],4),.0005,columns=slice(50,62))
        A=np.vstack(matrices);b=np.concatenate(targets)
        H=A.T@A+np.eye(n)*1e-8;linear=-A.T@b
        full_mass=np.empty((model.nv,model.nv));mujoco.mj_fullM(model,data,full_mass)
        mass=full_mass[np.ix_(self.dofs,self.dofs)]
        bias=(data.qfrc_bias-data.qfrc_passive-data.qfrc_applied)[self.dofs]
        # Current joint-limit/equality reactions are internal forces, distinct
        # from the wheel contact forces optimized below. Retain their measured
        # value over this short control interval instead of counting them as a
        # new motor torque. This is a local dynamics approximation.
        if not mujoco.mj_isSparse(model) and data.nefc:
            internal=np.asarray(data.efc_type[:data.nefc])<int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)
            constraint_jac=np.asarray(data.efc_J).reshape(data.nefc,model.nv)
            bias-= (constraint_jac[internal].T@np.asarray(data.efc_force[:data.nefc])[internal])[self.dofs]
        dynamics=np.zeros((28,n));dynamics[:,:28]=mass;dynamics[6:,28:50]=-np.eye(22);dynamics[:,50:62]=-C.T
        constraints=[dynamics];lows=[-bias];highs=[-bias]
        row_labels=['dynamics']*28
        lower=np.full(n,-np.inf);upper=np.full(n,np.inf)
        lower[:28]=-80.;upper[:28]=80.
        lower[:6]=[-3.,-3.,-3.,-6.,-6.,-6.];upper[:6]=-lower[:6]
        position=np.r_[np.arange(12),np.arange(16,22)]
        command_low=np.maximum(self.bounds[0],previous-np.r_[np.full(12,.008),np.full(4,.003),np.full(6,.008)])
        command_high=np.minimum(self.bounds[1],previous+np.r_[np.full(12,.008),np.full(4,.003),np.full(6,.008)])
        torque_low=np.empty(22);torque_high=np.empty(22)
        for joint in position:
            torque_low[joint]=s.kp[joint]*(s.q0[joint]+s.action_scale[joint]*command_low[joint]-state[7+joint])-s.kd[joint]*velocity[6+joint]
            torque_high[joint]=s.kp[joint]*(s.q0[joint]+s.action_scale[joint]*command_high[joint]-state[7+joint])-s.kd[joint]*velocity[6+joint]
        for joint in range(12,16):
            torque_low[joint]=s.kd[joint]*(command_low[joint]*s.action_scale[joint]-velocity[6+joint])
            torque_high[joint]=s.kd[joint]*(command_high[joint]*s.action_scale[joint]-velocity[6+joint])
        lower[28:50]=np.clip(torque_low,-s.effort,s.effort)
        upper[28:50]=np.clip(torque_high,-s.effort,s.effort)
        for wheel in range(4):
            lower[50+3*wheel:53+3*wheel]=[-np.inf,-np.inf,0.]
            upper[50+3*wheel:53+3*wheel]=[np.inf,np.inf,float(model.body_mass.sum())*9.81*.8] if active[wheel] else [0.,0.,0.]
            if not active[wheel]:lower[50+3*wheel:53+3*wheel]=0.
        constraints.append(np.eye(n));lows.append(lower);highs.append(upper)
        row_labels+=['acceleration_bounds']*28+['motor_bounds']*22+['contact_force_bounds']*12
        for wheel in range(4):
            if active[wheel] and not self.soft_contact_acceleration:
                block=np.zeros((2,n));block[:,:28]=C[[3*wheel,3*wheel+2]]
                desired_contact=contact_desired[[3*wheel,3*wheel+2]]
                constraints.append(block);lows.append(desired_contact-.15);highs.append(desired_contact+.15)
                row_labels+=['rolling_normal_acceleration']*2
            # An inactive contact already has all three force coordinates
            # fixed to zero. Its eight friction faces add no restriction and
            # cause redundant active rows in the dense dual solver.
            if not active[wheel]:continue
            for theta in np.arange(8)*np.pi/4:
                row=np.zeros((1,n));row[0,50+3*wheel:53+3*wheel]=[np.cos(theta),np.sin(theta),-float(sim.xi[0,0])*np.cos(np.pi/8)]
                constraints.append(row);lows.append(np.array([-np.inf]));highs.append(np.array([0.]))
                row_labels.append('friction_pyramid')
        # Acceleration barriers around the same exact forbidden proximity pairs.
        qd=self.query_data;qd.qpos[:]=data.qpos;qd.qvel[:]=data.qvel;mujoco.mj_forward(self.query_model,qd)
        barriers=0;minimum=.04
        for c in qd.contact[:qd.ncon]:
            if self.allowed_support(c):continue
            key=tuple(sorted((int(c.geom1),int(c.geom2))));margin=self.initial_clearance.get(key,self.clearance)
            minimum=min(minimum,float(c.dist))
            if c.dist>margin+.015:continue
            grad=c.frame[:3]@(self.jacobian(self.query_model,qd,c.pos,self.query_model.geom_bodyid[c.geom2])
                                  -self.jacobian(self.query_model,qd,c.pos,self.query_model.geom_bodyid[c.geom1]))
            row=np.zeros((1,n));row[0,:28]=grad
            bound=-12.*float(grad@velocity)-36.*(float(c.dist)-margin)
            constraints.append(row);lows.append(np.array([bound]));highs.append(np.array([np.inf]));barriers+=1
            row_labels.append('collision_barrier')
        all_constraints=np.vstack(constraints);all_low=np.concatenate(lows);all_high=np.concatenate(highs)
        scale=np.r_[[1.]*3,[2.]*3,[5.]*12,[20.]*4,[5.]*6,s.effort,np.full(12,float(model.body_mass.sum())*9.81/4)]
        scaled_constraints=all_constraints*scale[None]
        row_scale=np.maximum(np.linalg.norm(scaled_constraints,axis=1),1e-6)
        normalized=scaled_constraints/row_scale[:,None]
        lo,hi=all_low/row_scale,all_high/row_scale
        equality=np.isfinite(lo)&np.isfinite(hi)&(np.abs(hi-lo)<1e-10)
        lower_only=np.isfinite(lo)&~equality;upper_only=np.isfinite(hi)&~equality
        # quadprog uses C.T @ z >= b, with equalities first. This is the
        # identical bounded QP; an active-set solve avoids accepting the
        # iteration-limited ADMM answers seen in the first physical pilot.
        constraint_rows=np.vstack((normalized[equality],normalized[lower_only],-normalized[upper_only]))
        constraint_rhs=np.r_[lo[equality],lo[lower_only],-hi[upper_only]]
        cost=H*scale[:,None]*scale[None]
        # Keep the latest local problem for an offline infeasibility audit.
        # The audit never relaxes constraints in an executed rollout.
        self.last_problem=dict(matrix=normalized,lower=lo,upper=hi,labels=np.asarray(row_labels),
            cost=cost,linear=linear*scale,variable_scale=scale)
        try:
            solution=quadprog.solve_qp((cost+cost.T)*.5,-linear*scale,
                constraint_rows.T,constraint_rhs,int(equality.sum()))
        except ValueError as exc:
            raise RuntimeError('inverse-dynamics whole-body QP:'+str(exc)) from exc
        z=solution[0]
        violation=float(np.max(np.maximum(lo-normalized@z,normalized@z-hi)))
        if not np.isfinite(z).all() or violation>1e-6:
            raise RuntimeError(f'inverse-dynamics QP returned invalid solution: normalized violation={violation:.3g}')
        x=z*scale;torque=x[28:50]
        command=previous.copy()
        command[position]=(state[7+position]+(torque[position]+s.kd[position]*velocity[6+position])/s.kp[position]-s.q0[position])/s.action_scale[position]
        command[12:16]=(velocity[18:22]+torque[12:16]/s.kd[12:16])/s.action_scale[12:16]
        command=np.clip(command,command_low,command_high)
        return command,dict(qp_status='solved',qp_solver='quadprog',qp_iterations=int(solution[3][0]),
            qp_normalized_constraint_violation=max(0.,violation),
            dynamics_residual=float(np.max(np.abs(dynamics@x+bias))),
            contact_velocity_residual=float(np.max(np.abs(contact_velocity))),
            task_acceleration_residual=float(np.linalg.norm(J@x[:28]-desired)),
            passive_base_velocity=velocity[:6].tolist(),passive_base_acceleration=x[:6].tolist(),
            contact_forces_local=x[50:].tolist(),active_contacts=active.tolist(),
            motor_torque=torque.tolist(),
            minimum_forbidden_query_clearance_m=minimum,collision_constraints=barriers)
