"""Native MuJoCo TCP error norms, numerically checked against analytic FK."""
import numpy as np
import mujoco


def tcp_error_norms(sim,goals):
    goals=np.asarray(goals,dtype=float).reshape(sim.num_envs,13)
    result=np.empty((sim.num_envs,4))
    jp=np.zeros((3,sim.model.nv));jr=np.zeros_like(jp)
    for i,d in enumerate(sim.data):
        actual=np.empty(4);mujoco.mju_mat2Quat(actual,d.site_xmat[sim.spec.tcp])
        target=goals[i,3:7].copy();mujoco.mju_normalize4(target)
        inverse=actual.copy();inverse[1:]*=-1
        error=np.empty(4);mujoco.mju_mulQuat(error,inverse,target)
        angle=2*np.arctan2(np.linalg.norm(error[1:]),abs(error[0]))
        mujoco.mj_jacSite(sim.model,d,jp,jr,sim.spec.tcp)
        result[i]=[np.linalg.norm(goals[i,:3]-d.site_xpos[sim.spec.tcp]),angle,
                   np.linalg.norm(goals[i,7:10]-jp@d.qvel),np.linalg.norm(goals[i,10:13]-jr@d.qvel)]
    return result
