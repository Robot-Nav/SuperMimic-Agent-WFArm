# Private robot assets / 私有机器人资产

[English README](../README.md) · [中文 README](../README_CN.md)

The source release does not distribute the WF-Arm model. Set `WFARM_ASSET_DIR`
to a directory you are authorized to use. An explicit `asset_dir` takes priority.
The variable and `~` are expanded by the asset resolver.

```text
<WFARM_ASSET_DIR>/
  urdf/wf-arm-tbot.urdf
  meshes/                         Files referenced by the private URDF
  config/robot.json
  config/robot_calibration.json
  config/operating_limits.json     Required by the fixed-base arm adapter
```

`robot.json` supplies the canonical 12-leg/4-wheel/6-arm joint order and default
joint positions. `robot_calibration.json` supplies the TCP parent link, local
translation and rotation. These files contain model-specific information;
the repository supplies readers rather than example geometry or calibration.

The fixed-base adapter reads `arm_joint5_lower_rad` and `arm_joint5_upper_rad`
from `operating_limits.json`. Fill these from the validated limits of the
authorized model. It does not replace missing limits with published guesses.

```bash
export WFARM_ASSET_DIR=/path/to/authorized/wf-arm-tbot
python tools/check_assets.py --fixed-arm
```

This checks required files and the operating-limit interval. The model builder
performs the actual URDF, joint-order, TCP and actuator checks. A directory with
the right filenames alone is not a validated robot model.

## 发布边界

禁止将 URDF、Xacro、MJCF、网格、CAD、标定、操作限位、动作数据和模型权重
加入发布仓库。导出模型也可能包含 FK 或几何缓存，因此 TorchScript、ONNX、
检查点和相关数据同样不分发。不要用变更后缀、内嵌字符串或压缩包绕过资产排除。

这些资源应放在仓库外；若本地工作流仍将它们放在工作区内，发布时必须使用
独立白名单目录。历史实验文档、视频、日志和内部 manifest 均不属于发布材料。

## 准备后续源码更新

```bash
# From the research workspace, build into a new directory.
python tools/prepare_release.py --output /tmp/wfarm-source-release
python tools/release_audit.py --root /tmp/wfarm-source-release

# In the isolated Git repository, check exactly the files staged/tracked.
python tools/release_audit.py --tracked
```

The release tool follows Python source dependencies and a small explicit file
allowlist. It never follows an asset directory or a compatibility symlink.
Review the selected files before every publication, including future code that
may embed robot data rather than reading it from private assets.
