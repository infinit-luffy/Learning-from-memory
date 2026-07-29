# HippoAct — 真机任务设计 + 数据采集协议

> 目标：以最少的真机时间（<40 小时）采集足以支撑 §IV.D (Q3 Sim-to-Real)、§IV.F (Q5 Safety Gate) 和 supplementary video 的数据。
> 前提硬件：6/7-DOF 机械臂 + parallel gripper + 一路第三人称 RGB（RealSense D435 / Azure Kinect）+ 桌面工作区约 60×80 cm。

---

## 0. 任务选型总原则

论文有 4 个 claim，每个 claim 至少要一个真机任务把它钉死：

| Claim | 需要的任务特征 | 首选任务 |
|---|---|---|
| C1 背景/前景解耦 | 背景多变 + 有干扰物 | **T1 Pick-with-Distractors** |
| C2 跨模态绑定（本体感必要） | 富接触 / 视觉不足以完成 | **T2 Drawer Opening** |
| C3 Slot-swap → Sim-to-Real | 有清晰的 sim 对应 | **T1 + T3** |
| C4 Safety Gate OOD 检测 | 可脚本化 OOD 事件 | **T1 复用做 T-safety** |

我做完选型后，认为 3 个 core 任务已经能覆盖全部 claim。T4/T5 是"如果时间富余"的加分项，非必需。

---

## 1. Core 任务 × 3

### T1  Pick-and-Place with Distractors  ⭐ 论文封面级任务

**故事**：机器人从一堆干扰物中挑出目标物体放到指定区域。
**为什么选**：这是 slot-attention 论文的教科书 setting，视觉图也最漂亮（可以叠 slot alpha 热力图上去）。同时覆盖 C1、C3、C4。

**物理布置**：
- 60 × 80 cm 桌面，工作区中央 30 × 40 cm
- 4 张可换的桌布：素白、素黑、红色格子、木纹（每张 ~20 元，宜家/淘宝）
- 3 组打光：顶灯（默认）、侧灯（模拟窗口光）、暖色补光（黄光）
- 目标物体：红色泡沫立方体 4 cm 边长（$1，泡沫玩具积木）
- 干扰物：5–6 个不同颜色/形状的杂物（塑料水果、木块、笔盒等）
- 放置区：桌面右侧 15×15 cm 绿色胶带方框

**变化维度**（用于 Table VI 的 3 conditions）：
- Condition A：素白布 + 顶灯 + 3 个干扰物
- Condition B：木纹布 + 侧灯 + 5 个干扰物
- Condition C：红格布 + 暖光 + 5 个干扰物（其中 1 个也是红色，压力测试）

**成功判据**（预注册，写进 §IV.D）：
- 30 秒内完成
- 目标立方体落入绿色区域内（判定：中心点在框内即算）
- 无干扰物被移动 > 5 cm
- 三条同时满足 = success

**Sim 对应**：Robosuite `PickPlace` 环境，把 milk/bread/cereal 换成同尺寸方块，加干扰物到场景（改 XML 即可，30 行代码）

**采集量**：
- Stage 1 预训练：**50 条遥操作 demo**（各种桌布/干扰物）→ 约 2 小时
- Stage 2 评估：**90 trials** (每个 method 30 × 3 conditions)
- 采集时间：demo 2h + eval 5h × 5 methods = **27h**（最贵的一项）

**图 / 视频价值**：
- 论文 Fig. 5 主图：三条件下的时序帧
- Supplementary video 主段：15 秒对比 HippoAct vs TD-MPC2-pixel 在 Condition C 的表现
- Fig. 4：叠 slot alpha 热力图，展示 fast slot 追红立方体、slow slot 覆盖桌布

---

### T2  Drawer Opening  ⭐ Proprioception 必答题

**故事**：机器人拉开抽屉到指定深度。
**为什么选**：contact-rich，纯视觉 policy 常常拉过头或松脱。是把"proprioception 必要性"写进论文的最干净方式。

**物理布置**：
- 桌面固定一个小型抽屉柜（宜家 MOPPE 或类似，~$30，木质小抽屉一格约 15×20 cm）
- 抽屉把手：换 3 种（原装圆钮 / 长条铝把手 / 布带）→ 3 种视觉但相同物理接口
- 抽屉外观：贴 3 种颜色/纹理的贴纸 → 视觉变化
- 抽屉柜放在两种距离机器人基座的位置（近端 40cm、远端 60cm）

**变化维度**：
- Condition A：原装把手 + 木色 + 近端
- Condition B：铝把手 + 白色贴纸 + 近端
- Condition C：布带 + 红色贴纸 + 远端

**成功判据**：
- 抽屉从关闭状态被拉开 ≥ 15 cm
- 20 秒内完成
- 无过力事件（关节力矩不超过阈值 → 从关节读数直接判）
- 抽屉不脱轨

**Sim 对应**：Robosuite 有 `Door` 但没有 drawer；改自 RoboMimic 的 SquareInsertion 环境的 XML 加一个 slider joint，10 行 URDF

**采集量**：
- Stage 1：**30 条 demo**（3 种把手/位置组合）→ 1 小时
- Stage 2：**90 trials**
- 采集时间：demo 1h + eval 4.5h × 5 = **23h**

**独特价值**：这个任务里**视觉相同、proprioception 不同**的组是 killer——同样的把手视觉，一个是抽屉阻力大一个小，只有靠 proprioception 才能区分。可以在附录 Fig. S 里专门做一张图证明这点。

---

### T3  Two-Cup Stacking  ⭐ Long-Horizon + 记忆

**故事**：机器人抓上面的杯子，稳定地叠到下面的杯子上。
**为什么选**：多阶段任务（find → grasp → lift → align → release），最能压出 c<sub>t</sub> 情景记忆的作用。相机抖动条件下更能凸显 slot 稳定跟踪的优势。

**物理布置**：
- 两个塑料/纸杯（一次性咖啡杯即可，$1 一大包），高度 ~10 cm，直径 8 cm
- 初始位置：下杯固定在桌面中心 ± 3 cm 随机；上杯放在离下杯 20 cm 处随机方向
- 桌布：只用 T1 的素白 + 木纹两种（这个任务对背景没那么敏感）
- **关键变化维度**：相机位姿抖动。用可换基座把相机基座在 x/y/z 各方向 ±5 cm 抖动 3 种设定

**变化维度**：
- Condition A：相机标定位（默认）
- Condition B：相机右移 5 cm + 下移 3 cm
- Condition C：相机远离桌面 5 cm + 向左旋转 5°

**成功判据**：
- 30 秒内完成
- 上杯放到下杯口上后，**不接触时保持 3 秒稳定**
- 无掉落、无碰倒

**Sim 对应**：Robosuite `Stack` 环境的杯子版本（把方块换成 mesh 杯子），简单

**采集量**：
- Stage 1：**30 条 demo** → 1 小时
- Stage 2：**90 trials**
- 采集时间：demo 1h + eval 5h × 5 = **26h**（stacking 每次 trial 稍长）

**故事价值**：这个任务失败率会比 T1/T2 高（预期 55–60%），正好可以做**failure mode analysis**——写进 §IV.D 的段落，是顶会级 signal。

---

## 2. Extension 任务（时间富余再做）

### T4  Peg-in-Hole Insertion（精细动作）

**为什么可能想做**：如果 T1/T2/T3 结果都好但 reviewer 会问"你们能 handle 更精细的 task 吗"，T4 是回答。20mm 圆柱插 22mm 孔。

**代价**：需要 3D 打印一个夹具 + 5 mm 精度的插入需要更好的相机标定。**评估**：约 15h。

**跳过条件**：如果 T3 已经拿到好数字，可以不做，只在 discussion 里 mention。

### T5  Cluttered Bin Picking（背景极端变化）

**为什么可能想做**：极端 background/distractor 场景，能把 Q2 rebustness 故事推到顶点。

**代价**：15+ 物体的准备麻烦；重置耗时（每次要把物体倒回去）。**评估**：约 12h。

**跳过条件**：如果 T1 Condition C 已经足够压力测试，跳过。

---

## 3. Safety Gate Demo（复用 T1 setup）

**T-safety**：不是独立训练任务，只是评估阶段的一段特殊 protocol。用 T1 setup + T1 训练好的 HippoAct policy，脚本化 3 类 OOD 事件：

1. **人手侵入**：策略执行到中段，人手从右侧伸入工作区 5 秒，然后撤回
2. **未见物体**：桌面突然放一个训练里从没见过的物体（比如蓝色海绵）
3. **相机故障**：临时给相机盖住镜头 3 秒 / 手动降低曝光

每类事件 20 次触发。测量：
- gate 触发 → 机器人保持关节位置的成功率
- 事件结束 → 策略恢复的成功率
- 干扰事件（灯光轻变、桌布小移）下 gate 的假触发率

采集时间：**~3h**（一次性录 supplementary video + 数据）

---

## 4. Stage 1 表征预训练数据（不需要成功轨迹）

Stage 1 只需要**多样的图片**，不需要成功轨迹，所以最省事。三种来源：

| 来源 | 采集方式 | 数量 |
|---|---|---|
| 遥操作 demo | T1+T2+T3 的 110 条 demo 已经贡献 ~22K 帧 | 22K |
| 随机臂运动 | 机器人在工作区 random policy 挥手 30 分钟 × 4 种背景 | ~35K |
| 静态场景遍历 | 手动摆各种物体组合，臂静止，采多角度 | ~5K |
| **合计** | | **~62K frames** |

**这已经够 Stage 1 预训练**。DINOSAUR 论文里 slot attention 在 ~50K 帧上就收敛，我们数据只多不少。

**关键提醒**：随机臂运动阶段一定要开安全边界（不要撞到桌面）。可以脚本化：`q_target = q_home + N(0, σ)` 每 3 秒发一次目标。

---

## 5. 完整采集时间预算

| 阶段 | 时间 | 备注 |
|---|---|---|
| 硬件搭建 + 标定 | 8 h | 相机内外参、tool center point、URDF 匹配 |
| Stage 1 图像数据 | 4 h | 主要是随机臂运动 + 静态多角度 |
| T1 demo | 2 h | 50 条 × ~2 分钟 |
| T2 demo | 1 h | 30 条 |
| T3 demo | 1 h | 30 条 |
| T1 eval | 25 h | 5 method × 30 × 3 = 450 trials × ~3 min（含重置） |
| T2 eval | 22 h | 同上，稍快 |
| T3 eval | 25 h | 同上，稍慢 |
| T-safety | 3 h | 一次性 |
| Buffer / 补录 | 10 h | 总有各种意外 |
| **合计** | **~100 h** | 约 **12-13 个工作日** |

**加速手段**：
- 装一个"自动 reset"脚本：每次 trial 结束，机械臂自动扫回一个中间态；干扰物用木框限位方便复位（省 30% 时间）
- 5 个 method 的 eval 可以在**同一 session 内 round-robin**（换 checkpoint 而不换 setup），避免每次重摆桌子
- Demo 采集用双人：一人遥操作、一人重置和记录标签

---

## 6. 数据记录格式

每次 trial 保存以下内容（每 trial 一个 rosbag / hdf5）：

```
trial_YYYYMMDD_HHMMSS_taskT1_condA_methodHippoAct_seed03.h5
├── /rgb              (T, 224, 224, 3)  uint8   每 33 ms 一帧
├── /rgb_hires        (T, 480, 640, 3)  uint8   保留原始分辨率供后期用
├── /proprio/joint_q  (T, 7)            float32
├── /proprio/joint_dq (T, 7)            float32
├── /proprio/ee_pose  (T, 7)            float32 (x,y,z, quaternion)
├── /action           (T, 8)            float32 (joint_target[7] + gripper)
├── /reward           (T,)              float32
├── /done             (T,)              bool
├── /success          scalar bool
├── /failure_reason   scalar str        人工标注（见 T3 failure mode）
├── /trial_meta       json
│   ├── task         "T1"
│   ├── condition    "A"  or  "B"  or  "C"
│   ├── method       "HippoAct"
│   ├── seed         3
│   ├── datetime     "..."
│   ├── operator     "..."
│   ├── backdrop     "white_cloth"
│   ├── lighting     "top_only"
│   ├── distractors  ["yellow_lego", "green_bottle", ...]
│   └── notes        "..."
```

**关键**：`failure_reason` 每次失败**当场手工填**，别攒到最后。这是 §IV.D failure mode analysis 段的原材料，事后靠回忆没法写。

---

## 7. 常见坑（我踩过的和听说过的）

1. **相机白平衡 / 曝光自动变化** — 一定在相机 SDK 里关闭 auto-exposure/auto-WB，锁定固定值，否则同一段 policy 在阴天/晴天不 comparable
2. **URDF 与真机不一致** — Sim 里的 gripper 高度、夹爪宽度必须和真机 mm 级对齐，否则 sim-to-real 会莫名崩
3. **相机坐标系不一致** — Sim 里用 blender 世界坐标系，真机用 ROS REP-105，容易错。写一个 test 脚本：给定 EE 位姿，sim 和真机的相机 render 出来的图应该基本对齐
4. **控制频率不一致** — Sim 用 30 Hz，真机也必须 30 Hz。低于 20 Hz 会失去 planning 优势，高于 50 Hz 会让本体感读数噪声进入策略
5. **失败的定义不清晰** — 每个任务的 success 判据要预注册（写下来锁死），evaluation 时严格执行，避免"这个看起来算成功吧"的漂移
6. **同一物体的位姿分布过窄** — Demo 阶段一定人为让物体位姿覆盖 workspace 60% 面积，别让所有 demo 都是"物体在中心"
7. **相机被自己的臂遮挡** — 挑选相机位置时提前用 URDF forward-kinematics 扫一遍工作区，避免 60% 的动作路径把物体挡住

---

## 8. 优先级建议（如果时间不够）

**必须做（不做就没论文）**：T1 all conditions × 3 methods (HippoAct + TD-MPC2-pixel + VC-1)
**强烈建议**：T2 all conditions × 3 methods
**建议**：T3 all conditions × 3 methods，safety demo
**加分**：T1/T2/T3 全部 5 methods（补 DrQ-v2、R3M）
**可放弃**：T4、T5

**极端 fallback**：如果只能做 T1 一个任务，把它做扎实（5 methods、5 conditions、40 trials each）也够写一篇 workshop paper，之后加真机再投正会。

---

## 9. 下一步 checklist（今天到明天可做）

- [ ] 采购物料清单（桌布、泡沫立方体、纸杯、抽屉柜，共 ~$150）
- [ ] 相机 + 灯光位置固化（marker 贴在地面上，每次不动）
- [ ] 遥操作接口测试（能否 30 Hz 稳定采）
- [ ] 写数据记录脚本（HDF5 dumper + trial metadata 表单）
- [ ] Success 判据脚本（视觉判定 → 自动打分，避免人工偏差）
- [ ] Sim 场景搭好 T1 一个，先让 TD-MPC2-pixel 跑通并 evaluate，作为 sim-to-real gap 基线
