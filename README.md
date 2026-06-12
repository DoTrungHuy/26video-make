# 26video-make

用 Manim 制作大学生物理实验竞赛“微视频类”动画效果的项目。项目最初是为了快速做比赛视频动画，因此仓库里可能会出现一些早期试验文件、临时备份文件，以及带 `1`、`2` 这类数字后缀的历史版本。


## 项目用途

本项目适合用来制作物理实验/科普视频中的 Manim 动画片段，例如：

- 光的反射、折射与全反射；
- 光从光密介质射向光疏介质时，折射光逐渐减弱并最终发生全反射；
- 光从光疏介质射向光密介质时，不会出现同样的全反射现象；
- 用分屏对比的方式展示“入射角、折射角、临界角”的变化；
- 用发光光束、运动粒子、箭头等视觉效果增强视频表现力。

## 当前推荐入口

当前建议优先使用这个入口文件：

```text
total_internal_reflection_demo.py
```

它只是一个更清楚的入口文件，实际核心代码仍然复用原来的：

```text
contrast_reflection.py
```

这样做的原因是：`contrast_reflection.py` 是早期文件名，能运行但名字不够直观；`total_internal_reflection_demo.py` 更能说明这个动画主题是“全反射演示”。

## 文件整理说明

仓库中的文件大致分成三类：

```text
26video-make/
├── README.md                         # 项目说明文档
├── LICENSE                           # 开源许可证
├── contrast_reflection.py             # 原始主文件，保留兼容
├── total_internal_reflection_demo.py  # 推荐运行入口
├── *_1.py / *1.py                     # 可能是第一版、试验版或旧备份
├── *_2.py / *2.py                     # 可能是第二版、改进版或旧备份
└── 其他 Manim 动画脚本                # 后续按主题继续整理
```

### 怎么理解带数字的文件？

如果文件名里有 `1`、`2`，一般表示当时手动保存的“第一版 / 第二版 / 临时备份”。这种做法在刚开始写代码时很常见，但放到 GitHub 以后不推荐长期这样做。

更推荐的方式是：

```text
不要：reflection1.py、reflection2.py、reflection3.py
推荐：reflection_demo.py + Git commit 历史
```

也就是说，文件名应该表达“这个文件做什么”，版本变化交给 Git 记录。

## 当前动画内容

当前核心动画是：

```text
SplitScreenTIR
```

它是一个 Manim `Scene`，作用是分屏展示两种情况：

1. 左侧：水 → 空气，入射角增大到临界角后出现全反射；
2. 右侧：空气 → 水，作为对比，说明不是所有介质方向都会发生全反射；
3. 左下角信息面板实时显示入射角、折射角和临界角；
4. 最后给出总结：全反射的必要条件是光必须从光密介质射向光疏介质。

## 环境要求

建议使用 Python 3.10 或 3.11，并安装 Manim：

```bash
pip install manim
```

如果你使用的是 Windows，并且 Manim 安装或渲染报错，通常还需要检查：

- FFmpeg 是否安装；
- LaTeX 是否能正常渲染公式；
- 中文字体是否存在，例如 `Microsoft YaHei`。

## 如何运行

### 推荐运行方式

```bash
manim -pqh total_internal_reflection_demo.py SplitScreenTIR
```

其中：

- `-p`：渲染完成后自动预览；
- `-q h`：高质量渲染；
- `total_internal_reflection_demo.py`：推荐入口文件；
- `SplitScreenTIR`：当前主动画场景类名。

### 快速预览

如果只是检查动画效果，可以先用低质量预览：

```bash
manim -pql total_internal_reflection_demo.py SplitScreenTIR
```

### 继续使用原文件运行

原文件仍然可以运行：

```bash
manim -pqh contrast_reflection.py SplitScreenTIR
```

## 代码大致结构

`contrast_reflection.py` 现在大致可以按下面方式理解：

```text
基础数学工具
├── Vector2D / Vector3D
├── Matrix3x3 / Matrix4x4
├── Quaternion
├── ComplexMath
└── Tensor2D / Tensor3D

物理计算工具
├── JonesVector / JonesMatrix
├── StokesVector / MuellerMatrix
├── ThermodynamicsEngine
├── FluidDynamicsSolver
├── LorentzForceEngine
├── ElectromagneticField
├── DipoleRadiationEngine
├── RelativityEngine
├── CelestialMechanics
├── QuantumScattering
├── SolidStatePhysics
├── WaveOpticsEngine
└── OpticsPhysics

Manim 可视化组件
├── RealisticEnergyBeam   # 发光光束与流动粒子
├── EnergyArrow           # 光线方向箭头
└── LightSource           # 光源光晕

主动画场景
└── SplitScreenTIR        # 分屏全反射对比动画
```

注意：目前有些物理工具类并不一定都被当前 Scene 使用，它们更像是以后扩展其他物理动画时的备用模块。


## 常见问题

### 1. 为什么中文不显示？

代码中使用了：

```python
font="Microsoft YaHei"
```

如果电脑没有微软雅黑，中文可能显示异常。可以改成你电脑已有的中文字体，例如：

```python
font="SimHei"
```

### 2. 为什么公式或角度符号渲染失败？

Manim 的 `MathTex` 需要 LaTeX 环境。如果报 LaTeX 相关错误，需要安装 LaTeX，或者把复杂公式临时改成普通 `Text`。

### 3. 为什么渲染很慢？

当前动画有发光线条、粒子流动和较高质量渲染。调试时建议先用：

```bash
manim -pql total_internal_reflection_demo.py SplitScreenTIR
```

确认效果后再用高质量：

```bash
manim -pqh total_internal_reflection_demo.py SplitScreenTIR
```

## 一句话总结

这个项目目前为：**用 Manim 为物理竞赛微视频制作动画素材的仓库，其中主线动画是全反射对比演示，历史备份文件后续逐步归档或删除。**
