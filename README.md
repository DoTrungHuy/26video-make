# 26video-make

用 Manim 制作大学生物理实验竞赛“微视频类”动画效果的项目。目前主要代码集中在**光的折射、反射与全反射对比演示**这一段动画。

> 这个仓库最初是为了比赛视频快速做动画，所以早期文件名和类名不一定完全按模块分类。现在 README 里把用途、运行方式和命名关系整理清楚，后续再扩展新动画时会更好维护。

## 项目用途

本项目适合用来制作物理实验/科普视频中的动画片段，例如：

- 光从光密介质射向光疏介质时，折射光逐渐减弱并最终发生全反射；
- 光从光疏介质射向光密介质时，不会出现同样的全反射现象；
- 用分屏对比的方式展示“入射角、折射角、临界角”的变化；
- 用发光光束、运动粒子、箭头等视觉效果增强视频表现力。

## 当前文件结构

```text
26video-make/
├── README.md                         # 项目说明文档
├── LICENSE                           # MIT License
├── contrast_reflection.py             # 原始主文件，保留兼容
└── total_internal_reflection_demo.py  # 更清楚的新入口文件，推荐以后使用
```

### 文件说明

| 文件 | 作用 | 建议 |
| --- | --- | --- |
| `contrast_reflection.py` | 原始动画主文件，包含物理计算工具类、光束组件、箭头组件和主 Scene | 先保留，不建议直接删除 |
| `total_internal_reflection_demo.py` | 新增的清晰入口文件，直接复用原来的 `SplitScreenTIR` | 推荐以后运行这个文件 |
| `README.md` | 项目说明、运行方式、命名规范 | 后续每加一个动画都更新这里 |

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

建议使用 Python 虚拟环境，避免和其他课程项目的库混在一起。

### 1. 安装 Python

建议使用 Python 3.10 或 3.11。

### 2. 安装 Manim

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
- `total_internal_reflection_demo.py`：更清楚的新入口文件；
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

## 命名调整说明

这次没有把所有内容都大改名，原因是：

1. 原文件里类比较多，直接批量改名容易破坏 Manim 运行；
2. `SplitScreenTIR` 这个类名已经比较准确，其中 `TIR` 是 Total Internal Reflection，全反射；
3. 真正最容易误解的是文件名 `contrast_reflection.py`，所以新增了更清楚的入口文件：

```text
contrast_reflection.py            原始文件名，保留兼容
↓
total_internal_reflection_demo.py 更清楚的新入口，推荐使用
```

以后如果继续整理，可以逐步拆成：

```text
scenes/total_internal_reflection.py
components/light_beam.py
components/optics_labels.py
utils/physics_optics.py
utils/math_helpers.py
```

但当前项目规模还不大，先不强行拆分，避免为了“看起来专业”而增加复杂度。

## 后续扩展建议

如果这个项目继续做大学生物理实验竞赛微视频，建议按主题新增文件：

```text
wave_interference_demo.py       # 波的干涉
polarization_demo.py            # 偏振光
thin_film_interference_demo.py  # 薄膜干涉
photoelectric_effect_demo.py    # 光电效应
mechanics_motion_demo.py        # 力学运动演示
```

每个文件最好只放一个主要动画主题，这样以后找起来更方便。

## 建议的命名规范

### 文件名

使用小写英文和下划线：

```text
total_internal_reflection_demo.py
wave_interference_demo.py
polarization_demo.py
```

### Scene 类名

使用大驼峰命名，并体现动画主题：

```python
class SplitScreenTIR(Scene):
    ...

class PolarizationDemo(Scene):
    ...

class WaveInterferenceDemo(Scene):
    ...
```

### 组件类名

用于画面元素的类，名字要能看出它负责“显示什么”：

```python
RealisticEnergyBeam
EnergyArrow
LightSource
```

### 物理计算类名

用于计算公式和物理量的类，名字要体现物理模块：

```python
OpticsPhysics
WaveOpticsEngine
ElectromagneticField
```

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

这个项目目前可以理解为：**用 Manim 为物理竞赛微视频制作“全反射条件对比”的动画素材库雏形**。后续重点不是堆很多文件，而是让每个动画主题的文件名、Scene 名、README 说明都对应清楚。
