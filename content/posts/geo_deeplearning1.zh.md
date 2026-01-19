+++
date = '2026-01-17T02:53:59+08:00'
draft = false
math = true
tags = ['geometric deep learning', 'machine learning']
title = '几何深度学习简介(续)'
[cover]
    image = "/geo1/p0.png"
    relative = false
+++


## 群作用
在几何深度学习中，我们通常更关心的是群是如何作用到我们的数据上面的。我们假设数据来自于某个域$\Omega$。
我们将群记作$\mathfrak{G}$，将在域$\Omega$上的群作用定义为映射：
$$ (\mathfrak{g},u) \mapsto \mathfrak{g}.u$$
可以想象到的是，当我们将$\mathfrak{G}$作用在$\Omega$上的时候，我们同时也得到了$\mathfrak{G}$在$\mathcal{X}(\Omega)$上的作用，有：
$$\mathfrak{g}.x(u) = x(\mathfrak{g}^{-1}.u)$$

我们接下来接触到最多的群作用是线性群作用，线性群作用满足：
$$\mathfrak{g}.(\alpha x + \beta x') = \alpha (\mathfrak{g}.x) + \beta (\mathfrak{g}.x')$$

**柯里化**
原始定义：把变换看作一个接收两个参数的函数：$f(\mathfrak{g},x) \mapsto \mathfrak{g}.x$（输入一个动作和一个信号，输出变换后的信号）。
柯里化后：把它看作一个只接收一个参数 $\mathfrak{g}$ 的函数，但它返回的是另一个函数 $ρ(\mathfrak{g})$。
这个 $ρ(\mathfrak{g})$ 专门负责处理信号 $x$。
在数学上，这个 ρ 就被称为群表示 (Group Representation)。

**群表示的正式定义**
一个 $n$ 维实表示是一个映射 $\rho: \mathfrak{G} \to \mathbb{R}^{n \times n}$，它必须满足以下**同态条件**：
$$\rho(\mathfrak{gh}) = \rho(\mathfrak{g})\rho(\mathfrak{h})$$
* 这保证了群的代数结构（动作的顺序）在矩阵运算中得以完美保留。
* **酉表示/正交表示**：如果 $\rho(\mathfrak{g})$ 始终是酉矩阵或正交矩阵，则称该表示为酉表示或正交表示。这类表示在物理学和稳定性分析中非常重要，因为它们保持了信号的范数不变。


于是，用群表示的语言来书写，$\mathfrak{G}$作用在$x \in \mathcal{X}(\Omega)$ 同时也可以被定义为 $\rho (\mathfrak{g})x(u) = x(\mathfrak{g}^{-1}.u)$。

## 不变性与等变函数
由于数据分布的域$\Omega$具有对称性，这就为我们提供了很好的归纳偏置，可以减少很多不必要的插值。

如果 $f(\rho(\mathfrak{g})x) = f(x), \forall x \in \mathcal{X}(\Omega), \mathfrak{g} \in \mathfrak{G}$，我们则称函数$f$是$\mathfrak{G}$不变的，换句话说，就是群作用对输出结果不会造成影响。


![](/geo1/p1.png)

上图很清晰的展示出了群表示、对称群、信号、域和等变函数之间的关系。

如果 $f(\rho(\mathfrak{g})x) = \rho(\mathfrak{g})f(x), \forall x \in \mathcal{X}(\Omega), \mathfrak{g} \in \mathfrak{G}$，我们则称函数$f$是$\mathfrak{G}$等变的，也就是群作用在输入或者输出是没有区别的。

我们以CNNs(卷积神经网络)为例，记$\mathfrak{t} \in \mathfrak{T}$为一种变换算子，显然有：
$$conv(\mathfrak{t} * x) = \mathfrak{t} * conv(x)$$
这是由于卷积核是全局平移共享的（卷积操作就是卷积核沿着图片平移作用在像素上），而这里的卷积操作就是我们所说的$f$，也即，$f$是$\mathfrak{T}$等变的。

![](/geo1/p2.png)

## 同构与自同构
1. 集合层次
把域$\Omega$看作一个集合。它唯一的属性是“势”(Cardinality)，即集合里面有多少个元素。
保持结构的映射：双射

2. 拓扑层次
$\Omega$是一个拓扑空间，我们开始关心点与点之间的“邻近”。
保持结构的映射：同胚

3. 微分层级
$\Omega$是一个微分流行，我们不仅要求连续，还要求能做微积分。
保持结构的映射：微分同胚，记作Diff($\Omega$)

随着层级的增加，对称群也会变得更小。实际上，增加层级实际上等同于在寻找子群。(一个满足群运算规则的子集)

## 变形稳定性

### 对于信号变形的稳定性
在处理信号（如图像 $x$）时，我们直觉上认为微小的形变不应该改变输出 $f(x)$。但在数学定义上面临两个困境：

小形变不成“群”：微小的形变虽然看起来是某种对称，但多个小形变复合后会变成大形变。数学上，这些“小形变”本身并不构成一个闭合的变换群。

全微分同胚群太强：如果我们要求对所有可微变换$(Diff(Ω))$都保持不变，这又太过分了。因为剧烈的形变会改变图像的语义（比如把数字“3”拉扯成“8”），我们并不希望模型对这种剧烈变化也“没反应”。

所以，我们不再追求绝对的不变性（Invariance），而是追求几何稳定性（Geometric Stability）。核心在于：输出的变化程度，应该被限制在形变的“大小”之内。
$$|| f(\rho(\tau)x) - f(x)|| \leq C c(\tau) ||x||$$
$c(\tau)$：形变的“复杂程度”或“大小”。这里通常假设 $\tau$ 是一个微分同胚变换。
C：一个常数。
直观理解：如果形变$\tau$很小（属于某个对称子群 G，如平移，此时 $c(\tau)=0$），那么输出几乎不变。如果形变很大，输出允许有相应的变化，但我们可以知道这个变化的上界在哪。

如何度量形变的大小？（Dirichlet Energy）
第二张图片给出了一个具体的度量方法，即狄利克雷能量 (Dirichlet energy)。

对于定义在欧几里得平面上的图像，如果我们把形变看作是一个位移场 $\tau(u)$（即点 $u$ 被移动到了 $u+\tau(u)$），那么形变的成本可以定义为：
$$c^2(\tau) := \int_{\Omega}||\nabla\tau(u)||^2du$$
这个指标衡量了$\tau$的**弹性(Elasticity)**。
它实际上测量了该形变偏离“常数向量场（即纯平移）”的程度。
如果 $\tau$ 只是简单的平移（常数位移），$\nabla\tau$ 为 0，则 $c(\tau)$ 为 0；如果 $\tau$ 让图像局部发生了剧烈的扭曲，$\nabla\tau$ 就会很大，导致惩罚项增加。

![](/geo1/p3.png)

上图就展示出了这种不同程度之间形变的关系，$Aut(\Omega)$ 是自同构群，$\mathfrak{G}$ 不变或 $\mathfrak{G}$ 等变对图像要求太苛刻了，对实际情况适用性不强，我们追求几何稳定性的原因就是找到更大的满足几何稳定性的群$\mathfrak{G}'$

### 域变形的稳定性 (Stability to Domain Deformations)
如果数据的域也可能发生变化。
应用场景：
1. 图 (Graphs)：社交网络随时间增加或减少了一些关系（边）。
2. 流形 (Manifolds)：一个 3D 角色在做非刚性运动（比如弯腰），其表面的几何结构发生了改变。
域距离 $d(\Omega, \tilde{\Omega})$：为了衡量两个域之间有多“像”。

引入了域之间的度量：对于图，可以使用图编辑距离 (Graph Edit Distance)。对于流形，可以使用 Gromov-Hausdorff 距离。
域稳定性公式：公式 (5) 是信号稳定性的通式推广：$$ \|f(x, \Omega) - f(\tilde{x}, \tilde{\Omega})\| \leq C \|x\| d_{\mathcal{D}}(\Omega, \tilde{\Omega}) $$
这意味着：如果两个图或两个流形结构很接近，那么模型在上面处理得到的结果也应该很接近。

关于域变形的稳定性，我们将在之后的文章中讨论。

## 参考资料
1. Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P. (2017). Geometric deep learning: going beyond Euclidean data. *IEEE Signal Processing Magazine*, 34(4), 18-42.
2. Cohen, T. S., & Welling, M. (2016). Group equivariant convolutional networks. In *International conference on machine learning* (pp. 2990-2999). PMLR.
3. Mallat, S. (2012). Group invariant scattering. *Communications on Pure and Applied Mathematics*, 65(10), 1331-1398.
4. Kondor, R., & Trivedi, S. (2018). On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *International conference on machine learning* (pp. 2747-2755). PMLR.
