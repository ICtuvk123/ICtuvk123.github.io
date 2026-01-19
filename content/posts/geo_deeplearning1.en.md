+++
date = '2026-01-17T02:53:59+08:00'
draft = false
math = true
tags = ['geometric deep learning', 'machine learning']
title = 'Introduction to Geometric Deep Learning (Continued)'
[cover]
    image = "/geo1/p0.png"
    relative = false
+++

## Group Actions
In geometric deep learning, we are often more concerned with how groups act on our data. We assume the data comes from some domain $\Omega$.
We denote the group as $\mathfrak{G}$, and define the group action on the domain $\Omega$ as a map:
$$ (\mathfrak{g},u) \mapsto \mathfrak{g}.u$$
As one might imagine, when we apply $\mathfrak{G}$ to $\Omega$, we simultaneously obtain the action of $\mathfrak{G}$ on $\mathcal{X}(\Omega)$, given by:
$$\mathfrak{g}.x(u) = x(\mathfrak{g}^{-1}.u)$$

The group actions we will encounter most frequently are linear group actions, which satisfy:
$$\mathfrak{g}.(\alpha x + \beta x') = \alpha (\mathfrak{g}.x) + \beta (\mathfrak{g}.x')$$

**Currying**
Original definition: View the transformation as a function taking two arguments: $f(\mathfrak{g},x) \mapsto \mathfrak{g}.x$ (input an action and a signal, output the transformed signal).
After Currying: View it as a function taking only one argument $\mathfrak{g}$, but returning another function $\rho(\mathfrak{g})$.
This $\rho(\mathfrak{g})$ is specifically responsible for processing the signal $x$.
Mathematically, this $\rho$ is called a Group Representation.

**Formal Definition of Group Representation**
An $n$-dimensional real representation is a map $\rho: \mathfrak{G} \to \mathbb{R}^{n \times n}$ that must satisfy the following **homomorphism condition**:
$$\rho(\mathfrak{gh}) = \rho(\mathfrak{g})\rho(\mathfrak{h})$$
*   This ensures that the algebraic structure of the group (the order of actions) is perfectly preserved in matrix operations.
*   **Unitary/Orthogonal Representations**: If $\rho(\mathfrak{g})$ is always a unitary or orthogonal matrix, the representation is called a unitary or orthogonal representation. Such representations are very important in physics and stability analysis because they preserve the norm of the signal.

Thus, written in the language of group representations, the action of $\mathfrak{G}$ on $x \in \mathcal{X}(\Omega)$ can also be defined as $\rho (\mathfrak{g})x(u) = x(\mathfrak{g}^{-1}.u)$.

## Invariance and Equivariant Functions
Since the domain $\Omega$ of the data distribution possesses symmetry, this provides us with a strong inductive bias, reducing the need for unnecessary interpolation.

If $f(\rho(\mathfrak{g})x) = f(x), \forall x \in \mathcal{X}(\Omega), \mathfrak{g} \in \mathfrak{G}$, we say that the function $f$ is $\mathfrak{G}$-invariant. In other words, the group action does not affect the output result.

![](/geo1/p1.png)

The figure above clearly illustrates the relationship between group representations, symmetry groups, signals, domains, and equivariant functions.

If $f(\rho(\mathfrak{g})x) = \rho(\mathfrak{g})f(x), \forall x \in \mathcal{X}(\Omega), \mathfrak{g} \in \mathfrak{G}$, we say that the function $f$ is $\mathfrak{G}$-equivariant. That is, applying the group action to the input or the output makes no difference.

Taking CNNs (Convolutional Neural Networks) as an example, let $\mathfrak{t} \in \mathfrak{T}$ be a translation operator. Clearly, we have:
$$conv(\mathfrak{t} * x) = \mathfrak{t} * conv(x)$$
This is because the convolution kernel is globally shared across translations (the convolution operation involves the kernel sliding across the image acting on pixels). Here, the convolution operation is what we call $f$, meaning $f$ is $\mathfrak{T}$-equivariant.

![](/geo1/p2.png)

## Isomorphism and Automorphism
1.  **Set Level**
    View the domain $\Omega$ as a set. Its only attribute is "Cardinality", i.e., how many elements are in the set.
    Structure-preserving map: Bijection.

2.  **Topological Level**
    $\Omega$ is a topological space, and we start to care about "proximity" between points.
    Structure-preserving map: Homeomorphism.

3.  **Differential Level**
    $\Omega$ is a differentiable manifold. We require not only continuity but also the ability to perform calculus.
    Structure-preserving map: Diffeomorphism, denoted as Diff($\Omega$).

As the level increases, the symmetry group becomes smaller. In fact, increasing the level is equivalent to finding subgroups (a subset satisfying group operation rules).

## Deformation Stability

### Stability to Signal Deformations
When dealing with signals (such as images $x$), we intuitively believe that slight deformations should not change the output $f(x)$. However, we face two dilemmas in mathematical definitions:

Small deformations do not form a "group": Although small deformations look like some kind of symmetry, composing multiple small deformations can result in a large deformation. Mathematically, these "small deformations" themselves do not constitute a closed transformation group.

The full diffeomorphism group is too strong: If we require invariance under all differentiable transformations ($Diff(\Omega)$), this is too excessive. Because drastic deformations change the semantics of an image (e.g., stretching the digit "3" into an "8"), we do not want the model to be "unresponsive" to such drastic changes.

Therefore, we no longer pursue absolute Invariance, but rather Geometric Stability. The core idea is: the degree of change in the output should be bounded by the "magnitude" of the deformation.
$$|| f(\rho(\tau)x) - f(x)|| \leq C c(\tau) ||x||$$
$c(\tau)$: The "complexity" or "magnitude" of the deformation. Here, we usually assume $\tau$ is a diffeomorphism.
$C$: A constant.
Intuitive understanding: If the deformation $\tau$ is very small (belonging to some symmetry subgroup $G$, such as translation, where $c(\tau)=0$), then the output remains almost unchanged. If the deformation is large, the output is allowed to change accordingly, but we know the upper bound of this change.

How to measure the magnitude of deformation? (Dirichlet Energy)
The second image gives a specific measurement method, namely Dirichlet energy.

For images defined on the Euclidean plane, if we view the deformation as a displacement field $\tau(u)$ (i.e., point $u$ is moved to $u+\tau(u)$), then the cost of deformation can be defined as:
$$c^2(\tau) := \int_{\Omega}||\nabla\tau(u)||^2du$$
This metric measures the **Elasticity** of $\tau$.
It actually measures the degree to which the deformation deviates from a "constant vector field (i.e., pure translation)".
If $\tau$ is just a simple translation (constant displacement), $\nabla\tau$ is 0, so $c(\tau)$ is 0; if $\tau$ causes drastic local distortion in the image, $\nabla\tau$ will be large, leading to an increased penalty term.

![](/geo1/p3.png)

The figure above shows the relationship between deformations of different degrees. $Aut(\Omega)$ is the automorphism group. Requiring $\mathfrak{G}$-invariance or $\mathfrak{G}$-equivariance is too harsh for images and not very applicable to real-world situations. The reason we pursue geometric stability is to find a larger group $\mathfrak{G}'$ that satisfies geometric stability.

### Stability to Domain Deformations
What if the domain of the data itself changes?
Application scenarios:
1.  Graphs: Social networks add or remove relationships (edges) over time.
2.  Manifolds: A 3D character performing non-rigid motion (like bending over), where the geometric structure of its surface changes.
Domain distance $d(\Omega, \tilde{\Omega})$: To measure how "similar" two domains are.

Metrics between domains are introduced: For graphs, Graph Edit Distance can be used. For manifolds, Gromov-Hausdorff distance can be used.
Domain stability formula: Formula (5) is a generalization of signal stability:
$$ \|f(x, \Omega) - f(\tilde{x}, \tilde{\Omega})\| \leq C \|x\| d_{\mathcal{D}}(\Omega, \tilde{\Omega}) $$
This means: If two graphs or two manifold structures are very close, the results obtained by the model processing them should also be very close.

We will discuss stability regarding domain deformations in future posts.

## References
1. Bronstein, M. M., Bruna, J., LeCun, Y., Szlam, A., & Vandergheynst, P. (2017). Geometric deep learning: going beyond Euclidean data. *IEEE Signal Processing Magazine*, 34(4), 18-42.
2. Cohen, T. S., & Welling, M. (2016). Group equivariant convolutional networks. In *International conference on machine learning* (pp. 2990-2999). PMLR.
3. Mallat, S. (2012). Group invariant scattering. *Communications on Pure and Applied Mathematics*, 65(10), 1331-1398.
4. Kondor, R., & Trivedi, S. (2018). On the generalization of equivariance and convolution in neural networks to the action of compact groups. In *International conference on machine learning* (pp. 2747-2755). PMLR.
