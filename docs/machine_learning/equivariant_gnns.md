---
template: blog.html
---

# Equivariant GNNs

## Preliminaries: Group Theory and Equivariance

This part is based on the [exellent materials on equivariant neural networks by Andrew White](https://dmol.pub/dl/Equivariant.html).

- **Group** $G = \langle V, \cdot \rangle$: a set $V$ equipped with a binary operation $\cdot$ that satisfies closure, associativity, identity and inverse properties.

- **Group action** $\pi(g, x)$: $\newcommand{\cX}{\mathcal{X}} G\times\cX \to \cX$ that satisfies identity and compatibility.

!!! note "Notes on the group elements"
    We focus on groups of *transformations* here. E.g. "rotate 60° around the $z$-axis" is an element of the $SO(3)$ group, which operates on 3D points ($\newcommand{\R}{\mathbb{R}}\cX = \R^3$).

### Combining translations and rotations

- **Left coset**: $gH := \{ gh : h \in H \le G \}, \forall g \in G$

- **Normal subgroup**: $N \triangleleft G \Longleftrightarrow \forall g \in G, gN = Ng \Longleftrightarrow \forall n \in N, \forall g \in G, gng^{-1} \in N$. Normal subgroups are invariant under conjugation.
    - $\forall g \in G, gNg^{-1}$ is an automorphism of N, *i.e.*, $gNg^{-1} \in \mathrm{Aut}(N)$.

- **(Outer) semidirect product**: Given two groups $N$ and $H$ and a group homomorphism $\phi: H \to \mathrm{Aut}(N)$, their outer semidirect product $N \rtimes_\phi H$ is defined as:
    - Underlying set: $N\times H$,
    - Group operation: $(n_1, h_1) \cdot (n_2, h_2) = (n_1\phi(h_1)(n_2), h_1h_2)$.

    For $N \triangleleft G$ and $H \le G$, we can define $\phi$ as $\phi(h) = \phi_h$ where $\phi_h(n) = hnh^{-1}$.

!!! example "Example: $SE(3) = T(3) \rtimes SO(3)$"
    - We first show that $T(3) \triangleleft SE(3)$. $\forall g \in SE(3)$, we represent $g$ as $\newcommand{\vs}{\mathbf{s}}\begin{pmatrix}R&\vs\\0&1\end{pmatrix}$, $\forall t \in T(3)$, we represent $t$ as $\newcommand{\vt}{\mathbf{t}}\begin{pmatrix}I_3&\vt\\0&1\end{pmatrix}$, then
    $g^{-1}tg = \begin{pmatrix}R^{-1}&-\vs\\0&1\end{pmatrix} \begin{pmatrix}I_3&\vt\\0&1\end{pmatrix} \begin{pmatrix}R&\vs\\0&1\end{pmatrix} = \begin{pmatrix}I&R^{-1}(\vs+\vt)-\vs\\0&1\end{pmatrix} \in T(3)$.

    - We can decompose $\forall g = \begin{pmatrix}R&\vt\\0&1\end{pmatrix} \in SE(3)$ into a translation $t = \begin{pmatrix}I_3&\vt\\0&1\end{pmatrix}$ and a rotation $r = \begin{pmatrix}R&0\\0&1\end{pmatrix}$, *s.t.* $(t, r)$ is equivalent to $g$.
    We now verify that $g_2g_1 = (t_2\phi_{r_2}(t_1), r_2r_1)$.
    Since $g_2g_1 = \begin{pmatrix}R_2R_1&R_2\vt_1+\vt_2\\0&1\end{pmatrix}$,
    it suffices to check that
    $t_2\phi_{r_2}(t_1) = t_2r_2t_1r_2^{-1} = \begin{pmatrix}0&R_2\vt_1+\vt_2\\0&1\end{pmatrix}$ as $r_2r_1 = \begin{pmatrix}R_2R_1&0\\0&1\end{pmatrix}$.

### Equivariance

- **Input data** $\newcommand{\vr}{\mathbf{r}}\newcommand{\vx}{\mathbf{x}}f(\vr) = \vx$: function $\cX \to \R^n$ where the domain $\cX$ is a homogeneous space (usually just the 3D Euclidean space) and the image is an $n$-dim feature space.

- **$G$-function transform $\newcommand{\bbT}{\mathbb{T}}\bbT_g: \cX^{\R^n} \to \cX^{\R^n}, f \mapsto f^\prime$**: Given a group element $g \in G$, where $G$ is on the homogeneous space $\cX$, $G$-function transform $\bbT_g$ transforms $n$-dim functions on $\cX$ such that $f^\prime(gx) = f(x)$, *i.e.*, $\bbT_g(f)(x) = f^\prime(x) = f(g^{-1}x)$.

!!! example "Example: Translation of an image"
    **Input function**: $f(x, y) = (r, g, b)$ that specifies the pixel values for three channels at each pixel position.

    **Group element**: $t_{10,0}$ that stands for moving the image to the right by 10 pixels.

    **Output of G-function transform associated with $t_{10,0}$**: $f^\prime(x, y) = f(x-10, y)$.

- **$G$-Equivariant Neural Network $\phi: \cX^{\R^n} \to \cX^{\R^d}$**: Given a group element $g \in G$, where $G$ is on the homogeneous space $\cX$, and $\bbT_g$ and $\bbT^\prime_g$ on $n$- and $d$-dim functions, respectively, $\phi$ is a linear map such that $\phi(\bbT_g f(x)) = \bbT^\prime_g\phi(f(x)),\ \forall f(x): \cX \to \R^n$.

!!! note "Briefly..."
    "Transform then encode" has the same effect as "encode then transform".

- **$G$-Equivariant Convolution Theorem**: A neural network layer (linear map) $\phi$ is $G$-equivariant if and only if its form is a convolution operator $*$:
    $$\newcommand{\dd}{\mathrm{d}}
    \phi(f)(u) = (f * \omega)(u) = \int_G f^{\uparrow G}(u g^{-1}) \omega^{\uparrow G}(g) \dd \mu(g)
    $$
    where $f: H \to \R^n$ and $\omega: H^\prime \to \R^n$ are functions on quotient spaces $H$ and $H^\prime$, and $\mu$ is the group Haar measure.
    - Stabilizer subgroup $H_{x_0}$: $H_{x_0} = \{g \in G : gx_0 = x_0\}$ for some chosen point $x_0$ (usually the origin).
    - Orbit-stabilizer theorem: there is a bijection between the set $G/H_{x_0}$ of cosets for the stabilizer subgroup and the orbit $Gx_0$, which sends $gH_{x_0} \mapsto gx_0$.
    - Lifting up $f$: $f^{\uparrow G}(g) = f(gx_0)$.
    - Projecting down $f$: $f_{\downarrow \cX}(x) = \frac{1}{|H_{x_0}|}\sum_{u\in gH_{x_0}}f(u)$, where $g$ is found by solving $gx_0 = x$.

!!! note "Briefly..."
    There is only one way to achieve equivariance in a neural network.

!!! example "Example: SO(3)"
    + Function: defined on points on the sphere $f(x) = \sum_i \delta(x - x_i)$, where $\Vert x_i \Vert_2 = 1$.
    + Group representation: $R_z(\alpha) R_y(\beta) R_z(\gamma)$.
    + Stabilizer $x_0$: $(0, 0, 1)$. Note that $(0, 0, 0)$ is not on the sphere.
    + Stabilizer subgroup $H_{x_0}$: rotations that only involve $\gamma$, *i.e.* $R_z(0) R_y(0) R_z(\gamma)$.
    + Coset for $g = R_z(120) R_y(0) R_z(60)$: $gH_{x_0} = \{R_z(120) R_y(0) R_z(60+\gamma): \gamma \in [0, 2\pi]\}$.
    + Point in $\cX$ associated w/ this coset: $R_z(120) R_y(0) R_z(60+\gamma) x_0 = R_z(120) R_y(0) x_0$.
    + Orbit: $Gx_0 = R_z(\alpha) R_y(\beta) R_z(\gamma) x_0 = R_z(\alpha) R_y(\beta) x_0$.
    + Quotient space $G/H_{x_0}$: $SO(2)$.
    + Lifted $f$: $f^{\uparrow G}(g) = f(gx_0) = f(R_z(\alpha) R_y(\beta) x_0)$.