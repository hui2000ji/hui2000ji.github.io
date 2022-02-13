---
template: blog.html
---

# Graph Neural Networks

## Preliminaries

### Permutation In-/Equi-varience on Graphs

Let $\mathbf{P}$ be a permutation matrix.

**Invariance** &emsp; Invariant function outputs the same value for all permutations of the node index. This is required by graph-level tasks.

$$
f(\mathbf{PX}, \mathbf{PAP}^\top) = f(\mathbf{X}, \mathbf{A})
$$

**Equivariance** &emsp; Equivariant function outputs the permuted value for all permutations of the node index. This is required by node-/edge-level tasks.

$$
f(\mathbf{PX}, \mathbf{PAP}^\top) = \mathbf{P}f(\mathbf{X}, \mathbf{A})
$$
### General Convolution

#### Continuous 1-D Convolution

$$
\DeclareMathOperator{\mean}{\mathrm{mean}}
\DeclareMathOperator{\diag}{\mathrm{diag}}
\DeclareMathOperator{\relu}{\mathrm{ReLU}}
\DeclareMathOperator{\elu}{\rm{E{\small LU}}}
\DeclareMathOperator{\leakyrelu}{\mathrm{LeakyReLU}}
\DeclareMathOperator{\batchnorm}{\mathrm{BN}}
\DeclareMathOperator{\softmax}{\mathrm{softmax}}
\DeclareMathOperator{\msg}{\rm M{\small SG}}
\DeclareMathOperator{\agg}{\rm A{\small GG}}
\DeclareMathOperator{\mlp}{\rm{M{\small LP}}}
\DeclareMathOperator{\ltwonorm}{\mathscr{l}_2\mathrm{-norm}}
\def\dd{\mathrm{d}}
(f * g)(t) := \int_{-\infty}^{\infty} f(\tau) g(t-\tau) \dd\tau
$$

#### Discrete 1-D Convolution

$$
(f * g)[n] := \sum_{m=-\infty}^{\infty} f[m] g[n-m]
$$

This can be viewed as a multiplication with a circulant matrix.

$$
\begin{pmatrix} \dots \\ h_{0} \\ h_{1} \\ h_{2} \\ \dots \end{pmatrix} = 
\begin{pmatrix}
    \dots & \dots &  \dots &  \dots & \dots \\
    \dots & g_{0} & g_{-1} & g_{-2} & \dots \\
    \dots & g_{1} &  g_{0} & g_{-1} & \dots \\
    \dots & g_{2} &  g_{1} &  g_{0} & \dots \\
    \dots & \dots &  \dots &  \dots & \dots
\end{pmatrix}
\begin{pmatrix} \dots \\ f_{0} \\ f_{1} \\ f_{2} \\ \dots \end{pmatrix}
$$

where $h = f * g$. For finite-length $f$ and $g$, we could diagonize the circulant matrix

$$
\mathbf{h} = \mathbf{\Phi}\diag(\hat{g}_1, \dots, \hat{g}_n)\mathbf{\Phi}^\top \mathbf{f} = \mathbf{\Phi}(\mathbf{\Phi}^\top \mathbf{f})\circ(\mathbf{\Phi}^\top \mathbf{g}).
$$

A more detailed tutorial on circulant matrix and its relation with Fourier transforms can be found [here](https://www.researchgate.net/publication/325168238_Discovering_the_Fourier_Transform_A_Tutorial_on_Circulant_Matrices_Circular_Convolution_and_the_DFT).

#### Discrete 2-D Convolution

$$
(f * g)[i, j] := \sum_{a, b} f[a, b] g[i-a, j-b]
$$
With $(n_h, n_w)$ input, $(k_h, k_w)$ kernel, $(p_h, p_w)$ padding and $(s_h, s_w)$ stride, the output shape will be

$$\lfloor(n_h-k_h+p_h+s_h)/s_h\rfloor \times \lfloor(n_w-k_w+p_w+s_w)/s_w\rfloor.$$

### Spectral Graph Theory

We refer the readers to this elegant tutorial ([part1](https://web.stanford.edu/class/cs168/l/l11.pdf), [part2](https://web.stanford.edu/class/cs168/l/l12.pdf)), which is a subset of Stanford's [ *CS 168: The Modern Algorithmic Toolbox* ](https://web.stanford.edu/class/cs168/) course.

### Graph Convolution

There are two approaches to defining graph convolution: spectral methods and spatial methods [^1]. The former defines convolution through graph Fourier transform, and the latter defines convolution as aggregating messages coming from the neighborhood.

#### Spectral Graph Convolution

Let $G = \langle V=[n], E, \mathbf{W} \rangle$ be an undirected weighted graph.
The unnormalized graph Laplacian is $\mathbf{L} = \mathbf{D} - \mathbf{W}$, where the degree matrix $\mathbf{D} = \diag \big( \sum_{j \ne i} w_{ij} \big)$.

The Laplacian has an eigendecomposition $\mathbf{L} = \mathbf{\Phi}\mathbf{\Lambda}\mathbf{\Phi}^\top$. Changing the eigenvalues in $\mathbf{\Lambda}$ expresses any operation that commutes with $\mathbf{L}$.
Given a graph signal $\mathbf{f} = (f_1, \dots, f_n)^\top$, its *graph Fourier transform* is given by $\hat{\mathbf{f}} = \mathbf{\Phi}^\top \mathbf{f}$.
The *spectral convolution* of two graph signals is defined as (transform → multiplication → reverse transform)

$$
\mathbf{f}*\mathbf{g} = \mathbf{\Phi}\diag(\hat{g}_1, \dots, \hat{g}_n)\mathbf{\Phi}^\top \mathbf{f} = \mathbf{\Phi}(\mathbf{\Phi}^\top \mathbf{f})\circ(\mathbf{\Phi}^\top \mathbf{g}).
$$

!!! note "Why can't we define spectral graph convolution in the spatial domain as in [here](#discrete-2-d-convolution)?"
    Because there’s no notion of space or direction inherent to a graph.

    To get anisotropy back, one can use natural edge features if applicable, or adopt mechanisms invariant by index permutation yet treat neighbors differently, e.g. node degrees ([MoNet](#monet)), edge gates ([Gated GCN](#gatedgcn)) and attention ([GAT](#GAT)).

Though theoretically elegant, these methods suffer from several drawbacks, namely

- Filters are basis-dependent → only applies to transductive setting
- $O(n)$ parameters per layer
- $O(n^2)$ computation of GFT and IGFT
- No guarantee of spatial localization of filters

As shown above, directly learning the eigenvalues $\diag(\hat{g}_1, \dots, \hat{g}_n)$ is typically inappropriate.
We instead "spatialize" the spectral graph convolution by learning a function of the eigenvalues of the Laplacian. Observe that in the Euclidean setting, localization in space is equivalent to smoothness in the frequency domain

$$
\int_{-\infty}^\infty \left|x\right|^{2k} \left|f(x)\right|^2 \dd x =
\int_{-\infty}^\infty \left|\frac{\partial^k \hat{f}(\omega)}{\partial \omega^k}\right|^2 \dd \omega.
$$

To encourage smooth localization of filters and reduce computational cost, we parameterize the filter $\mathbf{g}$ using a smooth spectral transfer function $\tau(\lambda)$. In this way, application of the filter becomes

$$
\mathbf{f}*\mathbf{g} = \tau(\mathbf{L})\mathbf{f} = \mathbf{\Phi} \tau(\mathbf{\Lambda}) \mathbf{\Phi}^\top \mathbf{f}
$$

where $\tau(\mathbf{\Lambda}) = \diag(\tau(\lambda_1), \dots, \tau(\lambda_n))$. If we parameterize $\tau$ as a $K$-th order polynomial, this expression is now $K$-localized since the $K$-th order polynomial of the Laplacian $\mathbf{L}$ depends only on the $K$-th order neighborhood of each node. It also enjoys $O(1)$ parameters per layer and $O(NK)$ computational complexity.

#### Spatial Graph Convolution

Define neighborhood of node $i$ as $\mathcal{N}(i) := \{j : (i,j) \in E\}$,
and the extended neighborhood of node $i$ as $\tilde{\mathcal{N}}(i) := \mathcal{N}(i) \cup \{i\}$. A message passing GNN can be generalized as:

$$
\begin{align}
\mathbf{m}^{l+1}_u &= \msg^{l} (\mathbf{h}^{l}_u) \\
\mathbf{h}^{l+1}_v &= \agg^{l} (\{\mathbf{m}_u^{l+1}: u \in \mathcal{N}(v) \}, \mathbf{h}^{l}_v ).
\end{align}
$$

In the [MoNet](#monet) paper, the author defined another general form for spatial graph convolution. For each edge $(x, y) \in E$, they define a $d$-dimensional pseudo-coordinates $\mathbf{u}(x, y)$ similar to the $\msg$ function in the above definition. They also define a $\mathbf{\Theta}$-parameterized weighting kernel $w_{\mathbf{\Theta}} = (w_1(\mathbf{u}), \dots, w_K(\mathbf{u}))$, where $K$ is the number of kernels. The patch operator can therefore be written as

$$
\mathcal{D}_k(x)f = \sum_{y \in \mathcal{N}(x)} w_k\big( \mathbf{u}(x, y) \big) f(y),\quad k \in [K].
$$

A spatial generalization of the convolution on non-Euclidean domains is then given by

$$
(f*g)(x) = \sum_{k=1}^K g_k \mathcal{D}_k(x)f
$$

Several CNN-type geometric deep learning methods on graphs and manifolds can be obtained as a particular setting of MoNet. $\bar{\mathbf{u}}_k$, $\bar{\sigma}_\rho$, $\bar{\sigma}_\theta$ denote fixed parameters of the weight functions.

|Method      |Pseudo-coordinates $\mathbf{u}(x, y)$|Weight function $w_k(\mathbf{u})$      |
|------------|-------------------------------------|---------------------------------------|
|CNN         |$\mathbf{x}(y) - \mathbf{x}(x)$      |$\delta(\mathbf{u}-\bar{\mathbf{u}}_k)$|
|GCN         |$\deg(x), \deg(y)$                   |$\frac{1}{\sqrt{u_1u_2}}$|
|Geodesic CNN|$\rho(x,y), \theta(x,y)$             |$\exp \bigg(-\frac{1}{2}(\mathbf{u}-\bar{\mathbf{u}}_k)^\top \Big(\begin{smallmatrix} \bar{\sigma}_\rho^2 & \\ & \bar{\sigma}_\theta^2 \end{smallmatrix}\Big)^{-1} (\mathbf{u}-\bar{\mathbf{u}}_k) \bigg)$|
|MoNet       |$\tanh \bigg(\mathbf{A} \Big(\begin{smallmatrix} \deg(x)^{-1/2} \\ \deg(y)^{-1/2} \end{smallmatrix}\Big) +\mathbf{b}\bigg)$|$\exp \bigg( -\frac{1}{2} (\mathbf{u}-\mathbf{\mu}_k)^\top \Sigma_k^{-1} (\mathbf{u}-\mathbf{\mu}_k) \bigg)$|

#### Relation between Spectral and Spatial Methods

|Notation|ChebNet filter|Spatial filter|
|-----|-----|-----|
|vector|$\mathbf{h} = \sum_{k=0}^K \alpha_k \mathbf{L}^k \mathbf{f}$|$\mathbf{h} = (\mathcal{D}f)\mathbf{g}$|
|element|$h_i = \sum_{k=0}^K \alpha_k (\mathbf{L}^k \mathbf{f})_i$|$h_i = \sum_{k=1}^K g_k (\mathcal{D}f)_i$|

ChebNet is a particular setting of spatial convolution with local weighting functions given by the powers of the Laplacian $w_k(x, y) = \mathbf{L}^{k-1}_{x,y}$.

### GNN vs. Random Walk Models

Random walk objectives inherently capture local similarities
From a representation perspective, random walk node embedding models emulate a convolutional GNN.

**Corollary 1**: Random-walk objectives can fail to provide useful signal to GNNs.

**Corollary 2**: At times, DeepWalk can be matched by an *untrained* conv-GNN.

## Message Passing GCNs

Message passing Neural Networks (MPNNs) can be divided into isotropic and anisotropic GCNs by whether the node update function $\agg$ treats every edge direction equally.
The terminology of MPNNs and WL-GNNs and isotropic and anisotropic GNNs are from [^4].

### Isotropic GCNs

#### GCN

> 2017 ICLR - Semi-Supervised Classification with Graph Convolutional Networks [^5]

$$
\mathbf{h}^{l+1}_v = \relu \big( \mathbf{U}^l \mean \{ \mathbf{h}^l_u : u \in \mathcal{N}_v \}\big)
$$

In a normalized version, the $\mean$ operator becomes 

$$
\mean \{ \mathbf{h}^l_u : u \in \mathcal{N}_v \} = \sum_{u \in \mathcal{N}_v} \frac{\tilde{w}_{ij}}{\sqrt{\tilde{d}_i}\sqrt{\tilde{d}_j}} \mathbf{h}^l_u
$$

with $\tilde{\mathbf{W}} = \mathbf{W} + \mathbf{I}$ and $\tilde{\mathbf{D}} = \diag\big( \sum_{j} \tilde{w}_{ij} \big)$.

#### GraphSAGE

> 2017 NeurIPS - Inductive Representation Learning on Large Graphs [^6]

$$
\mathbf{h}^{l+1}_v = \ltwonorm \bigg( \relu \Big(
    \mathbf{U}^l \big[
        \mathbf{h}^l_v \Vert \mean \left\{
            \mathbf{h}^l_u : u \in \mathcal{N}_v
        \right\}
    \big]
\Big) \bigg)
$$

### Anisotropic GCNs

#### GAT

> 2018 ICLR - Graph Attention Networks [^7]

$K$-head attention.

$$
\begin{align}
\alpha_{ij}^{k,l} &= \softmax_{j \in \tilde{\mathcal{N}}(i)} \bigg(
    \leakyrelu \Big( 
        {\mathbf{a}^{k,l}}^{\top} [\mathbf{U}^{k,l}\mathbf{h}^l_i \Vert \mathbf{U}^{k,l}\mathbf{h}^l_j] 
    \Big) 
\bigg) \\
\mathbf{h}^{l+1}_i &= \Vert_{k=1}^K \bigg( \elu \Big(
    \sum_{j \in \tilde{\mathcal{N}}(i)} \alpha_{ij}^{k,l}\mathbf{U}^{k,l}\mathbf{h}^l_j 
\Big) \bigg)
\end{align}
$$

#### MoNet

> 2017 CVPR - Geometric deep learning on graphs and manifolds using mixture model CNNs [^8]

$$
\begin{align}
u_{ij}^l &= \tanh \bigg(\mathbf{A} \begin{pmatrix} \deg(x)^{-1/2} \\ \deg(y)^{-1/2} \end{pmatrix} +\mathbf{b}\bigg) \\
w_{ij}^{kl} &= \exp \bigg( -\frac{1}{2} (\mathbf{u}-\mathbf{\mu}_k)^\top \Sigma_k^{-1} (\mathbf{u}-\mathbf{\mu}_k) \bigg) \\
\mathbf{h}^{l+1}_i &= \relu \Big( \sum_{k=1}^K 
    \sum_{j \in \mathcal{N}(i)} w_{ij}^{k,l}\mathbf{U}^{k,l}\mathbf{h}^l_j
\Big)
\end{align}
$$

#### GatedGCN

> 2018 ICLR - Residual Gated Graph ConvNets [^9]

$$
\begin{align}
e_{ij}^{l+1} &= \softmax_{j \in \mathcal{N}(i)} \bigg( \hat{e}_{ij}^{l} + \relu \Big( \batchnorm \big(
    \mathbf{A}^l\mathbf{h}_i^l + \mathbf{B}^l\mathbf{h}_j^l + \mathbf{C}^l\hat{e}_{ij}^{l}
\big) \Big) \bigg) \\
h_i^{l+1} &= h_i^l + \relu \Big( \batchnorm \big(
    \mathbf{U}^l\mathbf{h}_i^l +
    \sum_{j \in \mathcal{N}_i} e^{l+1}_{ij} \odot \mathbf{V}^l\mathbf{h}_j^l
\big) \Big)  \\
\end{align}
$$

## Weisfeiler-Lehman GNNs

This line of research [^2] is based on the Weisfeiler-Lehman (WL) graph isomorphism test, which aims to develop provably expressive GNNs.

### GIN

> 2019 ICLR - How Powerful are Graph Neural Networks [^10]

The GIN architecture is a provable 1-WL GNN based on the Weisfeiler-Lehman Isomorphism Test.

$$
\mathbf{h}_i^{l+1} = \relu \Big( \mlp^l \big( (1+\epsilon) \mathbf{h}_i^l + \sum_{j \in \mathcal{N}_i} \mathbf{h}_j^l \big) \Big)
$$

A small improvement that incorporates edge features has the following form:

$$
\mathbf{h}_i^{l+1} = \relu \bigg( \mlp^l \Big(
    (1+\epsilon) \mathbf{h}_i^l + \sum_{j \in \mathcal{N}_i} \relu \big( \mathbf{h}_j^l + \mathbf{e}_{ij}^l \big)
\Big) \bigg)
$$

GIN also proposed an injective graph-level readout

$$
\mathbf{h}_G = \Vert_{l=0}^L \sum_{v \in V} \mathbf{h}_v^l
$$

### 3WL-GNN

> 2019 NeurIPS - Provably powerful graph networks [^11]

3-WL GNNs uses rank-2 tensors ($n \times n \times d$) while being 3-WL provable.
This 3-WL model improves the space/time complexities of $k$-GNN [^3] from $O(n^3)$/$O(n^4)$ to $O(n^2)$/$O(n^3)$ respectively.

We first introduce the $n \times n \times (1 + d_{\mathrm{node}} + d_{\mathrm{edge}})$ input tensor

$$\mathbf{h}^0 = \big[ A \Vert \mathbf{X}^{(\mathrm{node})} \Vert \mathbf{X}^{(\mathrm{edge})} \big]$$

where $\mathbf{X}^{(\mathrm{node})}_{i,i,:}$ is the $i$-th node feature, and $\mathbf{X}^{(\mathrm{edge})}_{i,j,:}$ is the $(i,j)$-th edge feature.
The model is defined as

$$
\mathbf{h}^{l+1} = \big[
    \mlp_1(\mathbf{h}^{l})\mlp_2(\mathbf{h}^{l})
    \Vert
    \mlp_3(\mathbf{h}^{l})
\big]
$$

and implemented (I don't know why, either) as

$$
\mathbf{h}^{l+1} = \mlp_3\Big( \big[
    \mlp_1(\mathbf{h}^{l})\mlp_2(\mathbf{h}^{l})
    \Vert
    \mathbf{h}^{l}
\big] \Big)
$$

where the tensor multiplication is defined as an einsum of `ipk,pjk->ijk`, *i.e.*, per-feature matrix multiplication.


[^1]: TowardsDataScience blogpost - [Graph Convolutional Networks for Geometric Deep Learning](https://towardsdatascience.com/graph-convolutional-networks-for-geometric-deep-learning-1faf17dee008); NeurIPS 2017 Tutorial - Geometric Deep Learning [slides](http://geometricdeeplearning.com/slides/NIPS-GDL.pdf) [video](https://www.youtube.com/watch?v=LvmjbXZyoP0)
[^2]: [Invariant Graph Networks](https://slideslive.com/38917604/invariant-graph-networks), ICML 2019 Workshop - Learning and Reasoning with Graph-Structured Representations
[^3]: [Graph Neural Networks and Graph Isomorphism](https://slideslive.com/38917609/graph-neural-networks-and-graph-isomorphism), ICML 2019 Workshop - Learning and Reasoning with Graph-Structured Representations; For the $k$-GNN paper published on AAAI 2019, see [Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks](https://aaai.org/ojs/index.php/AAAI/article/view/4384).
[^4]: [Benchmarking Graph Neural Networks](https://slideslive.com/38930553/benchmarking-graph-neural-networks), ICML 2020 Workshop - Graph Representation Learning and Beyond
[^5]: [Semi-Supervised Classification with Graph Convolutional Networks](https://openreview.net/forum?id=SJU4ayYgl), ICLR 2017; Thomas Kipf's [blog post](https://tkipf.github.io/graph-convolutional-networks/) explaining the basic ideas of GCN.
[^6]: [Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216), NeurIPS 2017
[^7]: [Graph Attention Networks](https://arxiv.org/abs/1710.10903), ICLR 2018; See this [blog post](https://petar-v.com/GAT/) by Petar Veličković for more detailed explanations.
[^8]: [Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/abs/1611.08402), CVPR 2017
[^9]: [Residual Gated Graph ConvNets](https://openreview.net/forum?id=HyXBcYg0b), ICLR 2018
[^10]: [How Powerful are Graph Neural Networks](https://openreview.net/forum?id=ryGs6iA5Km), ICLR 2019
[^11]: [Provably powerful graph networks](https://openreview.net/pdf/7fd2f63da562a351552b127a8602449f757fd7c0.pdf), NeurIPS 2019
[^12]: Theoretical Foundations of Graph Neural Networks, Computer Laboratory Wednesday Seminar, 17 February 2021 [video](https://www.youtube.com/watch?v=uF53xsT7mjc), [slides](https://petar-v.com/talks/GNN-Wednesday.pdf)
