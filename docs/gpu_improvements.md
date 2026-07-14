# GPU-Based Improvements

We outline some improvements for the algorithm computing the dimension of a neurovariety $\mathcal{V}_{\mathbf{d},r}$ with given architecture $\mathbf{d}$ and activation degree $r$. The point is to work over finite fields and compute $\operatorname{rank}(\operatorname{Jac}(\Psi_{\mathbf{d},r})(\theta)\bmod p)$, the rank of the Jacobian of the parameter map working over $\mathbb{F}_p$ and for a randomly chosen weight $\theta$.

## Explanation

Some slowdowns in using backprop.py from the original implementations of Kubjas et. al. and Kileel et. al as found in https://mathrepo.mis.mpg.de/PolynomialNeuralNetworks/ pertains to two steps.

(a) The need to take 2*nsamples to completely determine the Jacobian matrix's entries.

(b) The need to use monomials_pinv = monomials.pseudoinverse() as part of that same process.

(c) The prescence of a doubly nested loop which scales with the architecture size and the number of samples taken. This leads to a massive slowdown for architectures that are deep and/or have large hidden widths.

(d) Rank computation is heavy.


**Our solution to these issues:** Use a unisolvent set to determine the coefficients of a homogeneous polynomial by evaluation. Use tensors to replace the double nested loop.
 
# GPU Algorithm for Computing the Neurovariety Dimension

Consider a polynomial neural network with architecture

$$
\mathbf d=(d_0,d_1,\ldots,d_L),
$$

where $d_0$ is the input dimension, $d_L$ is the output dimension, and each hidden layer uses the coordinatewise activation

$$
z\longmapsto z^r.
$$

The final layer is linear. The collection of all weight matrices is denoted by

$$
W=(W_1,\ldots,W_L),
$$

where

$$
W_\ell\in\mathbb F_p^{d_\ell\times d_{\ell-1}}.
$$

The goal of the computation is to determine the dimension of the neurovariety obtained as the Zariski closure of the set of polynomial maps represented by this network.

## The coefficient map and its differential

For a fixed choice of weights $W$, the network defines a polynomial map

$$
F_W:\mathbb F_p^{d_0}\longrightarrow\mathbb F_p^{d_L}.
$$

Write its output coordinates as

$$
F_W(x)
=
\begin{bmatrix}
F_{W,1}(x)\
\vdots\
F_{W,d_L}(x)
\end{bmatrix}.
$$

There are $L-1$ hidden activation layers, so each coordinate $F_{W,j}(x)$ is a homogeneous polynomial of degree

$$
D=r^{L-1}.
$$

The vector space of homogeneous degree-$D$ polynomials in $d_0$ variables has dimension

$$
M=\binom{D+d_0-1}{d_0-1}.
$$

Therefore, the ambient coefficient space for all $d_L$ outputs has dimension

$$
Md_L.
$$

The weights determine the coefficients of the output polynomials. This gives a polynomial parameterization from the space of weights to the coefficient space. At a sufficiently general point $W$, the dimension of the image is the rank of the differential of this parameterization.

The computation therefore reduces to constructing an appropriate Jacobian matrix with respect to the weights and computing its rank.

## Meaning of $\nabla_W$

The symbol $W$ denotes all weight matrices in the network, not one individual matrix.

For one scalar output coordinate $F_{W,j}(x)$, the notation

$$
\nabla_WF_{W,j}(x)
$$

means the collection of derivatives of $F_{W,j}(x)$ with respect to every scalar entry of every weight matrix.

Layer by layer, this is

$$
\nabla_WF_{W,j}(x)
=
\left(
\frac{\partial F_{W,j}(x)}{\partial W_1},
\ldots,
\frac{\partial F_{W,j}(x)}{\partial W_L}
\right).
$$

For each $\ell$, the derivative

$$
\frac{\partial F_{W,j}(x)}{\partial W_\ell}
$$

is a matrix having the same shape as $W_\ell$.

The code flattens these matrices and concatenates them into one vector. If

$$
N=\sum_{\ell=1}^L d_{\ell-1}d_\ell
$$

is the total number of scalar weights, then the flattened gradient is a vector of length $N$:

$$
\operatorname{vec}\left(\nabla_WF_{W,j}(x)\right)\in\mathbb F_p^N.
$$

Because the network output is vector-valued, the derivative of the complete output with respect to the weights is more precisely the Jacobian

$$
D_WF_W(x)=
\begin{bmatrix}
\operatorname{vec}(\nabla_WF_{W,1}(x))\
,\ldots,
\operatorname{vec}(\nabla_WF_{W,d_L}(x))
\end{bmatrix}^T.
$$

This is a $d_L\times N$ matrix.

The GPU algorithm computes this weight Jacobian for many input points simultaneously.

## Choosing the input points

The algorithm chooses exactly $M$ input points

$$
x^{(1)},\ldots,x^{(M)}.
$$

These points are chosen to be unisolvent for homogeneous degree-$D$ polynomials. This means that a homogeneous degree-$D$ polynomial is uniquely determined by its values at these points.

The code works on the affine chart where the first input coordinate is $1$. It uses points of the form

$$
(1,\alpha_1,\ldots,\alpha_{d_0-1}),
$$

where each $\alpha_i$ is a nonnegative integer and

$$
\alpha_1+\cdots+\alpha_{d_0-1}\leq D.
$$

There are exactly

$$
\binom{D+d_0-1}{d_0-1}=M
$$

such points.

The points are stored as the rows of a matrix

$$
X\in\mathbb F_p^{M\times d_0}.
$$

The sample index is therefore treated as a matrix or tensor dimension rather than as a Python loop.

## Batched forward propagation

The initial activation matrix is

$$
A_0=X.
$$

For every hidden layer $\ell=1,\ldots,L-1$, the code computes

$$
Z_\ell=A_{\ell-1}W_\ell^T
$$

and then applies the activation entrywise:

$$
A_\ell=Z_\ell^{\circ r}.
$$

The notation $Z_\ell^{\circ r}$ means that every entry of $Z_\ell$ is raised to the $r$-th power.

The matrix shapes are

$$
A_{\ell-1}\in\mathbb F_p^{M\times d_{\ell-1}},
$$

$$
W_\ell^T\in\mathbb F_p^{d_{\ell-1}\times d_\ell},
$$

and therefore

$$
Z_\ell,A_\ell\in\mathbb F_p^{M\times d_\ell}.
$$

One matrix multiplication computes the layer output for all $M$ input samples simultaneously.

The code saves both the activations $A_\ell$ and the preactivations $Z_\ell$ because they are needed during backpropagation.

## Derivatives with respect to the final layer

The final layer is linear:

$$
F_W(x)=W_LA_{L-1}(x).
$$

For the $j$-th output coordinate,

$$
F_{W,j}(x)
=
\sum_{b=1}^{d_{L-1}}(W_L)*{jb}A*{L-1,b}(x).
$$

Therefore,

$$
\frac{\partial F_{W,j}(x)}
{\partial (W_L)_{kb}}
=
\begin{cases}
A_{L-1,b}(x),&k=j,\
0,&k\neq j.
\end{cases}
$$

Thus, the derivative with respect to the $j$-th row of $W_L$ is simply the last hidden activation vector.

The code inserts these derivatives directly into the columns of the Jacobian corresponding to $W_L$.

## The backpropagation tensor

For every hidden layer, the code defines a tensor $\delta_\ell$ by

$$
\delta_\ell[s,j,a]
=
\frac{\partial F_{W,j}(x^{(s)})}
{\partial Z_{\ell,a}(x^{(s)})}.
$$

The indices have the following meanings:

* $s$ is the sample index;
* $j$ is the output coordinate;
* $a$ is a neuron in hidden layer $\ell$.

Consequently,

$$
\delta_\ell\in\mathbb F_p^{M\times d_L\times d_\ell}.
$$

In ordinary backpropagation, one usually fixes one sample and one scalar output. The corresponding $\delta_\ell$ is then only a vector.

The GPU algorithm retains all samples and all output coordinates as tensor axes. This allows the derivatives for every sample and every output to be computed simultaneously.

## Initializing the backward pass

At the last hidden layer,

$$
\frac{\partial F_{W,j}}{\partial A_{L-1,a}}
=
(W_L)_{ja}.
$$

Since

$$
A_{L-1,a}=Z_{L-1,a}^r,
$$

we have

$$
\frac{\partial A_{L-1,a}}
{\partial Z_{L-1,a}}
=
rZ_{L-1,a}^{r-1}.
$$

The chain rule gives

$$
\delta_{L-1}[s,j,a]
=
(W_L)_{ja}rZ_{L-1,a}(x^{(s)})^{r-1}.
$$

The code broadcasts $W_L$ across the sample dimension and multiplies it entrywise by

$$
rZ_{L-1}^{\circ(r-1)}.
$$

This initializes the backward tensor for all samples and all output coordinates at once.

## Computing the gradient with respect to one weight matrix

At hidden layer $\ell$, the preactivation of neuron $a$ is

$$
Z_{\ell,a}
=
\sum_b (W_\ell)_{ab}A_{\ell-1,b}.
$$

Therefore,

$$
\frac{\partial Z_{\ell,a}}
{\partial (W_\ell)_{cd}}
=
\begin{cases}
A_{\ell-1,d},&a=c,\\
0,&a\neq c.
\end{cases}
$$

Applying the chain rule gives

$$
\frac{\partial F_{W,j}(x)}
{\partial (W_\ell)_{ab}}
=
\frac{\partial F_{W,j}(x)}
{\partial Z_{\ell,a}(x)}
A_{\ell-1,b}(x).
$$

Using $\delta_\ell$, this becomes

$$
\frac{\partial F_{W,j}(x^{(s)})}
{\partial (W_\ell)_{ab}}
=
\delta_\ell[s,j,a]A_{\ell-1}[s,b].
$$

Equivalently, the derivative with respect to the entire weight matrix is the outer product

$$
\nabla_{W_\ell}F_{W,j}(x^{(s)})
=
\delta_\ell[s,j,:]^T
A_{\ell-1}[s,:].
$$

This is the central backpropagation identity used by the GPU algorithm.

In the code, the tensor $\delta_\ell$ has shape

$$
M\times d_L\times d_\ell,
$$

while the layer input $A_{\ell-1}$ has shape

$$
M\times d_{\ell-1}.
$$

By inserting singleton dimensions and multiplying with broadcasting, the code obtains a tensor of shape

$$
M\times d_L\times d_\ell\times d_{\ell-1}.
$$

Its entries are exactly

$$
\frac{\partial F_{W,j}(x^{(s)})}
{\partial (W_\ell)_{ab}}.
$$

Thus, one broadcasted GPU operation computes the derivatives for every sample, every output coordinate, and every entry of $W_\ell$.

The last two tensor axes are then flattened and placed into the columns of the global Jacobian corresponding to $W_\ell$.

## Propagating to the previous layer

After computing the derivatives with respect to $W_\ell$, the code propagates $\delta_\ell$ to the preceding layer.

Since

$$
Z_\ell=W_\ell A_{\ell-1},
$$

the derivative with respect to $A_{\ell-1}$ is

$$
\frac{\partial F}{\partial A_{\ell-1}}
=
\delta_\ell W_\ell.
$$

The preceding activation satisfies

$$
A_{\ell-1}=Z_{\ell-1}^{\circ r}.
$$

Therefore,

$$
\delta_{\ell-1}
=
(\delta_\ell W_\ell)
\circ
\left(rZ_{\ell-1}^{\circ(r-1)}\right),
$$

where $\circ$ denotes entrywise multiplication.

The code repeats this process from the last hidden layer back to the first hidden layer.

## The evaluated Jacobian tensor

After processing every layer, the code has constructed a tensor

$$
\mathcal J\in\mathbb F_p^{M\times d_L\times N}
$$

whose entries are

$$
\mathcal J[s,j,\nu]
=
\frac{\partial F_{W,j}(x^{(s)})}
{\partial\theta_\nu},
$$

where $\theta_\nu$ is the $\nu$-th scalar weight parameter.

Equivalently,

$$
\mathcal J[s,j,:]
=
\operatorname{vec}\left(
\nabla_WF_{W,j}(x^{(s)})
\right).
$$

The tensor is reshaped into the matrix

$$
J_{\mathrm{eval}}
\in
\mathbb F_p^{(Md_L)\times N}.
$$

Its rows are the flattened weight gradients

$$
\operatorname{vec}\left(
\nabla_WF_{W,j}(x^{(s)})
\right)
$$

for all samples $s$ and all output coordinates $j$.

## Why coefficient reconstruction is unnecessary

Let $J_{\mathrm{coeff}}$ denote the Jacobian of the polynomial coefficients with respect to the network weights.

Let $E$ be the evaluation matrix at the chosen unisolvent points. Then the evaluated Jacobian and coefficient Jacobian are related by

$$
J_{\mathrm{eval}}
=
(E\otimes I_{d_L})J_{\mathrm{coeff}},
$$

up to a permutation of rows.

Since the points are unisolvent, $E$ is invertible. Therefore, $E\otimes I_{d_L}$ is also invertible, and

$$
\operatorname{rank}(J_{\mathrm{eval}})
=
\operatorname{rank}(J_{\mathrm{coeff}}).
$$

The original algorithm reconstructed $J_{\mathrm{coeff}}$ from sampled values using a pseudoinverse.

The GPU algorithm does not perform this reconstruction. It computes the rank of $J_{\mathrm{eval}}$ directly.

This removes the need to:

* enumerate all monomials;
* build the monomial evaluation matrix;
* compute a pseudoinverse;
* multiply the pseudoinverse by the sampled gradients.

## Exact finite-field rank computation

The matrix $J_{\mathrm{eval}}$ is computed over a finite field $\mathbb F_p$.

The code uses modular Gaussian elimination to calculate

$$
\operatorname{rank}*{\mathbb F_p}(J*{\mathrm{eval}}).
$$

At each pivot, it:

1. finds a nonzero pivot entry;
2. swaps the corresponding row into the pivot position;
3. multiplies the pivot row by the modular inverse of the pivot;
4. eliminates the pivot entry from all rows below it.

The row eliminations below a pivot are performed in parallel on the GPU.

The entire computation is repeated over two different primes. If the resulting ranks disagree, the code raises an error. Agreement helps detect unlucky weight choices or exceptional finite-field behavior.

## Ambient dimension, expected dimension, and defect

The ambient coefficient-space dimension is

$$
\operatorname{ambient_dim}
=
Md_L.
$$

The total number of scalar weights is

$$
N=\sum_{\ell=1}^L d_{\ell-1}d_\ell.
$$

There is a scaling symmetry associated with each hidden neuron. The symmetry-corrected parameter bound is

$$
N-\sum_{\ell=1}^{L-1}d_\ell.
$$

The expected dimension is therefore

$$
\operatorname{expected_dim}
=
\min{
Md_L,
N-\sum_{\ell=1}^{L-1}d_\ell
}.
$$

The computed dimension is the common rank obtained over the two finite fields.

Finally, the defect is

$$
\operatorname{defect}
=\operatorname{expected_dim}
\operatorname{dimension}.
$$

A positive defect means that the neurovariety has smaller dimension than the symmetry-corrected parameter count predicts.

## Main computational improvement

The original implementation calculated

$$
\nabla_WF_{W,j}(x^{(s)})
$$

separately for every sample $s$ and every output coordinate $j$.

The GPU implementation instead constructs the complete tensor

$$
\left(
\nabla_WF_{W,j}(x^{(s)})
\right)_{s,j}
$$

in one batched forward and backward computation.

The sample index and output index are tensor dimensions rather than Python loops.

The second improvement is that the code computes the rank of the evaluated Jacobian directly. Since evaluation at the selected points is invertible, reconstructing the coefficient Jacobian with a pseudoinverse would not change its rank and is therefore unnecessary.
