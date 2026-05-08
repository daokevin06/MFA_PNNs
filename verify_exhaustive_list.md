# Verify Completeness of List of MFAs 

Fix $d_0=2,d_L=1,r=2$.Fix $d_0=2, d_L=1, r=2$.
We verify that Table 2 of the paper is a complete list of MFAs for $d_0=2, d_L=1$, and $L\leq 7$. Given that verification for $L\leq 6$ is very different from verification for $L=7$, we split the proof into two lemmas.


Here is a description of the check: Let $\mathcal{A}$ be a finite set of MFAs we have found. By Dickson's lemma there are finitely many MFAs, so we might hope $\mathcal{A}$ contains all of them. To show $\mathcal{A}$ is a complete list, we must show there cannot exist MFAs in the uncovered regions $(\mathcal{R}_i)_{i\in I}$ (for some finite indexing set $I$). First, one has an algorithm to produce that list of $(\mathcal{R}_i)_{i\in I}$ to check. Second, one has a second algorithm which systematically bounds $\max\{D_{(d_0,\mathbf{a},d_L)}\}_{\mathbf{a}\in \mathcal{R}_i}$ using the recursive bound. If the set can be bounded strictly less than $\dim \operatorname{Sym}_{r^{L-1}}(\mathbb{R}^{d_0})^{\oplus d_L}$, then all architectures $(d_0,\mathbf{a},d_L)$ for $\mathbf{a}\in \mathcal{R}_i$ are not filling and so $\mathcal{R}_i$ cannot contain an MFA. For regions where the algorithm cannot produce such a bound, one must either (a) seek the missing MFA in $\mathcal{R}_i$ or (b) find a different proof that all $\mathbf{a}\in\mathcal{R}_i$ produce nonfilling architectures $(d_0,\mathbf{a},d_L)$.
%

**Lemma 1:**
Table 2 is complete for $L\leq 6$.  

*Proof:*
The proofs for each $L\leq 6$ are very similar, so we write it for $L=6$ and leave the rest as an exercise. Using the dimension of the ambient space and the recursive bound, we can impose conditions on the hidden layers' widths $d_i$ which must be satisfied for the architecture to be filling.

We claim that if $\mathbf{d}=(2,d_1,d_2,d_3,d_4,d_5,1)$ is filling, then $d_1\geq 3, d_2\geq 3, d_3\geq 4, d_4\geq 4$, and $d_5\geq 2$.
It is equivalent to show that if one of these inequalities fail, then $\mathbf{d}$ is not filling. The proof of these inequalities are quite similar and so we prove $d_4\geq 4$ and leave the others as an exercise. We must show that if $d_4<4$, then $\mathbf{d}$ cannot be filling. Indeed, if $d_4=3$, then $D_{\mathbf{d}}\leq D_{(2,d_1,d_2,d_3,3)}+D_{(3,d_5,1)}-3\leq 27+6-3=30$. The ambient space $\operatorname{Sym}_{2^{L-1}}(\mathbb{R}^2)$ has dimension $33$. So, $\mathbf{d}$ cannot be a filling architecture if $d_4<4$.

These inequalities imply that if $(2,\mathbf{a},1)$ is a filling architecture, then $\mathbf{a}\in\mathcal{S}$ where

$$
\mathcal{S}=[3,\infty) \times [3,\infty)\times [4,\infty)\times [4,\infty)\times [2,\infty)\subseteq \mathbb{N}^5.
$$

The minimal element in $\mathcal{S}$ with respect to the partial ordering $(\mathbb{N}^5,\leq)$ given by coordinate-wise comparison is $\mathbf{a}=(3,3,4,4,2)$. We observed by computer verification that $(2,3,3,4,4,2,1)$ is filling. Thus, it must be minimal filling and it is the only MFA for $L=6$.

---

**Lemma 2:**
Table 2 is complete for $L=7$.

*Proof:*
If we take our list of thirteen MFAs, then we can determine the regions which have not yet been checked for the presence of an MFA. There are fifteen such infinite rectangular regions:

$$
\begin{align*}
\text{Region 1:}\qquad & [3, \infty) \times [3, 3] \times [4, \infty) \times [4, 4] \times [4, \infty) \times [2, \infty)\\
\text{Region 2:}\qquad & [3, \infty) \times [3, 3] \times [4, 4] \times [4, 5] \times [4, 5] \times [2, \infty)\\
\text{Region 3:}\qquad & [3, \infty) \times [3, 3] \times [4, \infty) \times [4, 5] \times [4, 4] \times [2, \infty)\\
\text{Region 4:}\qquad & [3, \infty) \times [3, 3] \times [4, \infty) \times [4, 5] \times [4, 5] \times [2, 3]\\
\text{Region 5:}\qquad & [3, \infty) \times [3, 3] \times [4, 5] \times [4, 6] \times [4, 4] \times [2, 3]\\
\text{Region 6:}\qquad & [3, \infty) \times [3, 3] \times [4, \infty) \times [4, 6] \times [4, 4] \times [2, 2]\\
\text{Region 7:}\qquad & [3, \infty) \times [3, 3] \times [4, \infty) \times [4, 5] \times [4, \infty) \times [2, 2]\\
\text{Region 8:}\qquad & [3, \infty) \times [3, \infty) \times [4, 4] \times [4, 4] \times [4, \infty) \times [2, \infty)\\
\text{Region 9:}\qquad & [3, \infty) \times [3, \infty) \times [4, 4] \times [4, 5] \times [4, 5] \times [2, 3]\\
\text{Region 10:}\qquad & [3, \infty) \times [3, \infty) \times [4, 4] \times [4, \infty) \times [4, 4] \times [2, \infty)\\
\text{Region 11:}\qquad & [3, \infty) \times [3, \infty) \times [4, 4] \times [4, 5] \times [4, \infty) \times [2, 2]\\
\text{Region 12:}\qquad & [3, \infty) \times [3, \infty) \times [4, \infty) \times [4, 4] \times [4, 5] \times [2, \infty)\\
\text{Region 13:}\qquad & [3, \infty) \times [3, 4] \times [4, 5] \times [4, 5] \times [4, 4] \times [2, 3]\\
\text{Region 14:}\qquad & [3, \infty) \times [3, \infty) \times [4, \infty) \times [4, 5] \times [4, 4] \times [2, 2]\\
\text{Region 15:}\qquad & [3, \infty) \times [3, \infty) \times [4, \infty) \times [4, 4] \times [4, \infty) \times [2, 3]
\end{align*}
$$

Our goal is to show that architectures $(2,\mathbf{a},1)$ with $\mathbf{a}$ sitting in one of these regions cannot be filling. We use the recursive bound and the expected dimension to place upper bounds on the dimension of the neurovarieties. This can be done systematically by a computer. The results are displayed below.

$$
\begin{align*}
\text{Region 1:} & \quad D_{(2,d_{1},3,d_{3},4,d_{5},d_{6},1)} && \leq D_{(2,d_{1},3)} + D_{(3,d_{3},4)} + D_{(4,d_{5},d_{6},1)} - 7 \\
& && \leq 9 + 24 + 35 - 7 \\
& && = 61 < 65 \\
\text{Region 2:} & \quad D_{(2,d_{1},3,4,5,5,d_{6},1)} && \leq D_{(2,d_{1},3)} + D_{(3,4)} + D_{(4,5)} + D_{(5,5)} + D_{(5,d_{6},1)} - 17 \\
& && \leq 9 + 12 + 20 + 25 + 15 - 17 \\
& && = 64 < 65 \\
\text{Region 3:} & \quad D_{(2,d_{1},3,d_{3},5,4,d_{6},1)} && \leq D_{(2,d_{1},3)} + D_{(3,d_{3},5)} + D_{(5,4)} + D_{(4,d_{6},1)} - 12 \\
& && \leq 9 + 30 + 20 + 10 - 12 \\
& && = 57 < 65 \\
\text{Region 4:} & \quad \text{Unproven} && \text{(Lowest bound: } 66 \geq 65\text{)} \\
\text{Region 5:} & \quad \text{Unproven} && \text{(Lowest bound: } 66 \geq 65\text{)} \\
\text{Region 6:} & \quad D_{(2,d_{1},3,d_{3},6,4,2,1)} && \leq D_{(2,d_{1},3)} + D_{(3,d_{3},6)} + D_{(6,4)} + D_{(4,2)} + D_{(2,1)} - 15 \\
& && \leq 9 + 36 + 24 + 8 + 2 - 15 \\
& && = 64 < 65 \\
\text{Region 7:} & \quad D_{(2,d_{1},3,d_{3},5,d_{5},2,1)} && \leq D_{(2,d_{1},3)} + D_{(3,d_{3},5)} + D_{(5,d_{5},2)} + D_{(2,1)} - 10 \\
& && \leq 9 + 30 + 30 + 2 - 10 \\
& && = 61 < 65 \\
\text{Region 8:} & \quad D_{(2,d_{1},d_{2},4,4,d_{5},d_{6},1)} && \leq D_{(2,d_{1},d_{2},4)} + D_{(4,4)} + D_{(4,d_{5},d_{6},1)} - 8 \\
& && \leq 20 + 16 + 35 - 8 \\
& && = 63 < 65 \\
\text{Region 9:} & \quad \text{Unproven} && \text{(Lowest bound: } 66 \geq 65\text{)} \\
\text{Region 10:} & \quad D_{(2,d_{1},d_{2},4,d_{4},4,d_{6},1)} && \leq D_{(2,d_{1},d_{2},4)} + D_{(4,d_{4},4)} + D_{(4,d_{6},1)} - 8 \\
& && \leq 20 + 40 + 10 - 8 \\
& && = 62 < 65 \\
\text{Region 11:} & \quad D_{(2,d_{1},d_{2},4,5,d_{5},2,1)} && \leq D_{(2,d_{1},d_{2},4)} + D_{(4,5)} + D_{(5,d_{5},2)} + D_{(2,1)} - 11 \\
& && \leq 20 + 20 + 30 + 2 - 11 \\
& && = 61 < 65 \\
\text{Region 12:} & \quad D_{(2,d_{1},d_{2},d_{3},4,5,d_{6},1)} && \leq D_{(2,d_{1},d_{2},d_{3},4)} + D_{(4,5)} + D_{(5,d_{6},1)} - 9 \\
& && \leq 36 + 20 + 15 - 9 \\
& && = 62 < 65 \\
\text{Region 13:} & \quad \text{Unproven} && \text{(Lowest bound: } 66 \geq 65\text{)} \\
\text{Region 14:} & \quad D_{(2,d_{1},d_{2},d_{3},5,4,2,1)} && \leq D_{(2,d_{1},d_{2},d_{3},5)} + D_{(5,4)} + D_{(4,2)} + D_{(2,1)} - 11 \\
& && \leq 45 + 20 + 8 + 2 - 11 \\
& && = 64 < 65 \\
\text{Region 15:} & \quad D_{(2,d_{1},d_{2},d_{3},4,d_{5},3,1)} && \leq D_{(2,d_{1},d_{2},d_{3},4)} + D_{(4,d_{5},3)} + D_{(3,1)} - 7 \\
& && \leq 36 + 30 + 3 - 7 \\
& && = 62 < 65 
\end{align*}
$$

For Regions $4, 5, 9, 13$, better estimates on the dimensions are needed. The need for better estimates arise from influence of defective architectures e.g. $(5,5,3,1)$ is defective with computed dimension $32$ while $\operatorname{expdim} \mathcal{V}_{(5,5,3,1),2}=35$.

For Regions 4,5,9,13, we respectively have:

$$
\begin{align*}
D_{(2,d_1,3,d_3,5,5,3,1)} & \leq D_{(2,d_1,3)}+D_{(3,d_3,5)}+D_{(5,5,3,1)}-3-5\leq 9+30+D_{(5,5,3,1)}-3-5\\
D_{(2,d_1,3,5,6,4,3,1)} & \leq D_{(2,d_1,3)}+D_{(3,5,6,9,3,1)}-3\leq 9+58-3=64\\
D_{(2,d_1,d_2,4,5,5,3,1)} & \leq D_{(2,d_1,d_2,4)}+D_{(4,5,5,3,1)}-4\leq 20+47-4=63\\
D_{(2,d_1,4,5,5,4,3,1)} & \leq D_{(2,d_1,4,5)}+D_{(5,5)}+D_{(5,4,3,1)}-10\leq 24+25+25-10=64.
\end{align*}
$$

Here, we used finite field computations to verify that $D_{(5,5,3,1)}=32$, $D_{(3,5,6,9,3,1)}=58$, $D_{(4,5,5,3,1)}=47$, and $D_{(5,4,3,1)}=25$. We also used the statement $D_{(2,d_1,4,5)}\leq 24$ for $d_1\geq 3$ which follows from the main results of [Massarenti. et. al. 2025](https://arxiv.org/abs/2511.19703).