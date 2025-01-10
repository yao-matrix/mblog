# 用 L2 距离做 MIP，MCS 排序

## 问题
### MIP（Maximum Inner Product）
- **输入**
	- 查询向量（query）：$x \in \mathbb{R}^{d}$
	- 底库（database）：$Y=\{y_1, y_2, ...,y_{i}\}$，其中 $y_{i} \in \mathbb{R}^{d}$
- **输出**
	- 底库中与查询向量**点积**相似度最大的 $k$ 个向量：
	  $$x_k := \underset{i \in [1, n]}{arg \ maxk} <x, y_{i}>，k \in [1, K]$$

### MCS（Maximum Cosine Similarity）
- **输入**
	- 查询向量（query）：$x \in \mathbb{R}^{d}$
	- 底库（database）：$Y=\{y_1, y_2, ...,y_{i}\}$，其中 $y_{i} \in \mathbb{R}^{d}$
- **输出**
	- 底库中与查询向量**点积**相似度最大的 $k$ 个向量：
	  $$x_k := \underset{i \in [1, n]}{arg \ maxk} \frac {<x, y_{i}>}{||x||*||y_i||}，k \in [1, K]$$

## 转换

### MIP $\to$ L2
通过保序变换（Ordering Preserving Transformation）：

设 $\phi \overset{\triangle}{=}\underset{i}{max\ } ||y_i||$，对每个查询向量 $x$ 和库向量 $y_i$ 分别作如下变换：

$$\begin{align*} &\tilde{y}_{i} = (y_i^T, \sqrt{\phi^2-||y_i||^2})^T \\
&\tilde{x} = (x^T, 0)^T \end{align*}$$

则，新的 $d+1$ 维向量 $\tilde{x}$ 和 $\tilde{y_i}$ 的 L2 距离与 $x$，$y_i$ 的 IP 距离有如下关系：

$$\begin{align*}d^2 &= ||\tilde{x} - \tilde{y}_i||^2 \\
&= ||\tilde{x}||^2 + ||\tilde{y}_i||^2-2\tilde{x}*\tilde{y}_i \\      
&= ||x||^2 + (||y_i||^2 + \phi^2 - ||y_i||^2) - 2x*y_i \\
&= ||x||^2 + \phi^2 - 2x*y_i\end{align*}$$

$||x||^2$ 和 $\phi$ 都与 $i$ 无关，因此：

$$j = \underset{i}{argmin\ }\sqrt{||\tilde{x} - \tilde{y}_i||^2} = \underset{i}{argmax\ }x*y_i$$

即 $d+1$ 维向量 $\tilde{x}$、$\tilde{y_i}$ 的 L2 距离的升序排序与 $x$、$y_i$ 的 IP 距离的降序排列是一致的。

### MCS $\to$ L2
Cosine 相似性是归一化后的 IP 距离：

$$\frac{x*y_i}{||x||* ||y_i||} \propto  x * \frac{y_i}{||y_i||}$$ 

所以，可以先对 $y_i$ 做一个归一化，变成 $y'_i = \frac{y_i}{||y_i||}$。这样就把这个问题转换成了 MIP，可以用上面的 `MIP->L2` 的变换。特殊的是：此时 $\phi = 1$。因此，只需要做一个很简单的变换：

$$\begin{align*} &\tilde{y}_{i} = \frac{y_i}{||y_i||}  \end{align*}$$

则，

$$\begin{align*}d^2 &= ||x - \tilde{y}_i||^2 \\                  
&= ||x||^2 + 1 - 2\frac{x*y_i}{||y_i||} \end{align*}$$

即：

$$j = \underset{i}{argmin\ }\sqrt{||x - \tilde{y}_i||^2} = \underset{i}{argmax\ }\frac{x*y_i}{||y_i||}$$

从上式可得，$x$、$\tilde{y_i}$ 的 L2 距离的升序排列与 $x$、$y_i$ 的 cosine 相似性的降序排列是一致的。

## 实操适用

IVF Based Indexing 使用方式：
- 索引阶段不使用变换，召回阶段使用变换
  - **支持**，索引阶段还是使用 IP 或者 cosine 相似性构建索引，召回阶段使用相应变换后，使用 L2 距离召回。注意：在 MIP 中，第一阶段和第二阶段的 $y$ 需要独立计算 $\phi$。
- 索引阶段、召回阶段都使用变换
	- MIP: **支持，但需要修改训练过程**。需要注意：在**索引阶段**，质心是 $y$，因此每一轮迭代算出新的质心后，需要先计算把所有质心按照上文重新完整做一遍 $d$ 维到 $d+1$ 的变换。
	- MCS: **支持，但需要修改索引过程**。需要注意：在**索引阶段**，质心是 $y$，因此每一轮迭代算出新的质心后，需要先计算把所有质心重新做一遍归一化。

## 参考文献
1. [Speeding Up the Xbox Recommender System Using a Euclidean Transformation for Inner-Product Spaces](http://ulrichpaquet.com/Papers/SpeedUp.pdf)

*写于 2020 年 9 月*