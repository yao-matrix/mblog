# Winograd Convolution 推导 - 从 1D 到 2D

## 1D Winograd 卷积
1D Winograd 算法已经有很多文章讨论了，讨论得都比较清楚，这里就不再赘述，仅列出结论。

> **输入**：四维信号 $\vec{d} = [d_{0}, d_{1}, d_{2}, d_{3}]^{T}$
> 
> **卷积核**： 三维向量 $\vec{k} = [k_{0}, k_{1}, k_{2}]^{T}$
> 
> **输出**： 二维信号 $\vec{r} = [r_0, r_1]^T$

则 $\vec{r}$ 可表示为：

$\vec{r} = A^T[(G\vec{k}) \odot (B^T\vec{d})]$

其中：
$$ G = \left[
 \begin{matrix}
   1 & 0 & 0 \\
   \frac{1}{2} & \frac{1}{2} & \frac{1}{2} \\
   \frac{1}{2} & -\frac{1}{2} & \frac{1}{2} \\
   0 & 0 & 1
  \end{matrix}
  \right]_{4\times3}$$
$$B^{T} = \left[
 \begin{matrix}
   1 & 0 & -1 & 0 \\
   0 & 1 & 1 & 0 \\
   0 & -1 & 1 & 0 \\
   0 & 1 & 0 & -1
  \end{matrix}
  \right]_{4\times4}$$
$$A^{T}= \left[
 \begin{matrix}
   1 & 1 & 1 & 0 \\
   0 & 1 & -1 & -1
  \end{matrix}
  \right]_{2\times4}$$

## 2D Winograd 卷积
2D Winograd 可以由 1D Winograd 外推得到，因此为解决 2D Winograd 问题，首先要**重温 1D 卷积解决的问题**。在此复述一遍：
假设一个卷积核尺寸为 `3` 的一维卷积，假设每次我们输出 `2` 个卷积点，则我们形式化此问题：`F(2, 3)`。

因为输出为 `2`，卷积核大小为 `3`，对应的输入点数应该为 `4`，则此问题表述为：

> **输入**：四维信号 $\vec{d} = [d_{0}, d_{1}, d_{2}, d_{3}]^{T}$
> 
> **卷积核**： 三维向量 $\vec{k} = [k_{0}, k_{1}, k_{2}]^{T}$

因此，此卷积的**矩阵乘形式**应为：
$$\left[
 \begin{matrix}
   d_{0} & d_{1} & d_{2} \\
   d_{1} & d_{2} & d_{3}
  \end{matrix}
  \right] \left[
 \begin{matrix}
   k_{0} \\
   k_{1} \\
   k_{2}
  \end{matrix}
  \right] = \left[
 \begin{matrix}
   r_{0} \\
   r_{1}
  \end{matrix}
  \right] = D\vec{k}$$
  
请记住这个形式是 Winograd 算法解决的问题，后续 2D 算法将化归为这个问题。

下面我们来定义 2D 卷积问题，将 1D 卷积扩展一维：
假设一个卷积核尺寸为 `3x3` 的二维卷积，假设每次我们输出 `2x2` 个卷积点，则我们形式化此问题：`F(2x2, 3x3)`。

因为输出为 `2x2`，卷积核大小为 `3x3`，对应的输入点数应该为 `4x4`，则此问题表述为：

> **输入**：
> $$ D = \left[
 \begin{matrix}
   d_{00} & d_{01} & d_{02} & d_{03} \\
   d_{10} & d_{11} & d_{12} & d_{13} \\
   d_{20} & d_{21} & d_{22} & d_{23} \\
   d_{30} & d_{31} & d_{32} & d_{33}
  \end{matrix}
  \right] $$
> **卷积核**：
> $$ K = \left[
 \begin{matrix}
   k_{00} & k_{01} & k_{02} \\
   k_{10} & k_{11} & k_{12} \\
   k_{20} & k_{21} & k_{22} 
  \end{matrix}
  \right] $$

因此，此卷积的**矩阵乘形式**应为：

$$\left[
 \begin{matrix}
   d_{00} & d_{01} & d_{02} & d_{10} & d_{11} & d_{12} & d_{20} & d_{21} & d_{22} \\
   d_{01} & d_{02} & d_{03} & d_{11} & d_{12} & d_{13} & d_{21} & d_{22} & d_{23} \\
   d_{10} & d_{11} & d_{12} & d_{20} & d_{21} & d_{22} & d_{30} & d_{31} & d_{32} \\
   d_{11} & d_{12} & d_{13} & d_{21} & d_{22} & d_{23} & d_{31} & d_{32} & d_{33} \\
  \end{matrix}
  \right] \left[
 \begin{matrix}
   k_{00} \\
   k_{01} \\
   k_{02} \\
   k_{10} \\
   k_{11} \\
   k_{12} \\
   k_{20} \\
   k_{21} \\
   k_{22} \\
  \end{matrix}
  \right] = \left[
 \begin{matrix}
   r_{00} \\
   r_{01} \\
   r_{10} \\
   r_{11} \\
  \end{matrix}
  \right] $$
  
从这个式子里，我们可以看到 1D 卷积的影子，这个影子在我们对矩阵作了分块后会更加明显。

$$\left[
 \begin{array}{ccc|ccc|ccc} 
   d_{00} & d_{01} & d_{02} & d_{10} & d_{11} & d_{12} & d_{20} & d_{21} & d_{22} \\
   d_{01} & d_{02} & d_{03} & d_{11} & d_{12} & d_{13} & d_{21} & d_{22} & d_{23} \\
   \hline
   d_{10} & d_{11} & d_{12} & d_{20} & d_{21} & d_{22} & d_{30} & d_{31} & d_{32} \\
   d_{11} & d_{12} & d_{13} & d_{21} & d_{22} & d_{23} & d_{31} & d_{32} & d_{33} \\
  \end{array}
  \right] \left[
 \begin{matrix}
   k_{00} \\
   k_{01} \\
   k_{02} \\
   \hline
   k_{10} \\
   k_{11} \\
   k_{12} \\
   \hline
   k_{20} \\
   k_{21} \\
   k_{22} \\
  \end{matrix}
  \right] = \left[
 \begin{matrix}
   r_{00} \\
   r_{01} \\
   \hline
   r_{10} \\
   r_{11} \\
  \end{matrix}
  \right] $$

再明显一点，我们写成分块矩阵乘的形式：

$$
   \left[
 \begin{matrix}
   D_{00} & D_{10}  & D_{20}\\
   D_{10} & D_{20} & D_{30}
  \end{matrix}
  \right]  \left[
 \begin{matrix}
   \vec{k_{0}} \\
   \vec{k_{1}} \\
   \vec{k_{2}}
  \end{matrix}
  \right] = \left[
 \begin{matrix}
   \vec{r_{0}} \\
   \vec{r_{1}} 
  \end{matrix}
  \right]
$$

至此，我们对 2D 卷积推导出了跟 1D 形式一致的公式，只不过 1D 中的标量在 2D 中变成了小矩阵或者向量。

### 实操粉

对实操粉而言，到这个形式为止，已经可以写代码了。

由 1D Winograd 可知，我们可以将该式改写为 Winograd 形式, 如下：

$$
   \left[
 \begin{matrix}
   D_{00} & D_{10}  & D_{20}\\
   D_{10} & D_{20} & D_{30}
  \end{matrix}
  \right]  \left[
 \begin{matrix}
   \vec{k_{0}} \\
   \vec{k_{1}} \\
   \vec{k_{2}}
  \end{matrix}
  \right] = \left[
 \begin{matrix}
   \vec{r_{0}} \\
   \vec{r_{1}} 
  \end{matrix}
  \right] =  \left[
 \begin{matrix}
   M_{0} + M_{1} + M_{2}  \\
   M_{1} - M_{2} - M_{3}
  \end{matrix}
  \right]
$$

其中：
$$
   \begin{align*}
   & M_{0} = (D_{00} - D_{20})\vec{k_{0}} \\
   & M_{1} = (D_{10} + D_{20})\frac{\vec{k_{0}} + \vec{k_{1}} + \vec{k_{2}}}{2} \\
   & M_{2} =  (D_{20} - D_{10})\frac{\vec{k_{0}} - \vec{k_{1}} + \vec{k_{2}}}{2} \\
   & M_{3} = (D_{10} - D_{30})\vec{k_{2}}
   \end{align*}
$$

注意，这四个 `M` 的计算又可以用一维的 `F(2, 3)` Winograd 来做，因此 2D Winograd 是个**嵌套（nested）**的算法。  
 
### 理论粉
对一个有追求的理论粉来说，只是得到可以写程序的递归表达肯定是不完美的，他们还是希望有一个最终的解析表达的。其实也很简单，我们把上面的式子规整规整，使得输出成为一个标准的 `2x2` 矩阵，有：
$$
 \left[
 \begin{matrix}
   \vec{r_{0}} , \vec{r_{1}} 
  \end{matrix}
  \right] =  \left[
 \begin{matrix}
   M_{0} + M_{1} + M_{2}, M_{1} - M_{2} - M_{3}
  \end{matrix}
  \right]
$$

可以写为：
$$
   \left[
 \begin{matrix}
   \vec{r_{0}} , \vec{r_{1}} 
  \end{matrix}
  \right] =  \left[
 \begin{matrix}
   M_{0}, M_{1}, M_{2}, M_{3}
  \end{matrix}
  \right] \left[
 \begin{matrix}
   1 & 0 \\
   1 & 1 \\
   1 & -1 \\
   0 & -1
  \end{matrix}
  \right] 
$$

依 1D Winograd 公式 $\vec{r} = A^T[(G\vec{k}) \odot (B^T\vec{d})]$，并结合各 `M` 的公式，有下式。

$$\begin{align*}
     \left[
 \begin{matrix}
   \vec{r_{0}} , \vec{r_{1}} 
  \end{matrix}
  \right] &=  \left[
 \begin{matrix}
   M_{0}, M_{1}, M_{2}, M_{3}
  \end{matrix}
  \right] A \\
  &= \left[
 \begin{matrix}A^T[(G\vec{k_0}) \odot (B^T(\vec{d_0} - \vec{d_2}))],  A^T[(G\frac{\vec{k_0} + \vec{k_1} + \vec{k_2}}{2}) \odot (B^T(\vec{d_1} + \vec{d_2}))],  A^T[(G\frac{\vec{k_0} - \vec{k_1} + \vec{k_2}}{2}) \odot (B^T(\vec{d_2} - \vec{d_1}))],  A^T[(G\vec{k_2}) \odot (B^T(\vec{d_1} - \vec{d_3}))]   \end{matrix} 
  \right]A \\
  &=A^T\left[
 \begin{matrix}(G\vec{k_0}) \odot (B^T(\vec{d_0} - \vec{d_2})),  (G\frac{\vec{k_0} + \vec{k_1} + \vec{k_2}}{2}) \odot (B^T(\vec{d_1} + \vec{d_2})),  (G\frac{\vec{k_0} - \vec{k_1} + \vec{k_2}}{2}) \odot (B^T(\vec{d_2} - \vec{d_1})),  (G\vec{k_2}) \odot (B^T(\vec{d_1} - \vec{d_3}))  \end{matrix}
  \right]A 
  \end{align*}
$$

注意到像 $(G\vec{k_0})$ 这些都是 2 维列向量，hadamard product 和 concat 可以交换而不影响结果，因此：

$$\begin{align*}
     \left[
 \begin{matrix}
   \vec{r_{0}} , \vec{r_{1}} 
  \end{matrix}
  \right]
  &=A^T\left[
 \begin{matrix}(G\vec{k_0}) \odot (B^T(\vec{d_0} - \vec{d_2})), (G\frac{\vec{k_0} + \vec{k_1} + \vec{k_2}}{2}) \odot (B^T(\vec{d_1} + \vec{d_2})), (G\frac{\vec{k_0} - \vec{k_1} + \vec{k_2}}{2}) \odot (B^T(\vec{d_2} - \vec{d_1})),  (G\vec{k_2}) \odot (B^T(\vec{d_1} - \vec{d_3}))  \end{matrix}
  \right]A\\
  &=A^T\left[
 \begin{matrix}(G[ \vec{k_0}, \frac{\vec{k_0} + \vec{k_1} + \vec{k_2}}{2}, \frac{\vec{k_0} - \vec{k_1} + \vec{k_2}}{2}, \vec{k_2}]) \odot (B^T[\vec{d_0} - \vec{d_2}, \vec{d_1} + \vec{d_2}, \vec{d_2} - \vec{d_1}, \vec{d_1} - \vec{d_3}]) \end{matrix}
  \right]A \\
    &=A^T\left[
 \begin{matrix}(G[ \vec{k_0}, \vec{k_1}, \vec{k_2}]  \left[
 \begin{matrix}
   1 & \frac{1}{2} & \frac{1}{2} & 0 \\
   0 & \frac{1}{2} & -\frac{1}{2} & 0 \\
   0 & \frac{1}{2} & \frac{1}{2} & 1
  \end{matrix}
  \right]) \odot (B^T[\vec{d_0}, \vec{d_1}, \vec{d_2}, \vec{d_3}]\left[
 \begin{matrix}
   1 & 0 & 0 & 0 \\
   0 & 1 & -1 & 1 \\
   -1 & 1 & 1 & 0 \\
   0 & 0 & 0 & -1
  \end{matrix}
  \right]) \end{matrix}
  \right]A \\
  &=A^T\left[
 \begin{matrix}(G[ \vec{k_0}, \vec{k_1}, \vec{k_2}]G^T) \odot (B^T[\vec{d_0}, \vec{d_1}, \vec{d_2}, \vec{d_3}]B) \end{matrix}
  \right]A\\
    &=A^T\left[
 \begin{matrix}(GKG^T) \odot (B^TDB) \end{matrix}
  \right]A
  \end{align*}
$$

至此证得。

## 参考文献
1. [Fast Algorithms for Convolutional Neural Networks](https://arxiv.org/pdf/1509.09308.pdf)
2. [Fast Algorithms for Signal Processing](http://213.55.83.214:8181/computer%20science%2055/%5BBlahut_R.E.%5D_Fast_Algorithms_for_Signal_Processin(BookZZ.org).pdf)
3. [Going beyond Full Utilization: The Inside Scoop on Nervana's Winograd Kernels](https://www.intel.ai/winograd-2/#gs.8skmj4)
4. [卷积神经网络中的Winograd快速卷积算法](https://www.cnblogs.com/shine-lee/p/10906535.html) *注：本文关于 2D Winograd 的公式推导是错误的。*

*写于 2019 年 10 月*