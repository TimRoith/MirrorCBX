
![image](https://github.com/user-attachments/assets/f5d9573f-5acc-458c-9b92-8d0470fb2ef4)

---------
# MirrorCBX

The mirror consensus-based optimization method MirrorCBO aims to compute the global minimizer of a cost function $J:\mathbb R^d\to\mathbb R$ by solving the following system of stochastic differential equation 
```math
\mathrm d y_t^{i} = -\left(\nabla\phi^\ast(y_t^{(i)})- m_\alpha^\ast[\mu_t^N]\right)\mathrm d t + \sigma |\nabla\phi^\ast(y_t^{i}) - m_\alpha^\ast[\mu_t^N]|\mathrm d W_t^{(i)},\quad i=1,\dots,N
```

where $\phi:\mathbb{R}^d\to[0,\infty]$ is the so-called distance generating function and 

```math
m_\alpha^\ast[\mu_t^N]:=\frac{\sum_{i=1}^N\exp\left(-\alpha J(\nabla\phi^\ast(y_t^{(i)}))\right)\nabla\phi^*(y_t^{(i)})}{\sum_{i=1}^N\exp\left(-\alpha J(\nabla\phi^\ast(y_t^{(i)}))\right)}
```
is the weighted CBO-mean of the primal particles $x_t^{(i)}:=\nabla\phi^\ast(y_t^{(i)})$. For $\phi(x):=\frac12|x|^2$ MirrorCBO reduces to standard CBO, for the choice $\phi(x):=\frac12|x|^2+\iota_C(x)$ one can perform constrained optimization on the set $C$, and for $\phi(x):=\frac12|x|^2+\lambda|x|_1$ the trajectories of the primal particles become sparse.


