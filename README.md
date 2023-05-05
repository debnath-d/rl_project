

# RL Project

## Optimal Control for a 2-State Dynamical System

Consider the dynamical system with 2 states given by $X(t) = [x_1(t), x_2(t)]^T$ and a control input $u(t)$:

$$
\frac{dX(t)}{dt} = X(t) = f(X(t)) + g(X(t))u(t)
$$

where $f(X(t))$ and $g(X(t))$ are defined as follows:

$$
f(X(t)) = \left(\begin{array}{c} -x_1(t) + x_2(t) \\ -\frac{1}{2} x_1(t) - \frac{1}{2} x_2(t)\sin^2(x_1(t))\end{array}\right)
$$

$$
g(X(t)) = \left(\begin{array}{c} 0 \\ \sin(x_1(t)) \end{array}\right)
$$

The goal is to find the optimal control input $u(t)$ and corresponding state trajectory $X(t)$ that minimize the cost function $J(X(t))$:

$$
J(X(t)) = \frac{1}{2} \int_t^\infty (x_1^2(\tau) + x_2^2(\tau) + u^2(\tau)) d\tau
$$

with the initial conditions given by $X(0) = [1,2]^T$. Note that $f(X(t))$ is unknown.

<!-- To solve this problem, we can use the **dynamic programming** approach. We start by defining the value function $V(X(t))$, which is the minimum cost-to-go from time $t$ to infinity, given the initial state $X(t)$ and optimal control input $u(t)$:

$$
V(X(t)) = \min_{u(t)} \left\{\frac{1}{2}(x_1^2(t) + x_2^2(t) + u^2(t)) + \int_t^\infty \frac{1}{2} (x_1^2(\tau) + x_2^2(\tau) + u^2(\tau)) d\tau \right\}
$$

We can rewrite this as:

$$
V(X(t)) = \min_{u(t)} \left\{\frac{1}{2}(x_1^2(t) + x_2^2(t) + u^2(t)) + \int_t^{t+\Delta t} V(X(t+\Delta t)) d\Delta t \right\}
$$

where $\Delta t$ is a small time interval. We can approximate $V(X(t+\Delta t))$ using the following Taylor series expansion:

$$
V(X(t+\Delta t)) \approx V(X(t)) + \Delta t \frac{\partial V(X(t))}{\partial t} + O(\Delta t^2)
$$

Substituting this approximation into the value function, we get:

$$
V(X(t)) = \min_{u(t)} \left\{\frac{1}{2}(x_1^2(t) + x_2^2(t) + u^2(t)) + \int_t^{t+\Delta t} \left(V(X(t)) + \Delta t \frac{\partial V(X(t))}{\partial t}\right) d\Delta t \right\}
$$

Expanding the integral, we get: -->
