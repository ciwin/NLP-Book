## Artificial Neuron

Activation function of a neuron $j$ receiving input from neurons $i$:
$$
o_j = f(\sum_i w_{ji}o_i+b_j)
$$

## Gradient Descent

Learning in a multilayer perceptron is optimizing the parameters (*weights*) in a way to minimize an *error function*  (also called *objective function, loss function* or *cost function*).

As usually the optimal values of the weights cannot be calculated directly, an iterative optimization approach is used. If $f(\textbf x)$ is the error function and we want to find the optimal value for $\textbf x$ so that $f(\textbf x)$ is minimal, we can use the derivative $f'(\textbf x)$ which gives us the slope at point $\textbf x$. If the slope $f'(\textbf x) > 0$, decreasing $\textbf x$ will decrease $f(\textbf x)$. If the slope $f'(\textbf x) < 0$, increasing $\textbf x$ will decrease $f(\textbf x)$. By iteratively calculating new values for $\textbf x$ with:
$$
\textbf x^{new} = \textbf x - \epsilon f'(\textbf x)
$$
we can find at least a local minimum for $f(\textbf x)$ if $\epsilon$ is small enough. $\epsilon$ is called the *learning rate* and is a positive small number (usually $\epsilon << 1$). 

As $\textbf x$ is an $n$-dimensional vector, the derivative is also a vector called the *gradient* $\nabla_{\textbf x} f(\textbf x)$. Element $i$ of the gradient is the partial derivative of $f$ with respect to $x_i$. The iterative process of formula (2) is written:
$$
\textbf x^{new} = \textbf x - \epsilon \nabla_{\textbf x} f(\textbf x)
$$
This iterative technique is called *gradient descent* and is generally attributed to *Augustin-Louis Cauchy*, who first suggested it in 1847. 





