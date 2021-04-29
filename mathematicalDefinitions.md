*Christoph Windheuser, April 29, 2021*

# Some mathematical definitions

## Linear Algebra

### Norms

*(Goodfellow et.al. Deep Learning Book, p. 39)*

*Norms* calculate a scalar value from a vector, which is equivalent to the *size* or *length* of the vector.

A norm is any function $f(\textbf x)$ of a vector $\textbf x$ with the following properties:

* $f(\textbf x) = 0 \Rightarrow \textbf x = 0$
* $f(\textbf x + \textbf y) \leq f(\textbf x) + f(\textbf y)$
* $\forall c \in \R: f(c \, \textbf x) = |c|f(\textbf x)$

$L^p$ for $p \in \N$ and $p \geq 1$  are a class of very popular norms:
$$
||\textbf x ||_P = (\sum_i |\textbf x_i|^p)^\frac{1}{p}
$$

#### $L^1$ Norm

The $L^1$ norm is the absolute value of a vector $\textbf x$:
$$
||\textbf x ||_1 = \sum_i |\textbf x_i|
$$

#### $L^2$ Norm

The $L^2$ norm is know as the *Euclidean norm*. It is the Euclidean distance from the origin to the point identified by the vector $\textbf x$. 
$$
||\textbf x ||_2 = \sqrt{\sum_{i} {\textbf x_{i}^{\mathstrut 2}}}
$$
As the $L^2$ is very common, the subscrpt 2 is sometimes omitted and the norm is written as $||\textbf x||$. 

#### Squared $L^2$ Norm

The squared $L^2$ norm is the $L^2$ norm without the square root:
$$
||\textbf x||^2_2 = \sum_i \textbf x_i^2
$$
The squared $L^2$ norm has a lot of mathematical advantages and is often used in machine learning. It can simply be calculated as $\textbf x^\intercal \textbf x$ and the derivative is much simpler and convenient as in case of the $L^2$ norm. 

## Probability and Information Theory

*(Goodfellow et.al. Deep Learning Book, p. 53 ff.)*

### Random Variables

*(Goodfellow et.al. Deep Learning Book, p. 56)*

A *random variable* describes the outcome of a random experiment. It is a variable $x$, that can take different values randomly. Random variables can be *discrete* or *continuous*. Discrete random variables have an finite or countably infinite number of states as the outcome of a random experiment.

Example: Throwing a dice is a discrete random variable with 6 different states. 

Random variables can also be vectors $\textbf x$ consisting of several random variables $x_i$.

Random variables only describes the possible outcomes of a random experiment. Usually random variables are coupled with a *probability distribution* that describes the probability of each outcome of the random variable.

### Probability Distributions

*(Goodfellow et.al. Deep Learning Book, p. 56)*

A *probability distribution* $P(x)$ of a random variable $x$ is a description of the probability that a random variable takes each of its possible states (discrete random variable) or outcome values (continuous random variable). 

A probability distribution is a function that maps a random variable $x$ to a real number $P(x)$ which describes the probability of the event $x$. $P(x)$ must satisfy the following properties:

* For all states $x$ $P(x)$ must be greater or equal to 0 and smaller or equal to 1. If $P(x) = 0$ the state $x$ is *impossible*. If $P(x) = 1$, the state $x$ is a *sure event*, guaranteed to happen.
* The sum of $P(x)$ over all states must be 1: $\sum_{x \in x} P(x) = 1$. 

A probability distribution is describing an experiment generating random event. If the probability distribution of an experiment is accurate, the probability distribution can be used to generate random events that cannot be distinguished from the random events generated by the experiment.

For example, if the experiment throwing a dice is described by the following probability distribution:
$$
P(1) = P(2) = P(3) = P(4) = P(5) = P(6) = \frac{1}{6}
$$
Then this probability distribution can be used to generate the events $1$ to $6$. If these events generated by the probability distribution and the events generated by throwing a real dice are sent to an observer by email, is will not be able to distinguish if an event was generated by throwing a dice or by generating an event by the probability distribution.

### Expectation

*(Goodfellow et.al. Deep Learning Book, p. 60)*

If there is a discrete probability distribution $P(x)$ and a function $f(x)$, then the *expectation, expected value* or *mean value* of the function $f(x)$ with respect to $P(x)$ (this means that $x$ in $f(x)$ is generated by $P(x)$) is the *mean value* of $f(x)$:
$$
E_{x \sim P}[f(x)] = \sum _x P(x) f(x)
$$

### Self-Information

*(Goodfellow et.al. Deep Learning Book, p. 72)*

*Self-information* $I(x)$ specifies the information a discrete random event $x$ generated by $P(x)$ has. A sure event should have zero information, a seldom event should have high information. 
$$
I(x) = - \log P(x)
$$
We always use the *natural logarithm* for $\log$ here.

### Shanon Entropy

*(Goodfellow et.al. Deep Learning Book, p. 74)*

The *Shanon Entropy* $H(P)$ of an discrete random distribution $P(x)$ is the expectation of the self-information of all events $x$ generated by $P(x)$:
$$
H(P) = E_{x \sim P}[I(x)] = -E_{x \sim P}[\log P(x)]
$$

### Kullback-Leibler (KL) Divergence

*(Goodfellow et.al. Deep Learning Book, p. 74)*

The *Kullback-Leibler (KL) Divergence* or *relative entropy* measures the difference of two separate random distributions $P(x)$ and $Q(x)$ over the same random variable $x$:
$$
D_{KL}(P||Q) = E_{x\sim P}\left[\log \frac{P(x)}{Q(x)}\right]= E_{x\sim P}[\log P(x) - \log Q(x)]
$$
The KL-Divergence is always *non-negative* and $0$ if and only if the two discrete distributions $P$ and $Q$ are the same. 

### Cross Entropy

*(Goodfellow et.al. Deep Learning Book, p. 75)*

The *Cross Entropy* $H(P,Q)$ of two discrete random distributions $P$ and $Q$ can directly be derived from the KL-Divergence and the Shanon Entropy (for discrete probability distributions):
$$
\begin{equation}
\begin{split}
H(P,Q) & = H(P) + D_{KL}(P||Q) \\
       & = -E_{x \sim P}[\log Q(x)] \\
       & = - \sum_{i = 1}^{m}P(x_i) \log Q(x_i)
\end{split}
\end{equation}
$$
Minimizing the cross entropy $H(P,Q)$ with respect to $Q$ is equivalent to minimizing the KL-divergence $D_{KL}(P||Q)$, because $Q$ does not participate in the term $H(P)$:
$$
\min_Q H(P,Q) = \min_Q D_{KL}(P||Q)
$$

### Maximum Likelihood Estimation

*(Goodfellow et.al. Deep Learning Book, p. 131)*

*Maximum Likelihood Estimation* is the idea to estimate the optimal parameter set for a parameterized probability distribution or *model* in a way that this probability distribution is equal to an real but unknown probability distribution that is generating patterns that we want to recognize.

We assume that a set of real examples we know $X = \{ x^{(1)}, ...,x^{(m)} \}$ is generated by an unknown probability distribution $p_{data}(\textbf x)$. 

We want to train a parameterized probability distribution (our model) $p_{model}(\textbf x; \theta)$ that it mimics the real but unknown probability distribution $p_{data}(\textbf x)$ as close as possible. That means that $p_{model}(\textbf x; \theta)$ maps any input example $x$ to a real number estimating the true probability $p_{data}(\textbf x)$. To find the optimal values for the parameters $\theta$ we define the maximum likelihood estimator:
$$
\begin{equation}
\begin{split}
\theta_{ML} & = \operatorname*{arg\,max}_\theta \ p_{model}(X; \theta) \\
            & = \operatorname*{arg\,max}_\theta \prod_{i=1}^{m}P_{model}(x^{(i)};\theta)
\end{split}
\end{equation}
$$
To simplify the calculation of the maximum likelihood estimator, we can take the log of the probabilities without changing the maximum, but replacing the product by a sum:
$$
\theta_{ML} = \operatorname*{arg\,max}_\theta \sum_{i=1}^{m} \log P_{model}(x^{(i)};\theta)
$$
This can be converted to the expectation of the maximum likelihood with respect to the probability distribution of the observed data $\hat p_{data}(\textbf x)$ (by scaling by $\frac{1}{m}$, which does not change the maximum) :
$$
\begin{equation}
\begin{split}
\theta_{ML} & = \operatorname*{arg\,max}_\theta \sum_{i=1}^{m} \log P_{model}(x^{(i)};\theta) \\
            & = \operatorname*{arg\,max}_\theta \sum_{i=1}^{m} \frac{1}{m} \log P_{model}(x^{(i)};\theta) \\
            & = \operatorname*{arg\,max}_\theta E_{\textbf x \sim \hat P_{data}}[log P_{model}(\textbf x;\theta)]
\end{split}
\end{equation}
$$
As we have seen in formula (9), the Kullback-Leibler (KL) Divergence between the probability distribution of the observed data $\hat P_{data}(\textbf x)$ and the probability distribution of our model $P_{model}(\textbf x)$ is defined by:
$$
D_{KL}(\hat P_{data}||P_{model}) = E_{x\sim \hat P_{data}}[\log \hat P_{data}(\textbf x)- \log P_{model}(\textbf x; \theta)]
$$
If we train our model to minimize the KL-Divergence, we only have to minimize the right term $- \log P_{model}(\textbf x; \theta)$ as the left term $\log \hat P_{data}(\textbf x)$ is only dependent on the training data and is not changed by the training of the model. Minimizing the right term is the same as maximizing formula (14):
$$
\operatorname*{arg\,min}_\theta - \log P_{model}(\textbf x; \theta) = \operatorname*{arg\,max}_\theta E_{\textbf x \sim \hat P_{data}}[log P_{model}(\textbf x;\theta)]
$$
As we have seen in formula (11), minimizing the KL-Divergence is equivalent to minimizing the cross-entropy.  That means that:

* Maximum likelihood
* Minimum negative log likelihood (NLL)
* Minimum KL-Divergence and
* Minimum cross-entropy

are all equivalent.

One reason that the maximum likelihood is often used as the preferred estimator for machine learning applications is that it can be shown that under appropriate conditions, the maximum likelihood estimator converges to the true values of the parameters, if the number of training examples approaches infinity. 

That means that the maximum likelihood estimator is the best possible estimator with the number of training examples asymptotically approaching infinity (see Goodfellow et.al. Deep Learning Book, p. 134).

#### Conditional Log-Likelihood

*(Goodfellow et.al. Deep Learning Book, p. 133)*

Now we have to generalize maximizing the likelihood estimation of a parameterized probability distribution to the supervised learning scenario, where we have to estimate the conditional probability $P(\textbf y |\textbf x; \theta)$ in order to predict $\textbf y$ given $\textbf x$. If the set $\textbf X$ presents all our input vectors and $\textbf Y$ all our observed target vectors, then the conditional maximum likelihood estimator is:
$$
\theta_{ML} = \operatorname*{arg\,max}_\theta \ P(Y|X;\theta)
$$
If the examples are supposed to be i.i.d. (independent and identically distributed) then this can be decomposed into:
$$
\theta_{ML} = \operatorname*{arg\,max}_\theta \sum_{i=1}^m P(y^{(i)}|(x^{(i)};\theta)
$$
