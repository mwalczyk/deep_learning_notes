# Optimization for Training Deep Models

* The goal of a machine learning algorithm is to reduce the expected generalization error or **risk**. Here, the expectation is taken over the true underlying distribution $$p_{data}$$. If we knew the true distribution $$p_{data}(x, y)$$, risk minimization would be an optimization task solvable by an optimization algorithm. When we do _not _know $$p_{data}(x, y)$$ but only have a training set of samples, we have a machine learning problem.
* The simplest way to convert a machine learning problem back into an optimization problem is to minimize the expected loss on the training set. This **empirical risk **can be written as $$\frac{1}{m}\sum_{i=1}^{m}L(f(x^{(i)};\theta), y^{(i)})$$. This process is known as **empirical risk minimization**. However, it has several problems.
  * Empirical risk minimization is prone to overfitting. 
  * In many cases, empirical risk minimization is not feasible, since many useful loss functions have no useful derivatives or the derivative is either zero or undefined everywhere. This makes it difficult \(or impossible\) to optimize with gradient descent.
  * The two problems above mean that we rarely use empirical risk minimization. Instead, we use a slightly different approach in which the quantity that we actually optimize is even more different from the quantity that we truly want to optimize. For example, instead of using a [0-1 loss](https://stats.stackexchange.com/questions/284028/0-1-loss-function-explanation), we might use the negative log-likelihood of the correct class.
  * This is known as a **surrogate loss function**.
* Optimization algorithms that use the entire training set are called **batch **gradient methods. Optimization algorithms that use only a single example at a time are called **stochastic **methods. Most algorithms fall somewhere in between. These are called **minibatch **methods.
  * It is crucial that the minibatches be selected randomly.
  * 





