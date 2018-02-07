# Deep Feedforward Networks

* It is best to think of feedforward networks as **function approximation machines** that are designed to achieve statistical generalization.
* **Linear models** \(such as logistic regression and linear regression\) have the obvious defect that the model capacity is limited to linear functions, so the model cannot understand the interaction between any two input variables. To extend linear models to represent non-linear functions of $$x$$, we can apply the linear model not to $$x$$ itself but to a transformed input $$\phi(x)$$, where $$\phi$$ is a non-linear transform. We can think of $$\phi$$ as providing a new representation for $$x$$.
* The excerpt below explains why a simple linear model cannot solve the XOR problem, which serves as the motivation for non-linear activation functions in deep feedforward networks:

![](https://www.evernote.com/shard/s463/res/c422c35f-b6c3-4910-b71d-9a4d83374928/Capture.PNG)

* In modern neural networks, the default recommendation is to use the **rectified linear unit**, or ReLU, defined by the activation function $$g(z) = \max (0, z)$$.
  * [How is the ReLU activation function able to approximate non-linear functions?](https://stats.stackexchange.com/questions/299915/how-does-the-rectified-linear-unit-relu-activation-function-produce-non-linear)
  * Suppose we want to approximate the function $$f(x) = x^{2}$$ using ReLUs $$g(ax + b)$$. One approximation might look like $$h_{1}(x) = g(x) + g(-x) = |x|$$. This is shown in the first graph below.
  * This obviously isn't a very good approximation. We can add more terms to improve the approximation, like: $$h_{2}(x) = g(x) + g(-x) + g(2x - 2) + g(-2x + 2)$$. This is shown in the second graph below.

![](https://www.evernote.com/shard/s463/res/9cda5bde-81dc-4f9d-bb7f-90fe8a3bc5dc.png "https://i.stack.imgur.com/R14EO.png")

![](https://www.evernote.com/shard/s463/res/45505ccc-69e1-4ce4-9b70-ffb6a9b55eea.png "https://i.stack.imgur.com/GUDKV.png")

* The largest difference between the linear models we have seen so far and neural networks is that the non-linearity of a neural network causes most interesting loss functions to become non-convex. This means that neural networks are usually trained by driving the cost function to a very low value rather than an absolute global minimum.
* Most modern neural networks are trained using maximum likelihood. This means that the cost function is simply the negative log-likelihood, equivalently described as the cross-entropy between the training data and the model distribution.
* We can think of learning as choosing a function rather than merely choosing a set of parameters.
  * As a review, note that:
    * **Probability** lets us predict unknown outcomes based on known parameters.
    * **Likelihood** lets us predict unknown parameters based on known outcomes.
    * This is explained in the following [article](http://www.onmyphd.com/?p=mle.maximum.likelihood.estimation).
    * An example of using MLE to solve linear regression can be found in the following [article](http://suriyadeepan.github.io/2017-01-22-mle-linear-regression/).
    * Consider the problem of binary classification. Maximizing the \(log\) likelihood of the data under a Bernoulli distribution is equivalent to minimizing the binary cross-entropy. This can be extended to the multi-class case using softmax cross-entropy and the so-called multinoulli likelihood. This is derived in the following Quora [post](https://www.quora.com/What-are-the-differences-between-maximum-likelihood-and-cross-entropy-as-a-loss-function).
    * The difference between MLE and cross-entropy is that MLE represents a structured, principled approach to modeling and training, whereas cross-entropy simply represents a special case of this applied to a type of classification problems that people typically care about.
    * The connection between MLE and expectation is explained in the following [post](https://stats.stackexchange.com/questions/286576/maximum-likelihood-as-an-expectation).
    * During MLE, we almost always convert the joint probability distribution of our dataset into a sum, using logarithms. We do this because a sum is a lot easier to optimize, since the derivatives of a sum is the sum of its derivatives:
      * By using a sum, we can load each training example \(one at a time\), compute its partial derivatives, and accumulate those gradients.
      * If we were using a product, all of the training examples are "entangled," so we would need to load the entire training set to calculate the product. Only then would we be able to compute the partial derivatives of all of the parameters and apply an optimization step.
    * This is [why we use the logarithm](https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability): to get rid of the product signs.
    * Of course, maximizing the log-likelihood of the parameters given a dataset is strictly equivalent to minimizing the negative log-likelihood. All of the bullet points above are discussed in the following [article](https://blog.metaflow.fr/ml-notes-why-the-log-likelihood-24f7b6c40f83).
    * The following [article](http://blog.shakirm.com/2015/05/a-statistical-view-of-deep-learning-v-generalisation-and-regularisation/) contains a table of different regularization strategies and their probabilistic interpretations as a prior used during MAP estimation.
* Any kind of neural network unit that may be used as an output can also be used as a hidden unit. Throughout this book, we suppose that the feedforward network provides a set of hidden features defined by $$h = f(x;\theta)$$. The role of the output layer is then to provide some additional transformation from the features to complete the task that the network must perform.
* There are many different types of output layers:
  * **Linear unit**: this unit is based on an affine transformation $$W^{T}h + b$$. It is often used to produce the mean of a conditional Gaussian distribution. Maximizing the log-likelihood is then equivalent to minimizing the mean squared error.
  * **Sigmoid unit**: this unit is often used for binary classification. In this case, the maximum likelihood approach is to define a Bernoulli distribution over $$y$$ conditioned on $$x$$. Because the cost function used with maximum likelihood is $$-\log P(y\mid x)$$, the $$\log$$ in the cost function undoes the $$\exp$$ of the sigmoid. This helps prevent the saturation that would normally occur with a different loss function, such as mean squared error.
  * **Softmax unit**: this unit is often used to represent a probability distribution over a discrete variable with $$n$$ possible values. This can be seen as an extension of the sigmoid unit, where we used exponentiation and normalization to give us a Bernoulli distribution controlled by the sigmoid function.
    * Overall, unregularized maximum likelihood will drive a model to learn parameters that drive the softmax to predict the fraction of counts of each outcome observed in the training set. In practice, limited model capacity and imperfect optimization will mean that the model is only able to approximate these fractions.
    * If we have $$n$$ classes, a linear layer before the softmax unit will actually overparametrize the distribution. The constraint that the $$n$$ outputs must sum to 1 means that only $$n - 1$$ parameters are necessary: the probability of the $$n$$-th value may be obtained by subtracting the first $$n - 1$$ probabilities from 1. To account for this, we can impose a requirement that one element of $$z$$ be fixed at 0. 
    * This is exactly what the sigmoid unit does: defining $$P(y = 1\mid x) = \sigma(z)$$ is equivalent to defining $$P(y = 1\mid x) = softmax(z)_{1}$$ with a 2-dimensional $$z$$ and $$z_{1} = 0$$. Note that in practice, there is rarely much different between using the overparametrized version or the restricted version.
  * **Other output units**: in general, we can think of the neural network as representing a function whose outputs are not direct predictions. Instead, this function provides the parameters for a distribution. For example, we may wish to learn the variance of a conditional Gaussian for y given x.



