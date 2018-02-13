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
  * **Other output units**: in general, we can think of the neural network as representing a function whose outputs are not direct predictions. Instead, this function $$f(x;\theta) = \omega$$ provides the parameters for a distribution. 
    * For example, we may wish to learn the variance of a conditional Gaussian for $$y$$ given $$x$$.
    * We often want to perform multimodal regression, that is, to predict real values from a conditional distribution $$p(y\mid x)$$ that can have several different peaks in $$y$$ space for the same value of $$x$$. In this case, a Gaussian mixture is a natural representation for the output. Neural networks with Gaussian mixtures as their output are often called **mixture density networks **\(MDN\).
      * A typical neural network can only predict one output value for each input. A MDN can predict a range of different values for each input.
      * To do this, the network predicts an entire probability distribution for the output. Practically speaking, this means that the last layer of the network will have 3 outputs for each of the $$k$$ Gaussians: $$p(c = k\mid x)$$ for the $$k$$-th Gaussian \(these will be forced to sum to 1\), the mean $$\mu_{k}$$, and the standard deviation $$\sigma_{k}$$. 
      * All of this is explained in the following blog [post](http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/), which also contains a TensorFlow implementation of a MDN.
      * Gradient-based optimization of conditional Gaussian mixtures can be unreliable, in part due to the divisions \(by the variance\) which can be numerically unstable. This occurs when the variance gets to be very small for a particular example, yielding very large gradients. One solution is to **clip gradients**.
      * Gaussian mixture outputs are effective in generative models of speech and movement of physical objects. The MDN strategy gives the network a way to represent multiple output modes and to control the variance of its output, which is crucial for obtaining a high degree of quality in these real-valued domains.
* Some types of hidden units, such as those including ReLUs, are not actually differentiable at all input points. In practice, this is not an issue because neural network training algorithms do not usually arrive at a local minimum of the cost function but instead, merely reduce its value significantly. Because we do not expect training to actually reach a point where the gradient is 0, it is acceptable for the minima of the cost function to correspond to points with an undefined gradient.
* There are many different types of hidden units:
  * **Rectified linear unit**: this unit is typically used on top of an affine transformation $$h = g(W^{T}x + b)$$. When initializing the parameters of the transformation, it is good practice to set all elements of $$b$$ to a small positive value, such as 0.1. Doing so makes it very likely that the ReLU will be initially active for most inputs in the training set. One drawback to ReLUs is that they cannot learn via gradient-based methods on examples for which their activation is zero. The **leaky ReLU** or **parametric ReLU** variations attempt to fix this. 
  * **Maxout unit**: this unit generalizes the ReLU by dividing $$z$$ into groups of $$k$$ values. Each maxout unit then outputs the maximum element of one of these groups. This provides a way of learning a piecewise linear function that responds to multiple directions in the input $$x$$ space.
    * When $$k = 2$$, a maxout neuron computes the function $$\max(w_{1}^{T}x + b_{1}, w_{2}^{T} + b_{2})$$.
    * Both ReLU and leaky ReLU are a special case of this form. For example, a ReLU sets $$w_{1} = b_{1} = 0$$.
    * A maxout neuron enjoys all of the benefits of a ReLU unit and does not have its drawbacks \(dead ReLUs\). However, using maxout neurons _doubles_ the number of parameters needed for every single neuron.
  * **Sigmoid and hyperbolic tangent units**: these units compute $$\sigma(z)$$ and $$\tanh(z)$$, respectively. Their use as hidden units in feedforward networks is discouraged because they can lead to poor training dynamics. Sigmoidal activation functions are more common in recurrent networks, probabilistic models, and autoencoders, which have additional requirements that rule out the use of piecewise linear activation functions.
  * **Other hidden units**: softmax units are usually used as an output unit but may sometimes be used as a hidden unit. They can be interpretted as a kind of "switch." A few other common hidden unit types include:
    * Radial basis function 
    * Softplus
    * Hard tanh

![](/assets/activation_functions.png)

* The word **architecture** refers to the overall structure of the network: how many units it should have and how these units should be connected to each other.
* The **universal approximation theorem **states that a feedforward network with a linear output layer and at least one hidden layer with any activation function can approximate any continous function on a closed and bounded subset of $$R^{n}$$ with any desired nonzero amount of error, provided that the network is given enough hidden units. A proof of this theorem can be found in the following blog [post](http://mcneela.github.io/machine_learning/2017/03/21/Universal-Approximation-Theorem.html).
* Even though this theorem means that a network _can_ \(in theory\) represent any function that we are trying to learn, this doesn't mean that it \_will \_in practice. Learning can fail for two reasons:
  * The optimization algorithm used for training may not be able to find the value of the parameters that corresponds to the desired function.
  * The training algorithm might choose the wrong function as a result of overfitting.
* In many circumstances, using a deeper model can reduce the number of units required to represent the desired function and reduce the amount of generalization error.

> Choosing a deep model encodes a very general belief that the function we want to learn should involve the composition of several simpler functions. This can be interpreted from a representation learning point of view as saying that we believe that the learning problem consists of discovering a set of underlying factors of variation that can, in turn, be described in terms of other, simpler underlying factors of variation.

* The term **back-propagation **is often misunderstood as meaning the \_whole \_learning algorithm for multi-layer neural networks. Actually, back-propagation refers only to the method for computing the gradient, while another algorithm \(like SGD\) is used to perform learning using this gradient.
* To facillitate back-propagation, we use the notion of a **computational graph**, where each node represents either:
  * A variable \(scalar, vector, matrix, tensor, or otherwise\).
  * An operation \(function of one or more variables\).
* Back-propagation is a highly efficient algorithm that computes the **chain rule** of calculus with a specific order of operations. The chain rule states that for a real number $$x$$ and two functions $$y = g(x)$$ and $$z = f(g(x)) = f(y)$$, the derivative of $$z$$ with respect to $$x$$ is: $$\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}$$. The chain rule is explained in detail in the following Khan Academy [article](https://www.khanacademy.org/math/differential-calculus/product-quotient-chain-rules-dc/chain-rule-dc/a/chain-rule-review).
* We can generalize the chain rule beyond the scalar case. Suppose that $$x\in R^{m}$$ and $$y\in R^{n}$$. Let $$g$$ be an intermediate function that maps from $$R^{m}$$ to $$R^{n}$$. Finally, let $$f$$ be a function that maps from $$R^{n}$$ to $$R$$. The gradient of $$z$$ with respect to $$x$$ can be written as the product of a Jacobian matrix $$\frac{dy}{dx}$$ and a gradient $$\bigtriangledown_{y}z$$. The back-propagation algorithm consists of performing such a Jacobian-gradient product for each operation in the graph.
* Usually we apply the back-propagation algorithm to tensors of arbitrary dimensionality, not just vectors. Conceptually, this is the exact same. The only difference is how the numbers are arranged in a grid to form a tensor. We could imagine flattening each tensor into a vector before we run back-propagation, computing a vector-valued gradient, and then reshaping the gradient back into a tensor. In this view, back-propagation is still just multiplying Jacobians by gradients.
* Computation graphs are explained in detail in the following blog [post](http://colah.github.io/posts/2015-08-Backprop/) and on the [course website](http://cs231n.github.io/optimization-2/#backprop) for Stanford's CS231n. In particular, there are two ways of calculating derivatives on computational graphs:
  * **Forward-mode** differentiation tracks how one _input_ affects every node by applying the operator $$\frac{\partial}{\partial{X}}$$ to every node for some input $$X$$.
  * **Reverse-mode** differentiation tracks how every node affects one _output_ by applying the operator $$\frac{\partial{Z}}{\partial}$$ to every node. The advantage of reverse-mode differentiation is that it gives us the derivative of some output $$Z$$ with respect to _every node_. This means that we can calculate all of the partial derivatives necessary for back-propagation in one pass.
* Both modes involve "factoring" the edges of the graph, so as to avoid the combinatorial explosion in the number of possible paths. An example of forward-mode differentiation \(from node $$b$$ upwards\) can be seen in the image below.

![](/assets/forward_mode_differentiation.png)

* Computational graphs operate on **symbols**, or variables that do not have specific values. These algebraic and graph-based representations are called **symbolic representations**. When we actually use or train a neural network, we must assign specific values to these symbols. We replace a symbolic input to the network $$x$$ with a specific numeric value, such as $$[1, 2, 3]^{T}$$. Some approaches to back-propagation take a computational graph and a set of numerical values for the inputs to the graph, then return a set of numerical values describing the gradient at those input values. We call this approach **symbol-to-number** differentiation.
* Another approach is to take a computational graph and add addtional nodes to the graph that provide a symbolic description of the desired derivatives. This is the approach taken by Theano and TensorFlow and is known as **symbol-to-symbol** differentiation. The advantage here is that the derivatives are described in the same language as the original expression. Any subset of the graph can be evaluated using specific numerical values at a later time. 
* 




