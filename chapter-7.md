# Regularization for Deep Learning

* In the context of deep learning, most regularization strategies are based on regularizing estimators, which works by trading increased bias for reduced variance. 

> An effective regularizer is one that makes a profitable trade, reducing variance significantly while not overly increasing the bias.

* In practice, an overly complex model family does not necessarily include the target function or the true data-generating process, or even a close approximation of either. The **data-generating process **is the true, underlying phenomenon that is creating the data. The **model** is the \(often imperfect\) attempt to describe and emulate the phenomenon, as described in the following Quora [post](https://www.quora.com/How-is-the-data-generating-process-DGP-different-from-the-model-in-regression-analysis).
* Many regularization approaches are based on limiting the capacity of models by adding a parameter norm penalty $$\Omega (\theta)$$ to the objective function $$J$$. We denote the regularized objective function by $$\widetilde{J}$$:

$$\widetilde{J}(\theta ; X, y) = J(\theta ; X, y) + \alpha \Omega (\theta)$$

* Generally, we choose a parameter norm penalty that penalizes \_only the weights \_of the network and leaves the biases unregularized. This is because the bias parameters don't contribute to the curvature of the model, so there is no point in regularizing them. The learning algorithm cannot put arbitrarily large values for the bias term since this will result in a grossly large loss value. In other words, given some training set, the learning algorithm cannot move the separating hyperplane arbitrarily far away from the true one.
* The $$L^{2}$$ parameter norm penalty is commonly known as **weight decay**. This regularization strategy drives the weights closer to the origin by adding a regularization term $$\Omega (\theta) = \frac{1}{2}||w||_{2}^{2}$$ to the object function. It is sometimes referred to as **ridge regression**.
* When using weight decay, a single gradient step to update the weights involves multiplicatively shrinking the weight vector by a constant factor. If we let $$w^{*}$$denote the value of the weights that obtains minimal unregularized training cost, then we can show that the effect of weight decay is to rescale $$w^{*}$$along the axes defined by the eigenvectors of the Hessian matrix $$H$$.

![](/assets/weight_decay.png)

* Only directions along which the parameters contribute significantly to reducing the objective function are preserved relatively intact. In directions that do not contribute to reducing the objective function, a small eigenvalue of the Hessian tells us that movement in this direction will not significantly increase the gradient. Components of the weight vector corresponding to such unimportant directions are decayed away through the use of the regularization throughout training.
* We can also study the effect of $$L^{2}$$ regularization on a simple model like linear regression. 
  * The cost function with regularization can be written as $$(Xw - y)^{T}(Xw - y) + \frac{1}{2}\alpha w^{T}w$$, where $$X$$ is the training data. 
  * This changes the **normal equations** for the solution from $$w = (X^{T}X)^{-1}X^{T}y$$ to $$w = (X^{T}X + \alpha I)^{-1}X^{T}y$$. 
  * The matrix $$X^{T}X$$ is proportional to the covariance matrix $$\frac{1}{m}X^{T}X$$. This follows from the fact that if the vectors \(i.e. rows of $$X$$\) are centered random variables, then the [Gram matrix](https://en.wikipedia.org/wiki/Gramian_matrix) \(which is $$X^{T}X$$\) is approximately proportional to the covariance matrix, with the scaling determined by the number of elements in the vector \(which is $$m$$\).
  * The new matrix in the parenthesis is the same as the original one but with the addition of $$\alpha$$ to the diagonal. The diagonal entries of this matrix correspond to the variance of each input feature. We can see that $$L^{2}$$ regularization causes the learning algorithm to "perceive" the input $$X$$ as having higher variance, which makes it shrink the weights on features whose covariance with the output target is low compared to this added variance. 



