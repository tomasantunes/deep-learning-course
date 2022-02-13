# deep-learning-course
Notes and code from Deep Learning course

Source: https://www.udemy.com/share/104Yle3@BcxH9NvdWlQe6vqfV5QPbR-oh8dIXeIT5absONlOvlrCCz17O38tCqkYfaKrL54=/

# Deep Learning Course

An artificial neural network is a way to transform an input into an output. It can be used to target ads based on your browsing history, to detect credit card fraud, to detect diseases, for self-driving cars or for translations.

![Image1](https://imgur.com/Bzw9v59.jpg "Image1")

In this example we could use a logistic regression or a support vector but we're going to use an ANN.

We have two features and X0 is the bias, then we get a weighted sum, then we apply a nonlinear activation function like a sigmoid and then we get the output.

![Image2](https://imgur.com/i1PMGCQ.jpg "Image2")

In this case the solution is nonlinear and deep learning can help us.

![Image3](https://imgur.com/lMlMH9g.jpg "Image3")

To get a nonlinear solution we take this equation and pass it through a function.

![Image4](https://imgur.com/c7t1jwa.jpg "Image4")

We can put together many of these functions and we get this diagram:

![Image5](https://imgur.com/VJIPU9n.jpg "Image5")

Each of these circles is an artificial neuron with the equation.

If you have a table to predict if the students pass or fail you use the ANN. If you have to predict if a picture contains a cat you use a CNN. If you have an audio clip of cough and you want to detect covid you use RNN and to make translations you also use RNN.

## How Models Learn

Imagine you are making PB&J sandwiches. You improve your sandwich by getting feedback from the customer. If he says "it's too sweet" you can reduce the jelly or increase the peanut butter or both. So you systematically make these adjustments and make many sandwiches until the customer is happy.

![Image6](https://imgur.com/GCLPlIi.jpg "Image6")

![Image7](https://imgur.com/sbcBiaT.jpg "Image7")

In forward propagation you move from the left to the right. In back propagation you go back and adjust the weights based on negative feedback.

A more complex example would be a PB&J business.

![Image8](https://imgur.com/FuYSghB.jpg "Image8")

In this example the owner would see that the profit is not as high as expected so he would talk to the marketing team for them to change behaviour and the marketing team would talk to the kitchen.

## Science of Deep Learning

Deep learning is one of the most promising avenues for a system that could develop general itnelligence. Lots of small, simple things is better than one complex thing.

DL is fundamentally empirical. We cannot know how things work from first principles. Instead, we tinker and experiment and explore and discover.

Society is increasingly putting blind trust into a technology we don't understand. It's different from a car or phone.

Our approach to understanding the universe is to go from theory to experiments and refining the theory.

With deep learning we forget about theory and let the model learn the complexity without us having to impose or imagine what it might look like.

We are replacing our rigorous, explicit, mathematical, exact theories with empirical black-box approximations.

The universal approximation theorem states that a sufficiently rich deep learning model can approximate any mathematical function.

There are DL models that can design engines and solve problems in molecular biology.

No one knows if AI can become conscious. It is unclear whether consciousness requires biology. AI consciousness may be very different from human consciousness.

Agnostic is the only reasonable position.

There are three types of research you can do with deep learning:

- Theoretical: Heavy on theory / math development
- Ecological: Use existing (pretrained) models
- Experimental: Systematically modify model parameters and observe which is best

## Parametric experiments

A parametric experiment is repeating an experiment while systematically manipulating one or two variables.

Independent variable: The variables you manipulate.

Dependent variable: The key outcome variable you use to evaluate model performance.

## Problems with parametric experiments

Feasability: small or simple models are fast to compute but large models take a long time to train and evaluate.

Generalizability: Specific findings from one model may not replicate in other arquitectures or other sets of parameters.

Use the experimental approach to build intuition and gain expertise about DL modeling in general. It's more of an art than science.

## Difference between artificial neurons and real neurons

Neurons can have thousands of neighbours and there are many kinds of neurons. Computational neuroscience tries to simulate neurons and these are much closer to reality.

Real neurons can only perform the following operations: and-not, or, coincidence detection, lowpass filter, atenuation, segregation and amplification.

The artificial neuron was developed by McCulloch and Pitts. It doesn't compare to the complexity of a real neuron.

A car is not the same as a horse, a plane is not the same as bird. Technology is amazing for what it is, it doesn't need to be like biology.

Technology can advance better and faster if we unhinge it from useless and forced biological constraints.

## Spectral theories in mathematics

The idea is that you start with something somplicated and you break it up into simple components.

A complicated system has lots of parts, is linear and intuitive and understandable. A complex system has fewer parts, many nonlinearities and is counter-intuitive and difficult to understand.

Deep learning is simple, complicated and complex. It's simple because it's made of simple, easy math. It's complicated because it contains many parts. It's complex because of the nonlinearities and being difficult to understand.

## Terms and datatypes

In linear algebra we have scalars, vectors, matrices and tensors.

Images are stored as matrices. Color images are tensors or three-dimensional matrices.

Data types can be integers, floats, booleans, strings, etc. We're going to use ND arrays and tensors.

## Converting reality to numbers

There are two types of reality: continuous (numeric) and categorical (discrete).

To represent categorical data we can use dummy-coding where we create a single vector with 0 and 1 values. With one-hot encoding we create a matrix with 0 and 1 values per category.

## Vector and matrix transpose

When we transpose the rows become columns or the columns become rows. The first row becomes the first column.

## Dot product

![Image9](https://imgur.com/d5gAQDr.jpg "Image9")
![Image10](https://imgur.com/WIovtXZ.jpg "Image10")

The dot product between two vectors is always a single number. It is only defined for vectors with the same number of elements.

To calculate the dot product between two matrices we multiply the numbers row by row.

The dot product is a single number that reflects the commonalities between two objects.

## Matrix Multiplication

Matrix size is defined in MxN for M rows and N columns.

The rule for validity of a matrix multiplication is that the number of columns in the first matrix must be equal to the number of rows in the second matrix.

The result size is going to be the number of rows in the first matrix by the number of columns in the second matrix. You can transpose matrices to make the multiplication valid.

![Image11](https://imgur.com/c138Tyq.jpg "Image11")

Each individual element of the product matrix is the result of the corresponding row in the left matrix and the corresponding column of the right matrix.

## Softmax

![Image12](https://imgur.com/jbkVMDF.jpg "Image12")

The softmax function takes a set of numbers and converts them to probabilities between 0 and 1 and together they add up to 1. The sum over the inputs can be any numerical value but the sum over the outputs is always 1.

## Logarithms

The logarithm is similar to the exponent but the increase in Y slows down as X gets bigger.

This creates larger values for small probabilities that are close to 0 which makes them easier to compute.

## Entropy

Entropy measures the amount of uncertainty in a variable. It's maximum value is at 0.5. And it's the most predictable at 0 and 1. At 0 this event never happens and at 1 this event always happens.

![Image13](https://imgur.com/yQUYB2K.jpg "Image13")

High entropy means that the dataset has a lot of variability. Low entropy means that most of the values at the dataset repeat (and therefore are redundant).

Entropy differs from variance because entropy is nonlinear and makes no assumptions about the distribution. Variance depends on the validity of the mean and therefore is appropriate for roughly normal data.

If you use logarithm base 2 the units are bits. If you use natural logarithm the units are nats.

## Cross-entropy

![Image14](https://imgur.com/LgN7tUk.jpg "Image14")

Cross-entropy describes the relationship between two probability distributions. It's used to measure the performance of a model.

## Min/max, argmin/argmax

Min/max finds the lowest and highest values in a set of numbers. Argmin/argmax finds the location of the lowest and highest values.

## Mean

![Image15](https://imgur.com/eGygVgo.jpg "Image15")

The mean is suitable for roughly normally distributed data. The suitable data types are intervals and ratios.

## Variance

![Image16](https://imgur.com/3wUtM3K.jpg "Image16")

The variance is suitable for any distribution. The suitable data types are: numerical, ordinal. The standard deviation is the square root of the variance.

## Sample variability

Different samples from the same population can have different values of the same measurement.

A single measurement may be an unreliable estimate of a population parameter.

Variability exists because of natural variation, because of measurement noise and because of complex systems.

So we have to take many samples. Averaging together many samples will approximate the true population mean. It's the law of large numbers.

DL models learn by examples, non-random sampling can introduce systematic biases in DL models. Non-representative sampling causes overfitting and limits generalizability.

## Reproducible randomness via seeding

In DL we sometimes define random initial states. If we want other people to reproduce our model we need to use seeds.

## T-test

When we run experiments we might need to measure the performance of two different distributions.

We're going to have a null hypothesis with data points from both models and if the accuracy is better than the null hypothesis we choose that model.

![Image17](https://imgur.com/ErWY6Jt.jpg "Image17")

## Derivatives

The derivative of a function tells us how the function is changing over the x variable. It's the slope of the function at each point.

## Derivative of a polynomial

![Image18](https://imgur.com/1DUB2VU.jpg "Image18")

Derivatives point us in the direction of increases and decreases in a mathematical function.

In DL, the goal is represented as an error. Thus, the best solution is the point with the smallest error. The derivative tells us which way to move in that error landscape in order to find the optimal solution.

DL wouldn't work without derivatives.

## Find local minima and maxima

When the derivative crosses the value 0 those will be our minima and maxima.

We set the derivative to 0 and solve for x.

To distinguish the minima from the maxima we look at the neighbour points of 0.

Minima have negative values to the left and positive values to the right.

Maxima have positive values to the left and negative values to the right.

## Product Rule

![Image19](https://imgur.com/PDlWy98.jpg "Image19")

## Chain Rule

![Image20](https://imgur.com/uROFK5N.jpg "Image20")

## Gradient Descent

How deep learning models work:
- Guess a solution
- Compute the error (mistakes)
- Learn from mistakes and modify the parameters

We need a mathematical description of the error "landscape" of the problem and we need a way to find the minimum of the landscape.

![Image21](https://imgur.com/xZqweQ8.jpg "Image21")

## Gradient Descent Algorithm

- Initialize random guess of minimum
- Loop over training iterations
- Compute derivative at guess min
- Updated guess min is itself minus derivative scaled by learning rate

## Local minima

Gradient descent is guaranteed to go downhill. Going downhill does not guarantee that we will find the correct solution. Gradient descent can go wrong if parameters are not set right for the particular error landscape. Error landscapes are impossible to visualize in 2D.

![Image22](https://imgur.com/djTL2nR.jpg "Image22")

The success of deep learning, in spite of the problems with gradient descent remains a mystery.

It is possible that there are many good solutions. This interpretation is consistent with the huge diversity of weight configurations that produce similar model performance.

Another possibility is that there are extremely few local minima in high-dimensional space. This interpretation is consistent with the complexity and absurd dimensionality of DL models.

In higher dimensions we could have a "saddle point" which is a local minima in one direction and local maxima in another direction.

Gradient descent will get trapped in a local minimum only if that point is a minimum in all direction.

If the model performance is good you don't have to worry about it.

A possible solution is to re-train the model many times using different random weights and pick the model that does best.

Another solution is to increase the dimensionality to have fewer local minima.

A vanishing gradient happens when the derivative is so close to 0 that the learning effectively stops.

## Parametric experiments

To run experiments we define a range of starting values, learning rates or training epochs. We use a loop or a nested loop for two variables and we store the results in a vector or matrix.

## Dynamic learning rate

If we change our learning rate to be dynamic through multiplying it by the gradient the learning rate will be larger when we are far away from the minima. The close we get to the minimum the smaller will be the learning rate.

We can also use training epochs. This method is unrelated to the model performance, it's called "learning rate decay".

The derivative method is adapative to the problem and is incorporated into the RMSprop and Adam optimizers.

## Vanishing and exploding gradients

When the function gets flat because it remains constant the derivative gets smaller and smaller and our model gets stuck, this is called a vanishing gradient.

When the derivative becomes very steep our next step might be too large and we could end up skipping over the minima. This is called an exploding gradient.

A vanishing gradient means the weights don't change and this is problematic for deep networks. An exploding gradient means the weights change wildly and this gives us bad solutions.

## How to minimize gradient problems

- Use models with few hidden layers
- Use activation functions that do not saturate (ReLU)
- Apply weight normalization
- Use regularization techniques like batch normalization, dropout and weight decay.
- Use arquitectures like residual networks (resnet)

## Artificial Neural Networks

## The perceptron

![Image23](https://imgur.com/NZBNIE9.jpg "Image23")

Linear operations use addition and multiplication. Anything else is nonlinear.

Linear models can only solve linearly separable problems where you can use a line to separate two groups.

Nonlinear models can solve more complex problems. Never use a linear model for a nonlinear problem and never use a nonlinear model for a linear problem.

The output of the perceptron applies a linear operation and passes it through a nonlinear activation function.

![Image24](https://imgur.com/SDiAStn.jpg "Image24")

The bias term helps us create a linear separation that doesn't pass through the origin. It's also called an intercept.

You don't always need a bias term, you can transform the data to be mean-centered.

![Image25](https://imgur.com/u1wBWiX.jpg "Image25")

The feature space is a geometric representation of the data, where each feature is an axis and each observation is a coordinate.

The separating hyperplane is aboundary that binrizes and categorizes data. It is used as a decision boundary.

Model output can be discrete or continuous.

The most common activation functions are the sigmoid, hyperbolic tangent and ReLU.

The sigmoid is usally used for the final output of the model and the other two are used in the middle.

To make a good model we need to pick the right weights. To do this we need to learn from data, via back-propagation with the gradient descent.

## Error

![Image26](https://imgur.com/Ftmr9YX.jpg "Image26")

Binarized error is easier to interpret but less sensitive. Continuous error is more sensitive but is signed.

We use the continuous error to teach the model and we use the binarized error to evaluate the model.

## Loss functions

The two most common loss functions are the Mean-squared error and the cross-entropy.

The mean-squared error is used for continuous data when the output is a numerical prediction.

![Image27](https://imgur.com/MdcG3oA.jpg "Image27")

Cross-entropy (also called logistic error function) is used for categorical data when the output is a probability.

![Image28](https://imgur.com/a1FxdUm.jpg "Image28")

The loss function is computed for each sample. The cost function is the average of the losses for many samples.

![Image29](https://imgur.com/kPL3PZz.jpg "Image29")

The goal of DL is to find the set of weights that minimizes the cost.

Training on each sample is time-consuming and may lead to overfitting. But averaging over too many samples may decrease sensitivity. A good solution is to train the model in "batches" of samples.

In a deep network each node is a perceptron taking in multiple inputs and passing the output to the nodes in the next layer.

Each node is an independent node. It doesn't know where the inputs come from and where the outputs are going.

In forward propagation we compute an output based on the input.

In back propagation we adjust the weights based on loss/cost.

For each activation function we update the weights by subtracting the learning rate times the derivative of the loss.

## ANNs for regression

Regression means to predict one continuous variable from another.

![Image30](https://imgur.com/h2MVJfT.jpg "Image30")

Sometimes we don't use DL models and we use traditional statistical models because they tend to work better on smaller datasets, are better mathematically characterized, they have guaranteed optimal solutions and are more interpretable.

DL doesn't really predict values, it learns relationships across variables which might be too complex for humans.

## Binarizing the model output

![Image31](https://imgur.com/z5NrxIT.jpg "Image31")

Why have a sigmoid? Why not just threshold the raw model output?

The restricted numerical range increases stability and accuracy. It prevents errors from being too large and it makes forward prop more stable when the sigmoid is internal.

For linear problems you can remove the nonlinear functions from your model and that will increase the accuracy.

There is no such thing as multilayer linear models. Without nonlinearities layers collapse because all the weights combine into one layer.

"Fully connected" means that each node in layer n projects to each node in layer n+1. Each connection has it's own weight.

Depth vs. breadth has implications on the total number of units and weights. A deeper model can learn more abstract and complex representations of the data but deeper models are not necessarily better. Breadth is also called width.

![Depth vs. Breadth](https://imgur.com/RXvsuta.jpg "Depth vs. Breadth")

## Are DL models understandable?

The model is simple: every node implements an equation so simple we could compute it by hand.

The model is complex: the nonlinearities and interactions across hundreds of parameters means that we have no idea what each node is actually encoding.

DL is best for complex problems classification tasks when you don't need to know how the classification works.

DL is less appropriate for gaining mechanistic insights into how a system behaves and why.

Traditional statistical models (ANOVA or regression) are more appropriate for mechanistic insights into system behaviour.

## Overfitting

Overfitting limits our ability to generalize to new data.

Overfitting:
- overly sensitive to noise
- increased sensitivity to subtle effects
- reduced generalizability
- over-parameterized models become difficult to estimate

Underfitting:
- less sensitive to noise
- less likely to detect true effects
- reduced generalizability
- parameters are better estimated
- good results with less data

Overfitting is not intrinsically bad, it reduces generalizability, which may or may not be problematic depending on the goals and scope of the model.

## Cross-validation

Sometimes we need to do cross-validation to determine the correct number of parameters and avoid overfitting.

To do cross-validation you take 80% of the your data for training, 10% of your data for development and 10% of your data for testing.

So you go through a cycle of training, testing on the devset and adapting your model and when you are satisfied you use the test set.

In k-fold cross validation you use 90% of data for training and 10% for testing but you change the testing data on every iteration.

## Regularization

- Penalizes "memorization" (over-learning examples)
- Helps the model generalize to unseen examples
- Changes the representations of learning (either more sparse or more distributed depending on the regularizer)
- Can increase or decrease training time
- Can decrease training accuracy but increase generalization
- Works better for large models with multiple hidden layers
- Generally works better with sufficient data

There are three types of regularization:
- Modify the model (dropout)
- Add a cost to the loss function (L1/2)
- Modify or add data (batch training, data augmentation)

With dropout we remove nodes randomly during learning by forcing the activation to be 0.

With L1/L2 regulariation we add a cost to the loss function to prevent weights from getting too large.

Regularization adds a cost to the complexity of the solution. It forces the solution to be smooth and it prevents the model from learning item-specific details.

Dropout works because it prevents a single node from learning too much, it forces the models to have distributed representations and it makes the model less reliant on individual nodes and thus more stable.

It requires more training (though each epoch computes faster). It can decrease training accuracy but increase generalization. It usually works better on deeper networks and it needs sufficient data.

L1 regularization is also known as lasso and L2 regularization is also known as ridge. L1+L2 is known as elastic net.

Regularization reduces overfitting because it discourages compex and sample-specific representations. It prevents overfitting to training examples and it reduces large weights.

Training in batches can decrease computation time because of vectorization (matrix multiplication instead of for-loops).

Batching is a form of regularization: it smooths learning by averaging the loss over many samples and thereby reduces overfitting. All batches should be the same size.

## Metaparameters

Parameters are features of the model that are learned by the algorithms (mainly, the weights between nodes). You do not set the parameters.

Metaparameters are features of the model that are set by you, not learned automatically by the model.

Here are some metaparameters:
- Model arquitecture
- Number of hidden layers
- Number of units per layer
- Cross-validation sizes
- Mini-batch size
- Activation functions
- Optimization functions
- Learning rate
- Dropout
- Loss function
- Data normalization
- Weight normalization
- Weight initialization

It is impossible to search the entire metaparameter space. It is difficult to know whether you are using the best model for your problem. Fortunately, parametric experiments on some metaparameters are feasible. You should have some experience and intuition to choose metaparameters.

## Data normalization

Data normalization helps ensure that:
- All samples are processed the same
- All data features are treated the same
- Weights remain numerically stable

## Z-Transform

- Mean-center: subtract the average from each individual value
- Variance-normalize: divide by the standard deviation

![Image32](https://imgur.com/RLBtlgS.jpg "Image32")

## Min-max scaling

Min-max scaling transforms all the values to be between 0 and 1.

![Image33](https://imgur.com/YdSvDlo.jpg "Image33")

## Batch normalization

Batch normalization applies to the inputs of each layer, it's useful for deep networks or datasets with low accuracy. It acts as a regularizer because the input distributions are shifted and stabilized.

## Universal Approximation Theorem

A sufficiently wide or deep network can approximate any possible function.

A function is an input-output mapping which is the entire goal of training and using FNNs.

A DL network can solve any problem or perform any task provided that the task can be represented as an input-output function.

The combination of linear weighted combination and nonlinear activation function guarantees that any mapping from any space to any other space is, in theory, possible.

# Auto-Encoder

![Auto-encoder](https://imgur.com/paeXzmJ.jpg "Auto-encoder")

## CNN

There are three types of layers in a CNN:
- Convolution: learn filters to create feature maps
- Pooling: reduce dimensionality and increase receptive field size
- Fully connected: prediction

## Transfer learning

Transfer learning allows you to leverage existing models made by the community.

Use transfer learning when:
- Your problem is similar to a problem that someone else has solved
- The model was trained on a lot more data that what you have
- The model is deep

## Style transfer

With style transfer you take a content image, a style image and you create a target image.

Here is the style transfer algorithm:
- Import and freeze a pretrained CNN
- Import and transform images
- Make a trainable target image using random numbers
- Functions to compute feature maps and Gram matrices
- Extract target feature activation maps
- ContentMSE: target vs. content
- Compute Gram of target feature maps
- StyleMSE: of target Gram vs. style Gram
- Loss: contentMSE + styleMSE
- Backdrop on target image

## GANs

Discriminative models classify or characterize existing data.

Generative models create new samples.

The generator uses random data to create samples and the descriminator labels them as correct or incorrect. The generator keeps getting better.

There are two phases: training the descriminator and training the generator.

## RNNs

RNNs are a better model to deal with sequences.

Sequances contain sequantial information (temporal autocorrelation structure) that can be leveraged to predict the future or categorize the past.

## LSTM and GRU

LSTM stands for Long Short Term Memory.

GRU stands for Gated Recurrent Unit.