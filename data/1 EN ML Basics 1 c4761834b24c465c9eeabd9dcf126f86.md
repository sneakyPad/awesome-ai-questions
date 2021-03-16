# 1 EN: ML Basics  1

- What is the output when using a convolution kernel?
    - A real number
- How is the conv result calculated?
    - Mean over the sum of the multiplications of image value with convolution kernel value
- What is a first level tensor called?
    - Vector
- What is a second level tensor called?
    - Matrix
- What is a third level tensor called?
    - Tensor
- How can nominal data be transformed to numerical data?
    - One Hot encoding

    ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled.png)

- What is meant by segmentation?
    - Set of data belonging to an object of an object class c
- Is the Euclidean distance a measure of similarity or dissimilarity?
    - Dissimilarity measure
- Is the correlation a similarity measure or dissimilarity measure?
    - Similarity measure
- What is a Voronoi cell?
    - Separation of the solution space into volumes (multi-dimensional) / surfaces (two-dimensional) using mid-perpendiculars between two adjacent points.
- How does the K-Neighbourhood Classifier work?
    - A hypersphere is enlarged by a new point in feature space until k elements are contained. Then the class is assigned by a simple majority decision.

- What is a Template?
    - Template objects are the mean of a certain class. This then results in the center of gravity of a cluster.

        ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%201.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%201.png)

        Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

- What is the Minimum Distance Classifier?
    - If vec_x closer to template 1, then class c_1, otherwise class c_2
- Perceptron: What is the idea of the weight vector?
    - The intuition behind the weight vector comes from the idea of calculating the Euclidean distance from an input vector and a template vector. (See questions about this). The Euclidean distance of two vectors can be resolved to

        ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%202.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%202.png)

        Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

        The middle part shows the weight vector and the +2 the bias. Since we are doing continuous learning, we can define a weight vector w and a random bias. This gives the Euclidean distance when we calculate the scalar product of the input vector with the weight vector. Since the Euclidean distance is a dissimilarity measure, if it is greater than 0, it is assigned a different class than if it is less than 0.

        ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%203.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%203.png)

        Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

- Perceptron: What is the idea of input vector and weight vector?
    - Calculate the Euclidean distance
- Perceptron: What is the idea of the perceptron?
    - By means of continuous learning, we change the weight vector so that it creates a dividing line between two classes in our feature space.
- A linear decision function in 2D space may have how many input variables?
    - Only one, because the function value adds another axis

    ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%204.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%204.png)

    (Linear decision function)

    Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

- Which is the required property for a cost function so that the minimum can be found?
    - Function must be continuous and differentiable
- What are the two options for the minimum search of the cost function?
    - Numerical (Full Scanning)
    - Analytical (derivation)
    - In the following example, the cost function is the sum of the quantity of incorrectly classified vec(x)

    ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%205.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%205.png)

    Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

- How can the distance of a point vec(x) to the decision line be calculated?

    ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%206.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%206.png)

    ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%207.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%207.png)

    Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

- How do I make the Perceptron cost function continuous and differentiable?
    - Use the distance of the incorrectly classified distances and add them up. The function becomes continuous, since we are now working with fine granularity, meaning that a slight change in w may produce a large variation in the result of the function value (hyperplane)

    ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%208.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%208.png)

    Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

- How are the weight values initialized?
    - Random!
    - Never use 0, otherwise the algorithm cannot start
- How do I update the weights of a perceptron?
    - Weight = weight - learning rate * slope of the cost function from w to at position k (which is the derivative of the cost function over w)
- Multiclass extension of the perceptron: Which Loss do I choose?
    - Hinge-Loss or Maximum Margin Loss
- How does the Hinge Loss work?
    - For each class the score is calculated and the goal is that the score for the correct class is greater than a certain delta + the accumulated sum of all other classes. If so, the cost would be 0, if in the delta, costs are calculated proportionally.

        ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%209.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%209.png)

- What is the composition of the Multiclass SVM Loss?
    - Composed of hinge loss and regularization loss (with lambda as regularization parameter)
- When do I use the loss and when a score?
    - Loss value implies how well or poorly a certain model behaves after each iteration of optimization. Ideally, one would expect the reduction of loss after each, or several, iteration(s).
    - The accuracy of a model is usually determined after the model parameters are learned and fixed and no learning is taking place. Then the test samples are fed to the model and the number of mistakes (zero-one loss) the model makes are recorded, after comparison to the true targets.
    - Source: [https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model](https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model)
- What is a weakness of the template approach?
    - Template becomes blurry if class instance has a very different feature value
- How can the problem be solved for a template?
    - Transformation of the actual features to so-called intermediate features, which are invariant to differences in original features. (non-linear transformation). This results in an intermediate feature template
- What is the structure of a non-linear classifier that addresses the problem of intermediate feature templates?

    ![1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%2010.png](1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%2010.png)

    Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link

- How can non-linearity be generated?
    - With AND, OR and XOR Grids
- What is the advantage of the templates in terms of inference?
    - They are faster because only Voinoi must be calculated for representatives
- What is the disadvantage of the templates ?
    - They are inaccurate
- How do I get the hyperplane of a perceptron?
    - Set zero for one of the two values (x1,x2). Solve the equation and do the same the other way round.
- What is the principle of the minimum distance classifier?
    - Calculate a center of gravity vector from the sample vectors of a class (mean over sum). This is also called a template.
    - Classification based on distance to template
- What is the advantage of the minimum distance classifier compared to the nearest neighbour classifier?
    - The Minimum Distance Classifier is more robust against outliers, because not every data point of the sample is considered, but only their main class focuses.
    - The inference should be faster, since only the distance to N class centers is determined
    and not the distance to each individual data point of the sample.
    - The distances must be calculated only for the centers of gravity
- What is the disadvantage of the minimum distance classifier compared to the nearest neighbour classifier?
    - Only works under certain conditions
    - Mean values are only a good template in special cases
    - Only linear class boundaries can be displayed
- How does the perceptron divide a two-dimensional feature space?
    - Through a straight line into two half spaces
- What are the parameters of the perceptrons and on which principle does the perceptron learn its values?
    - Parameters: Weight values 𝑤 Threshold values 𝑤
    - Principle: Minimization of the classification error-loss function by using a gradient descent method
- What does the perceptron cost function represent and what is it ?
    - Percepton cost function represents the sum of the distances of all incorrectly classified samples to the decision (hyper) level.
- How can the limitation of linear classifiers be overcome?
    - Combination of linear classifiers in layers with non-linear activation function