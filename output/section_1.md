<h1>1 EN: ML Basics  1</h1>
<ol start=1>
<li>What is the output when using a convolution kernel?<details><summary><b>Answer</b></summary><ul>
<li>A real number</li>
</ul></details>
</li>
<li>How is the conv result calculated?<details><summary><b>Answer</b></summary><ul>
<li>Mean over the sum of the multiplications of image value with convolution kernel value</li>
</ul></details>
</li>
<li>What is a first level tensor called?<details><summary><b>Answer</b></summary><ul>
<li>Vector</li>
</ul></details>
</li>
<li>What is a second level tensor called?<details><summary><b>Answer</b></summary><ul>
<li>Matrix</li>
</ul></details>
</li>
<li>What is a third level tensor called?<details><summary><b>Answer</b></summary><ul>
<li>Tensor</li>
</ul></details>
</li>
<li>
<p>How can nominal data be transformed to numerical data?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>One Hot encoding</li>
</ul>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled.png" width="50%"/></p>
</p></details></li>
<li>
<p>What is meant by segmentation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Set of data belonging to an object of an object class c</li>
</ul>
</p></details></li>
<li>Is the Euclidean distance a measure of similarity or dissimilarity?<details><summary><b>Answer</b></summary><ul>
<li>Dissimilarity measure</li>
</ul></details>
</li>
<li>Is the correlation a similarity measure or dissimilarity measure?<details><summary><b>Answer</b></summary><ul>
<li>Similarity measure</li>
</ul></details>
</li>
<li>What is a Voronoi cell?<details><summary><b>Answer</b></summary><ul>
<li>Separation of the solution space into volumes (multi-dimensional) / surfaces (two-dimensional) using mid-perpendiculars between two adjacent points.</li>
</ul></details>
</li>
<li>
<p>How does the K-Neighbourhood Classifier work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>A hypersphere is enlarged by a new point in feature space until k elements are contained. Then the class is assigned by a simple majority decision.</li>
</ul>
</p></details></li>
<li>
<p>What is a Template?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Template objects are the mean of a certain class. This then results in the center of gravity of a cluster.</p>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%201.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%201.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the Minimum Distance Classifier?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>If vec_x closer to template 1, then class c_1, otherwise class c_2</li>
</ul>
</p></details></li>
<li>
<p>Perceptron: What is the idea of the weight vector?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>The intuition behind the weight vector comes from the idea of calculating the Euclidean distance from an input vector and a template vector. (See questions about this). The Euclidean distance of two vectors can be resolved to</p>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%202.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%202.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
<p>The middle part shows the weight vector and the +2 the bias. Since we are doing continuous learning, we can define a weight vector w and a random bias. This gives the Euclidean distance when we calculate the scalar product of the input vector with the weight vector. Since the Euclidean distance is a dissimilarity measure, if it is greater than 0, it is assigned a different class than if it is less than 0.</p>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%203.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%203.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
</li>
</ul>
</p></details></li>
<li>
<p>Perceptron: What is the idea of input vector and weight vector?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Calculate the Euclidean distance</li>
</ul>
</p></details></li>
<li>Perceptron: What is the idea of the perceptron?<details><summary><b>Answer</b></summary><ul>
<li>By means of continuous learning, we change the weight vector so that it creates a dividing line between two classes in our feature space.</li>
</ul></details>
</li>
<li>
<p>A linear decision function in 2D space may have how many input variables?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Only one, because the function value adds another axis</li>
</ul>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%204.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%204.png" width="50%"/></p>
<p>(Linear decision function)</p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
</p></details></li>
<li>
<p>Which is the required property for a cost function so that the minimum can be found?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Function must be continuous and differentiable</li>
</ul>
</p></details></li>
<li>
<p>What are the two options for the minimum search of the cost function?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Numerical (Full Scanning)</li>
<li>Analytical (derivation)</li>
<li>In the following example, the cost function is the sum of the quantity of incorrectly classified vec(x)</li>
</ul>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%205.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%205.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
</p></details></li>
<li>
<p>How can the distance of a point vec(x) to the decision line be calculated?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%206.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%206.png" width="50%"/></p>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%207.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%207.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
</p></details></li>
<li>
<p>How do I make the Perceptron cost function continuous and differentiable?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Use the distance of the incorrectly classified distances and add them up. The function becomes continuous, since we are now working with fine granularity, meaning that a slight change in w may produce a large variation in the result of the function value (hyperplane)</li>
</ul>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%208.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%208.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
</p></details></li>
<li>
<p>How are the weight values initialized?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Random!</li>
<li>Never use 0, otherwise the algorithm cannot start</li>
</ul>
</p></details></li>
<li>How do I update the weights of a perceptron?<details><summary><b>Answer</b></summary><ul>
<li>Weight = weight - learning rate * slope of the cost function from w to at position k (which is the derivative of the cost function over w)</li>
</ul></details>
</li>
<li>Multiclass extension of the perceptron: Which Loss do I choose?<details><summary><b>Answer</b></summary><ul>
<li>Hinge-Loss or Maximum Margin Loss</li>
</ul></details>
</li>
<li>
<p>How does the Hinge Loss work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>For each class the score is calculated and the goal is that the score for the correct class is greater than a certain delta + the accumulated sum of all other classes. If so, the cost would be 0, if in the delta, costs are calculated proportionally.</p>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%209.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%209.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the composition of the Multiclass SVM Loss?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Composed of hinge loss and regularization loss (with lambda as regularization parameter)</li>
</ul>
</p></details></li>
<li>When do I use the loss and when a score?<details><summary><b>Answer</b></summary><ul>
<li>Loss value implies how well or poorly a certain model behaves after each iteration of optimization. Ideally, one would expect the reduction of loss after each, or several, iteration(s).</li>
<li>The accuracy of a model is usually determined after the model parameters are learned and fixed and no learning is taking place. Then the test samples are fed to the model and the number of mistakes (zero-one loss) the model makes are recorded, after comparison to the true targets.</li>
<li>Source: <a href="https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model">https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model</a></li>
</ul></details>
</li>
<li>What is a weakness of the template approach?<details><summary><b>Answer</b></summary><ul>
<li>Template becomes blurry if class instance has a very different feature value</li>
</ul></details>
</li>
<li>How can the problem be solved for a template?<details><summary><b>Answer</b></summary><ul>
<li>Transformation of the actual features to so-called intermediate features, which are invariant to differences in original features. (non-linear transformation). This results in an intermediate feature template</li>
</ul></details>
</li>
<li>
<p>What is the structure of a non-linear classifier that addresses the problem of intermediate feature templates?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%2010.png" height="50%" src="data/1%20EN%20ML%20Basics%201%20c4761834b24c465c9eeabd9dcf126f86/Untitled%2010.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Prof. Dr. Norbert Link</p>
</p></details></li>
<li>
<p>How can non-linearity be generated?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>With AND, OR and XOR Grids</li>
</ul>
</p></details></li>
<li>What is the advantage of the templates in terms of inference?<details><summary><b>Answer</b></summary><ul>
<li>They are faster because only Voinoi must be calculated for representatives</li>
</ul></details>
</li>
<li>What is the disadvantage of the templates ?<details><summary><b>Answer</b></summary><ul>
<li>They are inaccurate</li>
</ul></details>
</li>
<li>How do I get the hyperplane of a perceptron?<details><summary><b>Answer</b></summary><ul>
<li>Set zero for one of the two values (x1,x2). Solve the equation and do the same the other way round.</li>
</ul></details>
</li>
<li>What is the principle of the minimum distance classifier?<details><summary><b>Answer</b></summary><ul>
<li>Calculate a center of gravity vector from the sample vectors of a class (mean over sum). This is also called a template.</li>
<li>Classification based on distance to template</li>
</ul></details>
</li>
<li>What is the advantage of the minimum distance classifier compared to the nearest neighbour classifier?<details><summary><b>Answer</b></summary><ul>
<li>The Minimum Distance Classifier is more robust against outliers, because not every data point of the sample is considered, but only their main class focuses.</li>
<li>The inference should be faster, since only the distance to N class centers is determined
and not the distance to each individual data point of the sample.</li>
<li>The distances must be calculated only for the centers of gravity</li>
</ul></details>
</li>
<li>What is the disadvantage of the minimum distance classifier compared to the nearest neighbour classifier?<details><summary><b>Answer</b></summary><ul>
<li>Only works under certain conditions</li>
<li>Mean values are only a good template in special cases</li>
<li>Only linear class boundaries can be displayed</li>
</ul></details>
</li>
<li>How does the perceptron divide a two-dimensional feature space?<details><summary><b>Answer</b></summary><ul>
<li>Through a straight line into two half spaces</li>
</ul></details>
</li>
<li>What are the parameters of the perceptrons and on which principle does the perceptron learn its values?<details><summary><b>Answer</b></summary><ul>
<li>Parameters: Weight values 𝑤 Threshold values 𝑤</li>
<li>Principle: Minimization of the classification error-loss function by using a gradient descent method</li>
</ul></details>
</li>
<li>What does the perceptron cost function represent and what is it ?<details><summary><b>Answer</b></summary><ul>
<li>Percepton cost function represents the sum of the distances of all incorrectly classified samples to the decision (hyper) level.</li>
</ul></details>
</li>
<li>How can the limitation of linear classifiers be overcome?<details><summary><b>Answer</b></summary><ul>
<li>Combination of linear classifiers in layers with non-linear activation function</li>
</ul></details>
</li>
</ol start=1>