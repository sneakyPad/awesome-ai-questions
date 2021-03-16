<h1>6 XAI</h1>
<h1>Explainability</h1>
<ol start=299>
<li>
<p>What is the Saliency Map?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Saliency describes unique features in an image (Pixels, resolution, ..).  The purpose is to visualize the conspicuity of an image or in other words to differentiate the visual features of it.</li>
<li>This could be the color scale of an image, which is done by converting it to black and white, displaying a given temperature, or think of night vision (brightness is green, and darkness is black).</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled.png" width="50%"/></p>
<p>Could be used for traffic light detection. (Saliency detection)</p>
<p>Source: <a href="https://analyticsindiamag.com/what-are-saliency-maps-in-deep-learning/">https://analyticsindiamag.com/what-are-saliency-maps-in-deep-learning/</a></p>
</p></details></li>
<li>
<p>What is Integrated Gradients?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Attribution Algorithm. Works by specifying the target class.</li>
</ul>
</p></details></li>
<li>
<p>What is the difference betweeen CNN Heat Maps: Gradients vs. DeconvNets vs. Guided Backpropagation?</p><details><summary><b>Answer</b></summary><p>
<p><a href="https://glassboxmedicine.com/2019/10/06/cnn-heat-maps-gradients-vs-deconvnets-vs-guided-backpropagation/">CNN Heat Maps: Gradients vs. DeconvNets vs. Guided Backpropagation</a></p>
</p></details></li>
<li>
<p>What is NMF?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>A matrix factorisation technique (another one would be SVD). It stands for Non Negative Matrix Factorization and splits the user item matrix R into the matrices U and V. The optimization problem is that the multiplication of U and V should result in R.</li>
</ul>
</p></details></li>
<li>
<p>What is NCF (Neural Collaborative Filtering)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Models User Item interactions in the latent space with a NN</li>
<li>Is a generalization of Matrix Factorization</li>
<li>In the input layer, the user and item are one-hot encoded.</li>
<li>Neural CF Layers can be anything that is non linear</li>
</ul>
<p>They claim that the precendent dot product of matrix factorization limits the expressiveness of user and item latent vectors. (vector a might be closer to vector b than to c, although it is more similar to c.</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%201.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%201.png" width="50%"/></p>
<p>Neural CF Layer is replaced by a multiplication layer, which performs element-wise multiplication on the two inputs. Weights from multiplication layer to output is a identity matrix with a linear activation function.</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%202.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%202.png" width="50%"/></p>
<p>y_ui = predicted item of recommender matrix</p>
<p>L = Linear activation function</p>
<p>circle with dot = element-wise multiplicaton</p>
<p>p_u = latent vector user</p>
<p>q_I = latent vector item</p>
<p>Since J is an identity matrix it transforms the element-wise multiplication to a dot product. Because L is a linear function it means that the input would only be up or downscaled, but there is no real transformation. Hence, we can omit it. We end up with the same function as for Matrix Factorization, which shows, that NCF can be a generalization of Matrix Factorization. </p>
<p>To leverage the architecture of NCF we need non-linearity. It includes Multiple Layer Perceptron (MLP), Generalized Matrix Factorization (GMP) and a non linear activation function sigmoid.</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%203.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%203.png" width="50%"/></p>
<p>Source:</p>
<p><a href="https://towardsdatascience.com/paper-review-neural-collaborative-filtering-explanation-implementation-ea3e031b7f96">https://towardsdatascience.com/paper-review-neural-collaborative-filtering-explanation-implementation-ea3e031b7f96</a></p>
</p></details></li>
<li>
<p>What are the components of a NeuMF (Non-linear NCF)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>GMF (Generalized Matrix Factorization)</li>
<li>MLP (Multi Layer Perceptron)</li>
</ul>
</p></details></li>
<li>
<p>What is a Multilayer perceptron network (MLP)?</p><details><summary><b>Answer</b></summary><p>
<p>It is just a name for the simplest kind of deep neural network. </p>
<ul>
<li>There's at least one hidden layer</li>
<li>Each neuron has a non-linear activation function</li>
<li>Densely connected (each neuron in one layer is connected to every neuron the next or previous layer)</li>
<li>Uses as usual optimizer (e.g. Adam) and as a Loss function Binary cross-entroy for example</li>
<li>What is the Mean Reciprocal Rank (MRR)?</li>
</ul>
<p>MRR takes the single highest ranked positive/relevant item (since a system always returns an ordered list, it is the first positive solution). Here depicted as rank_i. It does not care about other relevant items (e.g. at position 4 or 20)</p>
<p>User MRR if:</p>
<ol>
<li>There is only one result</li>
<li>You care only about the highest ranked result (web search scenario)</li>
</ol>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%204.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%204.png" width="50%"/></p>
<p><strong>Example:</strong></p>
<p>```xml
Query: "Cities in California"</p>
<p>Ranked results: "Portland", "Sacramento", "Los Angeles"</p>
<p>Ranked results (binary relevance): [0, 1, 1]
Number of correct answers possible: 2
<strong>Reciprocal Rank</strong> = 1/2</p>
<p>Precision at 1: 0/1
Precision at 2: 1/2
Precision at 3: 2/3
<strong>Average precision</strong> = 1/m∗[1/2+2/3]=1/2∗[1/2+2/3]=0.38
```</p>
<p>Source: <a href="https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank">https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank</a></p>
</p></details></li>
<li>
<p>How does MRR differ to MAP?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>MAP considers all relevant items in the returned ranking. MAP and MRR are equivalent if MAP is set to k=1, since it then looks only at the first relevant item, and so does MRR.</p>
<p>Source:<a href="https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank">https://stats.stackexchange.com/questions/127041/mean-average-precision-vs-mean-reciprocal-rank</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the difference between implicit and explicit matrix factorization?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Implicit describes mostly unary ratings, such as that the user has seen an item or not (0 or 1)</li>
<li>Explicit desribes the actual given rating, such as 3.5 by a range from 0 to 5</li>
</ul>
</p></details></li>
<li>
<p>What is the median rank?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The position of the relevant item that is in the middle of the relevant item list (50%). If odd than it is one above.</li>
</ul>
<p><code>python
Example:
Relevant items position: [4, 60, 67, 77, 80]
Median rank: 67</code></p>
</p></details></li>
<li>
<p>What is Binary Cross-entropy?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>With binary cross entropy I can only classify two classes, whereas categorical cross entropy can classify many classes.</li>
</ul>
</p></details></li>
<li>
<p>What algorithms can be used for recommendation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Matrix Factorization (SVD</li>
<li>
<p>Nearest Neighbour (worse than MF)</p>
<ul>
<li>LSTMS, provide seen movies as a list of Ids to a LSTM network. Also, add tags to it.</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%205.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%205.png" width="50%"/></p>
<p>Source: <a href="https://medium.com/deep-systems/movix-ai-movie-recommendations-using-deep-learning-5903d6a31607">https://medium.com/deep-systems/movix-ai-movie-recommendations-using-deep-learning-5903d6a31607</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the Monte Carlo method?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>random sampling</li>
<li>On wikipedia it is described as follows:<ol>
<li>Define a domain of possible inputs</li>
<li>Generate inputs randomly from a <a href="https://en.wikipedia.org/wiki/Probability_distribution">probability distribution</a> over the domain</li>
<li>Perform a <a href="https://en.wikipedia.org/wiki/Deterministic_algorithm">deterministic</a> computation on the inputs</li>
<li>Aggregate the results</li>
</ol>
</li>
<li>
<p>Aggregating the result would be the mean for our P(x)</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%206.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%206.png" width="50%"/></p>
</li>
<li>
<p>Source: 
<a href="https://anotherdatum.com/vae.html">https://anotherdatum.com/vae.html</a>
<a href="https://en.wikipedia.org/wiki/Monte_Carlo_method">https://en.wikipedia.org/wiki/Monte_Carlo_method</a></p>
</li>
<li>What is the difference in the latent space of a variational autoencoder and a regular autoencoder?</li>
<li>AE learns just a compressed presentation of the input data under the restriction that it is able to reconstruct the input, whereas VAE assumes that the underlying input space has a probability distribution and the latent space itself is assumed as a probability distribution as well. (usually normal distribution). VAEs are penalized if the latent space distribution is divergent from the assumed prior (here normal distribution). Hence, VAEs are pushed torwards adjusting the distributions to the given prior distribution.</li>
<li>Source: <a href="https://www.quora.com/What-is-the-difference-in-the-latent-space-of-a-variational-autoencoder-and-a-regular-autoencoder#:~:text=AutoEncoders%20are%20free%20to%20have,is%20as%20good%20as%20mine">https://www.quora.com/What-is-the-difference-in-the-latent-space-of-a-variational-autoencoder-and-a-regular-autoencoder#:~:text=AutoEncoders are free to have,is as good as mine</a>.</li>
</ul>
</p></details></li>
</ol start=299>
<h2>XAI Methods</h2>
<ol start=312>
<li>
<p>What is a partial dependence plot and how does it work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Shows how the average prediction in your dataset changes when the j-th feature is changed</li>
<li>Requires to see the model as additive.</li>
<li>Computes the marginal effect on one or two features.</li>
</ul>
<p>Note:
Marginal effect means, what one feature adds to the prediction if they others are independent. In order to calculate it we have to average over those non-interesting</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%207.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%207.png" width="50%"/></p>
<p>Source:</p>
<p><a href="https://christophm.github.io/interpretable-ml-book/pdp.html">https://christophm.github.io/interpretable-ml-book/pdp.html</a></p>
<p>This means the following:
In order to make a statement about a variable and its partial dependency to the output we need to make the inferences with all possible values of the other features,
resulting Here in B3,B3, C2,C3. At the end we have n predicted outcomes, which need to be
averaged. This averaged y is the marginal contribution of A1.</p>
<p>Example:</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%208.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%208.png" width="50%"/></p>
<p>Source: <a href="https://towardsdatascience.com/introducing-pdpbox-2aa820afd312">https://towardsdatascience.com/introducing-pdpbox-2aa820afd312</a></p>
</p></details></li>
<li>
<p>What are advantages of PDP?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>perfect result, if features are independent</li>
<li>easy to implement</li>
<li>calculation is causal for model but not for the real world</li>
</ul>
</p></details></li>
<li>What are disadvantages of PDP?<details><summary><b>Answer</b></summary><ul>
<li>max feature = 2</li>
<li>Some plots do not show the distribution → misleading interpretation</li>
<li>Assumption of independence (weight ↔ height example) → Solution: Accumulated Local Effect plots (ALE) // works with conditional distribution</li>
<li>Heterogeneous effects might be hidden. Cancelling out large positive and negative predictions (→ resulting in zero, meaning no effect on the model) → Solution: Individual Conditional Expectation Curves ⇒ ICE-Curves</li>
</ul></details>
</li>
<li>What is Individual Conditional Expectation plot?<details><summary><b>Answer</b></summary><ul>
<li>Extension of Partidal Dependence Plot</li>
</ul></details>
</li>
<li>
<p>What is an additive model?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It is characterized that the effect of one variable does not depend on another (independence) and their contribution/effect adds up to the total effect/prediction. Therefore, we can assume to understand each variable individually (synonym marginally), and as a result, we are able to interpret the model.</li>
<li>Think of good old regression models. In a simple regression model, the effect of one variable does not depend on other variables and their effects add up to the total effect. If we can understand each variable individually or marginally, we would be able to interpret the entire model. However, this will no longer be true in the presence of interactions.</li>
</ul>
</p></details></li>
<li>
<p>What is the idea of Shapley Values?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Averaging over the contribution of all N! possible orderings. (Since non-linear models always depend on the order of the features)</li>
</ul>
</p></details></li>
<li>
<p>What is LAYER-WISE RELEVANCE PROPAGATION (LRP)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It is similar to Saliency Maps as it produces a heatmap, but LRP connects it to the neurons that contributed the most.</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%209.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%209.png" width="50%"/></p>
<p>Source: <a href="https://dbs.uni-leipzig.de/file/XAI_Alexandra_Nau.pdf">https://dbs.uni-leipzig.de/file/XAI_Alexandra_Nau.pdf</a></p>
</p></details></li>
<li>
<p>What is the idea of a regularizer?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Discourage the NN memorizing/overfitting</li>
</ul>
</p></details></li>
<li>
<p>What is the manifold space?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>A continuous, non-intersecting surface</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2010.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2010.png" width="50%"/></p>
</p></details></li>
<li>
<p>What is the null hypothesis?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>A null hypothesis is a hypothesis that says there is no statistical significance between the two variables in the hypothesis. It is the hypothesis that the researcher is trying to disprove. In the example, Susie's null hypothesis would be something like this: There is no statistically significant relationship between the type of water I feed the flowers and growth of the flowers. A researcher is challenged by the null hypothesis and usually wants to disprove it, to demonstrate that there is a statistically-significant relationship between the two variables in the hypothesis.
Source: <a href="https://study.com/academy/lesson/what-is-a-null-hypothesis-definition-examples.html">https://study.com/academy/lesson/what-is-a-null-hypothesis-definition-examples.html</a></li>
</ul>
</p></details></li>
<li>What Is an Alternative Hypothesis?<details><summary><b>Answer</b></summary><ul>
<li>An <strong>alternative hypothesis</strong> simply is the inverse, or opposite, of the null hypothesis. So, if we continue with the above example, the alternative hypothesis would be that there IS indeed a statistically-significant relationship between what type of water the flower plant is fed and growth. More specifically, here would be the null and alternative hypotheses for Susie's study:</li>
<li><strong>Null</strong>: If one plant is fed club soda for one month and another plant is fed plain water, there will be no difference in growth between the two plants.</li>
<li><strong>Alternative</strong>: If one plant is fed club soda for one month and another plant is fed plain water, the plant that is fed club soda will grow better than the plant that is fed plain water.</li>
<li>Source: <a href="https://study.com/academy/lesson/what-is-a-null-hypothesis-definition-examples.html">https://study.com/academy/lesson/what-is-a-null-hypothesis-definition-examples.html</a></li>
</ul></details>
</li>
<li>What is the Friedman's test?<details><summary><b>Answer</b></summary><ul>
<li>Similar to the parametric repeated measures ANOVA, it is used to detect differences in treatments across multiple test attempts. Essentially, you have n data records, e.g. participants and those participants have experienced different types of something, e.g. wine or drugs. Now, what you're gonna do is to assign a rank to each entry. The goal is to find out whether there is a difference among those populations or not. At the end you either reject or you don't the null hypothesis.</li>
<li>Source: <a href="https://www.statisticshowto.com/friedmans-test/">https://www.statisticshowto.com/friedmans-test/</a></li>
<li>Source: <a href="https://statistics.laerd.com/spss-tutorials/friedman-test-using-spss-statistics.php">https://statistics.laerd.com/spss-tutorials/friedman-test-using-spss-statistics.php</a></li>
</ul></details>
</li>
<li>
<p>What is the Lower Bound?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2011.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2011.png" width="50%"/></p>
<p>Source: <a href="https://www.mathsisfun.com/definitions/lower-bound.html">https://www.mathsisfun.com/definitions/lower-bound.html</a></p>
</p></details></li>
<li>
<p>What is Jensens inequality?</p><details><summary><b>Answer</b></summary><p>
<p>Linear transformation of a variable followed by another linear transformation leads to the same result as the other way around.</p>
<p>Example:</p>
<p><code>python
mean(f(x)) == f(mean(x)), for linear f()</code></p>
<p>Jensens inequality says:</p>
<p><strong><em>The intuition of linear mappings does not hold for nonlinear functions.</em></strong></p>
<p>Non linear functions are not a straight line when we think of the transformation on the input to the output. It is either convex (curving upward) or concarve (curving downwards).</p>
<p>Example:</p>
<p><code>python
(x) == x^2 is an exponential convex function.</code></p>
<p>I quote:</p>
<p><em>"Instead, the mean transform of an observation mean(f(x)) is always greater than the transform of the mean observation f(mean(x)), if the transform function is convex and the observations are not constant. We can state this as:"</em></p>
<p><code>python
mean(f(x)) &gt;= f(mean(x)), for convex f() and x is not a constant</code></p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2012.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2012.png" width="50%"/></p>
<p>Out intuition that the mean of a row of numbers multiplied by a non linear function (square), would be the same as applying the square to each of them and then calculate the mean, is wrong! The latter is greater.</p>
<ul>
<li>used to make claims about a function where little is known about the distribution</li>
<li>used to compare arithmetic mean and geometric mean</li>
</ul>
<p>Source: <a href="https://machinelearningmastery.com/a-gentle-introduction-to-jensens-inequality/">https://machinelearningmastery.com/a-gentle-introduction-to-jensens-inequality/</a></p>
</p></details></li>
<li>
<p>What is the geometric mean?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The Geometric Mean is a special type of average where we multiply the numbers together and then take a square root (for two numbers), cube root (for three numbers) etc.</li>
<li>For n numbers we take the nth root. (Nth root is concave)</li>
</ul>
</p></details></li>
<li>
<p>What is data scarcity?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>"Data scarcity, means too few data points often because it is difficult to get data or the data is small as compared to the amount needed."</p>
<p>Source: <a href="https://www.quora.com/What-is-the-difference-between-data-scarcity-and-data-sparsity#:~:text=Data%20scarcity%2C%20means%20too%20few,space%20between%20your%20data%20points">https://www.quora.com/What-is-the-difference-between-data-scarcity-and-data-sparsity#:~:text=Data scarcity%2C means too few,space between your data points</a>.</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is a Cox Model?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Quote: "A Cox model is a statistical technique for exploring the
relationship between the survival of a patient and several
explanatory variables. Survival analysis is concerned with studying the time
between entry to a study and a subsequent event (such as
death)."</p>
<p>Source: <a href="http://www.bandolier.org.uk/painres/download/whatis/COX_MODEL.pdf">http://www.bandolier.org.uk/painres/download/whatis/COX_MODEL.pdf</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>Can a linear model describe a curve?</p><details><summary><b>Answer</b></summary><p>
<p>Yes, that is possible. Usually a linear model only contains linear parameters and linear variables, but as long as the parameter remain linear and the variable to change is independent, it's exponent can be changed. For example can a independent variable be raised to quadratic, to fit a curve.</p>
<p>Rule for a linear model:</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2013.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2013.png" width="50%"/></p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2014.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2014.png" width="50%"/></p>
<p>Source: <a href="https://statisticsbyjim.com/regression/difference-between-linear-nonlinear-regression-models/">https://statisticsbyjim.com/regression/difference-between-linear-nonlinear-regression-models/</a></p>
</p></details></li>
<li>
<p>What are Restricted Boltzmann Machines?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Can be used for: dimensionality reduction, classification, regression, collaborative filtering, feature learning, and topic modeling.</li>
<li>The goal is to learn a compressed version of features (lower dimensional representation) that describe the input well. The concept is similar to AEs but in contrary they don't have a dedicated decoding part. The decoding flows back to the input and updates the weights directly instead of having additional weights. The flown back reconstruction is compared and the weights updated.</li>
<li>Therefore, we need two biases. One is for the forward pass (hidden bias) and the other for the backward pass (reconstruction). The input layer is considered as visible layer and the second layer as Hidden Layer.</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2015.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2015.png" width="50%"/></p>
<ul>
<li>RBM uses stochastic units with a particular distributions. Based on this we use Gibbs Sampling:</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2016.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2016.png" width="50%"/></p>
<ul>
<li>As an alternative to backpropagation and we use Contrastive Divergence. (I guess it is due to the probability distribution. VAE solves this problem by the reparameterization trick)</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2017.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2017.png" width="50%"/></p>
<p>Source: <a href="https://medium.com/edureka/restricted-boltzmann-machine-tutorial-991ae688c154">https://medium.com/edureka/restricted-boltzmann-machine-tutorial-991ae688c154</a></p>
<p>Additional Source: <a href="https://www.patrickhebron.com/learning-machines/week5.html">https://www.patrickhebron.com/learning-machines/week5.html</a></p>
</p></details></li>
<li>
<p>What is a Deep Belief Network?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Many layers of RBMs stacked together. For updating the weights, the Contrastive Divergense is still be used.</li>
</ul>
</p></details></li>
<li>What is synthetic data?<details><summary><b>Answer</b></summary><ul>
<li>data generated by an algorithm</li>
</ul></details>
</li>
<li>
<p>What is the inverse covariance matrix?</p><details><summary><b>Answer</b></summary><p>
<p><a href="https://www.quora.com/How-can-I-use-Gaussian-processes-to-perform-regression">https://www.quora.com/How-can-I-use-Gaussian-processes-to-perform-regression</a></p>
</p></details></li>
<li>
<p>What are explicit factor models?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Explicit factor models (EFM) [17] generate explanations based on the explicit features extracted from users’ reviews.</li>
</ul>
</p></details></li>
<li>
<p>What is Explainable Matrix Factorization (EMF)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>another algorithm that uses an explainability regularizer or soft constraint in the objective function of classical matrix factorization. The con- straining term tries to bring the user and explainable item’s latent factor vectors closer, thus favoring the appearance of explainable items at the top of the recommendation list.</li>
</ul>
<p>Source: An Explainable Autoencoder For Collaborative Filtering Recommendation (<a href="https://arxiv.org/abs/2001.04344">https://arxiv.org/abs/2001.04344</a>)</p>
</p></details></li>
<li>
<p>What is the optimal transport cost (OT)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It uses the Wasserstein distance to measure how much work has to be invested to transport one probability distribution to another. In other words: The cost to transform one distribution into the other.</li>
</ul>
</p></details></li>
<li>
<p>What is the infimum?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>That is the lower bound (not to be mixed up with the minimum).</li>
</ul>
</p></details></li>
<li>
<p>What is the difference between Wasserstein and Kullback-Leibler?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Wasserstein is a metric and KL not (KL is not symmetric and does not satisfy the triangle inequality)</li>
<li>
<p>Intuition:</p>
<ul>
<li>KL Divergence just tells you how similar two distributions are but Wasserstein distance gives you a measure for the effort to transport one probability mass to the other.</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2018.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2018.png" width="50%"/></p>
<p>Source: <a href="https://stats.stackexchange.com/questions/295617/what-is-the-advantages-of-wasserstein-metric-compared-to-kullback-leibler-diverg">https://stats.stackexchange.com/questions/295617/what-is-the-advantages-of-wasserstein-metric-compared-to-kullback-leibler-diverg</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the chi-squared test?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>used for categorical data</li>
<li>Does  one categorical variable affects another one?</li>
<li>calculates the p-value</li>
<li>In case of a low p-value:<ul>
<li>Result is thought of as being "significant" meaning we think the variables are not independent.</li>
</ul>
</li>
<li>chi square =  (Observed - Expected)^2 / Expected</li>
<li>Calculate the degree of freedom: Degree of Freedom = (rows − 1) × (columns − 1)</li>
<li>
<p>Look up p-value in pre-computed table by using chi-square and degree of freedom</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2019.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2019.png" width="50%"/></p>
</li>
<li>
<p>Source:
<a href="https://www.mathsisfun.com/data/chi-square-table.html">https://www.mathsisfun.com/data/chi-square-table.html</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is a CSR Matrix?</p><details><summary><b>Answer</b></summary><p>
<p>Compressed sparse row (CSR), Algorithm to compress the size of a matrix by indexing the 1s with row_id, col_id</p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2020.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2020.png" width="50%"/></p>
<p>Source: <a href="https://machinelearningmastery.com/sparse-matrices-for-machine-learning/">https://machinelearningmastery.com/sparse-matrices-for-machine-learning/</a></p>
</p></details></li>
<li>
<p>What is rocchio feedback?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Improving the search query by transforming the query and documents into a vector space and try to approach the centroid of the relevant documents while penalizing the distance to the centroid of the non-relevant documents.</li>
<li>Described as an equation:</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2021.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2021.png" width="50%"/></p>
<ul>
<li></li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2022.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2022.png" width="50%"/></p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2023.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2023.png" width="50%"/></p>
<p>Source and example calculation: <a href="https://www.coursera.org/lecture/text-retrieval/lesson-5-2-feedback-in-vector-space-model-rocchio-PyTkW">https://www.coursera.org/lecture/text-retrieval/lesson-5-2-feedback-in-vector-space-model-rocchio-PyTkW</a></p>
</p></details></li>
<li>
<p>What is a Weighted Matrix Factorisation?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2024.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2024.png" width="50%"/></p>
<ul>
<li>In contrast to SVD that has a cost function that processes all in one (seen and unseen), weighted MF sums over observed and unobserved items separately. SVD leads to poor generalisation when the corpus is large due to a sparse matrix.</li>
</ul>
<p><img alt="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2025.png" height="50%" src="data/6%20XAI%200fe6f8c58d1e489ba64035eab21d66e2/Untitled%2025.png" width="50%"/></p>
<ul>
<li>
<p>Matrix A is the interaction matrix.</p>
<p>Source: <a href="https://developers.google.com/machine-learning/recommendation/collaborative/matrix">https://developers.google.com/machine-learning/recommendation/collaborative/matrix</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>How can the objective function (cost) be minimized?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li><strong><a href="https://developers.google.com/machine-learning/crash-course/glossary#SGD">Stochastic gradient descent (SGD)</a></strong> is a generic method to minimize loss functions.</li>
<li>
<p><strong>Weighted Alternating Least Squares</strong> (<strong>WALS</strong>) is specialized to this particular objective.</p>
<p>Source: <a href="https://developers.google.com/machine-learning/recommendation/collaborative/matrix">https://developers.google.com/machine-learning/recommendation/collaborative/matrix</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What are the differences betweeen SGD and WALS?</p><details><summary><b>Answer</b></summary><p>
<p><strong>SGD</strong></p>
<p>+: <strong>Very flexible—can use other loss functions.</strong></p>
<p>+: <strong>Can be parallelized.</strong></p>
<p>-: <strong>Slower—does not converge as quickly.</strong></p>
<p>-: <strong>Harder to handle the unobserved entries (need to use negative sampling or gravity).</strong></p>
<p><strong>WALS</strong></p>
<p>-: <strong>Reliant on Loss Squares only.</strong></p>
<p>+: <strong>Can be parallelized.</strong></p>
<p>+: <strong>Converges faster than SGD.</strong></p>
<p>+: <strong>Easier to handle unobserved entries.</strong></p>
</p></details></li>
<li>
<p>What are advantages and disadvantages of Collaborative Filtering?</p><details><summary><b>Answer</b></summary><p>
<p>Taken from: <a href="https://developers.google.com/machine-learning/recommendation/collaborative/summary">https://developers.google.com/machine-learning/recommendation/collaborative/summary</a></p>
<p>Advantages:</p>
<p><strong>No domain knowledge necessary</strong></p>
<p>We don't need domain knowledge because the embeddings are automatically learned.</p>
<p><strong>Serendipity</strong></p>
<p>The model can help users discover new interests. In isolation, the ML system may not know the user is interested in a given item, but the model might still recommend it because similar users are interested in that item.</p>
<p><strong>Great starting point</strong></p>
<p>To some extent, the system needs only the feedback matrix to train a matrix factorization model. In particular, the system doesn't need contextual features. In practice, this can be used as one of multiple candidate generators.</p>
<p>Disadvantages:</p>
<p><strong>Cannot handle fresh items</strong></p>
<p>The prediction of the model for a given (user, item) pair is the dot product of the corresponding embeddings. So, if an item is not seen during training, the system can't create an embedding for it and can't query the model with this item. This issue is often called the <strong>cold-start problem</strong>. However, the following techniques can address the cold-start problem to some extent:</p>
<ul>
<li>
<p><strong>Projection in WALS.</strong> Given a new item i0 not seen in training, if the system has a few interactions with users, then the system can easily compute an embedding vi0 for this item without having to retrain the whole model. The system simply has to solve the following equation or the weighted version:</p>
<p>minvi0∈Rd‖Ai0−Uvi0‖</p>
<p>The preceding equation corresponds to one iteration in WALS: the user embeddings are kept fixed, and the system solves for the embedding of item i0. The same can be done for a new user.</p>
</li>
<li>
<p><strong>Heuristics to generate embeddings of fresh items.</strong> If the system does not have interactions, the system can approximate its embedding by averaging the embeddings of items from the same category, from the same uploader (in YouTube), and so on.</p>
</li>
</ul>
<p><strong>Hard to include side features for query/item</strong></p>
<p><strong>Side features</strong> are any features beyond the query or item ID. For movie recommendations, the side features might include country or age. Including available side features improves the quality of the model. Although it may not be easy to include side features in WALS, a generalization of WALS makes this possible.</p>
<p>To generalize WALS, <strong>augment the input matrix with features</strong> by defining a block matrix A¯, where:</p>
<ul>
<li>
<p>Block (0, 0) is the original feedback matrix .</p>
<p>A</p>
</li>
<li>
<p>Block (0, 1) is a multi-hot encoding of the user features.</p>
</li>
<li>Block (1, 0) is a multi-hot encoding of the item features.</li>
</ul>
</p></details></li>
</ol start=312>