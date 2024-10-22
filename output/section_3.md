<h1>5 EN: AI - Computer Vision, NLP, Reinforcement Learning</h1>
<h2>Computer Vision</h2>
<ol start=110>
<li>What are difficulties about classifying an image<details><summary><b>Answer</b></summary><ul>
<li>Intra-class variation (Different breed of dogs)</li>
<li>Shape Variation (Size of dogs)</li>
<li>Illumination variation</li>
</ul></details>
</li>
<li>
<p>What kind of feature did people use before 2012?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Color Histogram</li>
<li>
<p>Key Point descriptor</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled.png" width="50%"/></p>
</li>
<li>
<p>Histogram of gradients</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%201.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%201.png" width="50%"/></p>
</li>
<li>
<p>Bag of words model</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%202.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%202.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>Name popular benchmark datasets</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>MNIST</li>
<li>CIFAR-10 (Canadian Institute for Advanced Research)</li>
<li>IMAGENET (22 Categories)</li>
</ul>
</p></details></li>
<li>Why haven't been CNNs used before?<details><summary><b>Answer</b></summary><ul>
<li>Requires huge datasets</li>
<li>and immense computational power</li>
</ul></details>
</li>
<li>Name the most popular loss functions for regression and classification<details><summary><b>Answer</b></summary><ul>
<li>Regression: L2 loss (aka mean squared error), L1 loss (aka mean absolute error)</li>
<li>Classification: cross-entropy loss, hinge loss (used in SVMs)</li>
</ul></details>
</li>
<li>
<p>Explain cross-entropy loss</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%203.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%203.png" width="50%"/></p>
<ul>
<li>M - number of classes</li>
<li>y_i,c - binary indicator (0 or 1) if class label c is the correct classification for observation o</li>
<li>(p_i,c) - the predicted probability that observation i is of class c</li>
</ul>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 </p>
<p>Actually the log function would be mirrored on the x axis and a small value of the predicted probability would be &lt; 0, but since our goal is to minimize it we have to turn the function around and we do that by multiplying -1.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%204.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%204.png" width="50%"/></p>
<p>Source: <a href="https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html">ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html</a></p>
<p>Therefore, eventually the network learns to output high probabilities for the correct class and low probabilities for the incorrect ones.</p>
</p></details></li>
<li>
<p>What is a dense layer?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>A dense layer (also called a fully connected layer) is a layer where each input is connected to every output, i.e. what we modelled</li>
</ul>
</p></details></li>
<li>
<p>What is the standard activation function to use?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%205.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%205.png" width="50%"/></p>
<p>Source: <a href="https://cdn-images-1.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png">https://cdn-images-1.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png</a></p>
</p></details></li>
<li>
<p>What is the output of the conv filter when it slides over an image?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Its dot product</li>
</ul>
</p></details></li>
<li>
<p>What does a pooling layer?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Used for dimensionality reduction</li>
<li>Slide with a stride &gt; 1</li>
<li>
<p>Instead of a dot product of a conv filter the ouput is the max (there are others like average)</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%206.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%206.png" width="50%"/></p>
<p>Source Illustration: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>Impression of a ConvNet from the inside</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%207.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%207.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 </p>
</p></details></li>
<li>
<p>What kind of level feature does a ConvNet encode in the beginning and at the end?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The early layers encode most basic feature, whereas deeper layers pick up more specfific ones (e.g. eyes of a face)</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%208.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%208.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 </p>
</p></details></li>
<li>
<p>How did they improve neural networks further?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Smaller conv filter but deeper networks</li>
</ul>
</p></details></li>
<li>
<p>What is the idea of Inception Modules?</p><details><summary><b>Answer</b></summary><p>
<p>Create a good local network in a module and stack the modules on top of each other.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%209.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%209.png" width="50%"/></p>
<p>Source Illustration : Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</p></details></li>
<li>
<p>How do Inception Modules work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Different techniques are applied independently and concatenated afterwards. It is not clear in the beginning which of the approaches will work, but one of these just might.</li>
</ul>
</p></details></li>
<li>
<p>What is the problem of Inception Modules?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It creates a huge computational complexity due to a huge increase of the ouput depth</li>
<li>
<p>Example:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2010.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2010.png" width="50%"/></p>
<p>Source Illustration : Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>How can the problem of Inception Modules be solved?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>By applying 1x1 convolutions at the right position, such as prior to a 5x5 or as successor of a 3x3 max pooling</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2011.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2011.png" width="50%"/></p>
<p>Source Illustration : Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
<p>Applying 1x1, leads to:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2012.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2012.png" width="50%"/></p>
<p>Source Illustration : Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</p></details></li>
<li>
<p>What is the goal/purpose of a 1x1 conv filter?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It reduces the depth by combining feature maps</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2013.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2013.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</p></details></li>
<li>
<p>What are the three parts of the GoogleNet?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Stem Network</li>
<li>Stacked Inception Modules</li>
<li>Classifier Outputs</li>
</ul>
</p></details></li>
<li>
<p>What is the idea of Deep Residual Learning?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Learn difference to the identity map instead of map itself</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2014.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2014.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is Transfer Learning?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Transfer learning focuses on storing knowledge gained for a specific problem and applying it to a different but related problem”</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2015.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2015.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>What different challenges are solvable with computer vision?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Object localization: output the class of an image (object) but also output the position of the object (e.g. in form of a bounding box)</li>
<li>
<p>Object detection: classify multiple objects on an image as well as their locations (bounding boxes)</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2016.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2016.png" width="50%"/></p>
</li>
<li>
<p>Semantic segmentation: label the individual pixels that belong to different categories (don’t differentiate between instances). Widely used in self-driving cars.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2017.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2017.png" width="50%"/></p>
</li>
<li>
<p>Instance segmentation: detect multiple objects, then label the pixels belonging to each of the detected objects. In contrary to Semantic segmentation: Differentiate between instances!</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2018.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2018.png" width="50%"/></p>
<p>Source Illustration: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the idea for object localization?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>We provide the class we are looking for and the NN tries to locate it. We still need to train a classifier for scoring the different classes, because there could be more than one class in the picture.  If the picture showed a dog and cat, and we specify the class as cat, we would only get the bounding boxes for the cat. (or all cats)</li>
<li>
<p>train a regression output for the box coordinates</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2019.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2019.png" width="50%"/></p>
<p>Source Illustration: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>What are the different loss functions for object localization?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Find every object and its bounding boxes in the image.</li>
<li>softmax loss for classification</li>
<li>
<p>L2 loss (MSE) for bounding boxes</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2020.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2020.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is pose estimation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The human body gets split up in different parts, such as left hand, head, right foot. The NN tries to figure out these parts in the image.</li>
</ul>
</p></details></li>
<li>
<p>What's the ouput of a NN doing pose estimation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>A vector holding different parts and their coordinates.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2021.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2021.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the loss of pose estimation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>L2 loss</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2022.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2022.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>What are some approaches to do object detection (which is why harder than object localization?)</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>We do not know beforehand how many object there will be</li>
<li>Therefore, we cannot use the same approach as in localization (where only one object was to be found and classified).</li>
<li>
<p>A brute-force approach: apply a CNN to many different subsets of an image and classify each as an object or background.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2023.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2023.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>Why is the brute force approach bad?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>checking every crop is computationally costly</li>
</ul>
</p></details></li>
<li>
<p>What are other algorithms to do object detection?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Selective Search for Object Recognition, which does a region proposal for, for example, 2000 parts of the image. The regions are then passed to a normal cnn. This cnn is called Region-CNN (R-CNN)</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2024.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2024.png" width="50%"/></p>
<p>Source: <a href="https://cdn-images-1.medium.com/max/1000/1*REPHY47zAyzgbNKC6zlvBQ.png">https://cdn-images-1.medium.com/max/1000/1*REPHY47zAyzgbNKC6zlvBQ.png</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>R-CNNs are slow, thats why Fast R-CNN were born. What's the difference?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Fast R-CNN make use of the feature map that is created when an image is fed to the NN. Based on this feature map, regions are derived, they are wrapped into identical size by using pooling operations and then passed to a fully connected layer.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2025.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2025.png" width="50%"/></p>
<p>Source: <a href="https://cdn-images-1.medium.com/max/1000/1*0pMP3aY8blSpva5tvWbnKA.png">https://cdn-images-1.medium.com/max/1000/1*0pMP3aY8blSpva5tvWbnKA.png</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What does Fast R-CNN do?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Uses a predefined algorithm for creating region proposals, and a Region Proposal Network (RPN) to predict proposals from features.</li>
</ul>
</p></details></li>
<li>What are MobileNets and what do they use?<details><summary><b>Answer</b></summary><ul>
<li>On mobile we require nn to process in real time, therefore the number of computations has to be reduced. The engine for this is called depthwise separable convolutions.</li>
</ul></details>
</li>
<li>
<p>Explain the idea of Depthwise Separable Convolutions</p><details><summary><b>Answer</b></summary><p>
<p>Key to understand is to know that the dot product calculation is expensive.</p>
<ul>
<li>
<p>A standard convolution does a dot product of all channels</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2026.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2026.png" width="50%"/></p>
<p>Source: <a href="https://cdn-images-1.medium.com/max/800/0*rbWRzjKvoGt9W3Mf.png">https://cdn-images-1.medium.com/max/800/0*rbWRzjKvoGt9W3Mf.png</a></p>
</li>
<li>
<p>The idea of depthwise is to reduce the dimension of the channels before the dot product is applied.</p>
<p>1.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2027.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2027.png" width="50%"/></p>
<p>Source: <a href="https://machinethink.net/images/mobilenets/DepthwiseConvolution@2x.png">https://machinethink.net/images/mobilenets/DepthwiseConvolution@2x.png</a></p>
<p>2.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2028.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2028.png" width="50%"/></p>
<p>Source: <a href="https://machinethink.net/images/mobilenets/PointwiseConvolution@2x.png">https://machinethink.net/images/mobilenets/PointwiseConvolution@2x.png</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What impact do the two parameters have on MobileNets?</p><details><summary><b>Answer</b></summary><p>
<p>The parameters can be used to decrease latency with the drawback to sacrifice accuracy.</p>
<ul>
<li>Width multiplier: How thin or thick the feature map gets. (Must be dimensions going into the back)</li>
<li>Resolution multiplier: Parameter for the size of the resolution of input and feature map.</li>
<li>Image showing the dropping accuracy with less computation steps</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2029.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2029.png" width="50%"/></p>
<p>Source: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, A. Howard et al.</p>
<p>Figure 5 shows the trade off between ImageNet Accuracy and number of parameters for the 16 models made from the cross product of width multiplier α∈{1,0.75,0.5,0.25} and resolutions {224,192,160,128}</p>
</p></details></li>
<li>
<p>What's the difference between semantic segmentation and instance segmentation?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2030.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2030.png" width="50%"/></p>
<p>Source: <a href="http://vladlen.info/wp-content/uploads/FSO-1.jpg">http://vladlen.info/wp-content/uploads/FSO-1.jpg</a></p>
</p></details></li>
<li>
<p>What is Panoptic Segmentation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Combines segmenting  stuff (background regions, e.g. grass, sky) and Things (Objects with well defined shape, e.g. person, car). Therefore it combines Semantic and Instance Segmentation</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2031.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2031.png" width="50%"/></p>
<p>Source: Panoptic Segmentation A. Kirillov et al.</p>
</li>
</ul>
</p></details></li>
<li>
<p>How does Panoptic Segmentation work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Algorithm must label every pixel in the found objects but also label every pixel in the background "stuff". We map every pixel i to a tupel:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2032.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2032.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 </p>
</li>
</ul>
</p></details></li>
<li>
<p>Name datasets to train Semantic or Instance Segmentation models</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>COCO (Common Objects in Context)</li>
<li>BDD100K (A Large-scale Diverse Driving Video Database)</li>
<li>Mapillary Vistas Dataset (Data from cities all over the world)</li>
<li>Cityscapes (50 cities (49 german cities + Zurich), various seasons (spring, summer, fall))</li>
<li>The KITTI Vision Benchmark Suite</li>
</ul>
</p></details></li>
<li>
<p>What is the Loss Function for Semantic Segmentation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Cross-entropy on a per-pixel basis</li>
<li>
<p>Main idea: evaluate prediction loss for every individual pixel in an image and then take the average of these values for a per-image loss. This is called Pixel-Wise Cross Entropy Loss.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2033.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2033.png" width="50%"/></p>
<p>Source: <a href="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-24-at-10.46.16-PM.png">https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-24-at-10.46.16-PM.png</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the problem for the Loss Function for Semantic Segmentation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Here: every pixel is treated equally (because of the averaging). This can lead to some issues related to imbalanced classes.</li>
</ul>
</p></details></li>
<li>
<p>How can you evaluate the performance of semantic segmentation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>For image segmentation we have Intersection
over Union (IoU) metric.</li>
<li>
<p>IoU measures what percentage of pixels in
target and prediction images overlap.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2034.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2034.png" width="50%"/></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2035.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2035.png" width="50%"/></p>
<p>Source: <a href="https://www.jeremyjordan.me/evaluating-image-segmentation-models/">https://www.jeremyjordan.me/evaluating-image-segmentation-models/</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What approaches exist for Semantic Segmentation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Sliding Windows (Inefficient and unusable in practice)</li>
<li>Fully Convolutional Network for Segmentation (still inefficent, because no dimensionality reduction is applied) //Recall: When a CNN gets deeper we want to have more features but of lower dimension.</li>
<li>
<p>Image Segmentation Network</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2036.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2036.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>How can we incorporate dimensionality reduction to a CNN for Semantic Segmentation but still have the output of the original size?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Design network with downsampling and upsampling (same principal as autoencoder)</li>
</ul>
</p></details></li>
<li>
<p>How can downsampling be done?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Pooling or Strided Convolutions</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2037.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2037.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019</p>
</li>
</ul>
</p></details></li>
<li>
<p>How can upsampling be done?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Unpooling (parameter-free), e.g. "max-unpooling"</li>
<li>Transpose Convolution (trainable upsampling)</li>
</ul>
</p></details></li>
<li>
<p>What types of Unpooling do you know?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>1:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2038.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2038.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
<li>
<p><strong>Bed of Nails</strong> (Fill rest of the squares with zeroes)</p>
</li>
<li>
<p><strong>Nearest Neighbour</strong> Unpooling. A KIND OF INVERSION OF AVERAGE POOLING OPERATION (Fill rest of the squares with the same number)</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2039.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2039.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
<li>
<p><strong>Max Unpooling as Max Pooling Inverse</strong></p>
<ul>
<li>
<p>We remember which element was the max value and recover this in the new output</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2040.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2040.png" width="50%"/></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2041.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2041.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</li>
</ul>
</p></details></li>
<li>
<p>Why is the value of the unpooled output slighlty different than it was before?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>For this a transpose convolution is used and the scalar of the reduced output is multiplied by the filter. Important: The values in the filter are trained by the network and not predefined.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2042.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2042.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>How does the checkerboard artifact evolve?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2043.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2043.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</p></details></li>
<li>
<p>Convolution and its transpose side-by-side</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2044.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2044.png" width="50%"/></p>
<p>Source: <a href="https://github.com/vdumoulin/conv_arithmetic">https://github.com/vdumoulin/conv_arithmetic</a></p>
</p></details></li>
<li>
<p>What are other names for transpose convolution?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Upconvolution</li>
<li>Backward strided convolution</li>
<li>Fractionally strided convolution</li>
<li>Deconvolution - this one is particularly dangerous because deconvolution is a well-defined mathematical operation which <strong>is not the same</strong> as transpose convolution.</li>
</ul>
</p></details></li>
<li>
<p>What is the architecture idea for Instance segmentation?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>There are two independent branches trying to detect a) the categories and the bounding boxes and b) the second branch classifies each pixel whether its an object or not (also called mask prediction).</li>
<li>
<p>⇒ This procedure happens due to performance reason to the region proposals.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2045.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2045.png" width="50%"/></p>
<p>Source: <a href="https://github.com/vdumoulin/conv_arithmetic">https://github.com/vdumoulin/conv_arithmetic</a></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2046.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2046.png" width="50%"/></p>
<p>Source: <a href="https://cdn-images-1.medium.com/max/2000/1*lMEd6AcDmpH0mDzBHyiERw.png">https://cdn-images-1.medium.com/max/2000/1*lMEd6AcDmpH0mDzBHyiERw.png</a></p>
</li>
<li>
<p>Note: The idea of mask prediction can also be used for Pose Detection?</p>
</li>
<li>Give a summary of Encoder-Decoder Networks</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2047.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2047.png" width="50%"/></p>
<p>Source: <a href="https://saytosid.github.io/images/segnet/Complete%20architecture.png">https://saytosid.github.io/images/segnet/Complete architecture.png</a></p>
</p></details></li>
<li>
<p>What is the motivation for Autoencoders?</p><details><summary><b>Answer</b></summary><p>
<p>We want them to learn efficient and pertinent data encodings of the data.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2048.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2048.png" width="50%"/></p>
<p>Source: <a href="https://cdn-images-1.medium.com/max/1600/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png">https://cdn-images-1.medium.com/max/1600/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png</a></p>
</p></details></li>
<li>
<p>Give a definition of Autoencoders</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Autoencoder is a type of an encoder-decoder network where target space is the same as the input space.</li>
</ul>
</p></details></li>
<li>Provide Characteristics of Autoencoders<details><summary><b>Answer</b></summary><ul>
<li>Data does not have to be labelled</li>
<li>Latent feature space is usually of lower dimension than the input feature space</li>
<li>L2 as loss function</li>
</ul></details>
</li>
<li>
<p>Where are autoencoders used?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>All applications of AEs utilize the learned latent features for further purposes.</li>
<li>Use for dimensionality reduction</li>
<li>Important: AEs are quite data-specific. They will only work well on data that is similar to that on which they were trained. Extreme example: an AE trained on images will not work well on time-series data.</li>
<li>Not really good for compression of data</li>
<li>
<p>Denoising of Input data</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2049.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2049.png" width="50%"/></p>
<p>Source: <a href="https://cdn-images-1.medium.com/max/1600/1*G0V4dz4RKTKGpebeoSWB0A.png">https://cdn-images-1.medium.com/max/1600/1*G0V4dz4RKTKGpebeoSWB0A.png</a></p>
</li>
<li>
<p>Watermark Removal</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2050.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2050.png" width="50%"/></p>
<p>Source: <a href="https://www.edureka.co/blog/autoencoders-tutorial/">https://www.edureka.co/blog/autoencoders-tutorial/</a></p>
</li>
<li>
<p>Image Coloring</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2051.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2051.png" width="50%"/></p>
<p>Source: <a href="https://www.edureka.co/blog/autoencoders-tutorial/">https://www.edureka.co/blog/autoencoders-tutorial/</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>How could we initialize our neural network quite well?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>By passing the features of the autoencoder to the weight initialization</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2052.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2052.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019</p>
</li>
</ul>
</p></details></li>
<li>
<p>How can Autoencoders be used for Anomaly detection?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Main idea: let’s say we have a well-trained autoencoder (which means it can reconstruct the data it was trained on without too much error):<ul>
<li>If it works well on the new input, i.e. reconstruction error is low, we can assume that the input is of the normal class.</li>
<li>If the reconstruction error is high, we should think that the input is of an ‘unseen’ class, i.e. it is an anomaly.</li>
</ul>
</li>
</ul>
</p></details></li>
<li>Potential applications for Autoencoders<details><summary><b>Answer</b></summary><ul>
<li>Detect if input data is of the class that our model was trained on or not.</li>
<li>Detect outliers.</li>
<li>Fraud analytics.</li>
<li>Monitoring sensor data</li>
</ul></details>
</li>
<li>
<p>Give an idea of Generative Models</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>It’s not about predicting and classifying things, it’s</p>
<p>about learning some underlying hidden structure of the
data at hand.</p>
<p>We want to train a model distribution pθ ,parameterized by our choice, that would fit or resemble pdata.</p>
<p>If we can successfully obtain this trained model pθ, it in fact means that we know the underlying structure of the data and we can do a lot of interesting things with it.</p>
<p>→ We start by an empirical data distribution</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2053.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2053.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</li>
</ul>
</p></details></li>
<li>
<p>Which famous models belong to generative models?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Variational Autoencoder</li>
<li>Markov Chain (e.g. Boltzmann Machine)</li>
<li>GANs</li>
</ul>
</p></details></li>
<li>
<p>What are the flavours of Generative Models?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2054.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2054.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</p></details></li>
<li>
<p>On which flavour focuses VAEs and GANs?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Variational Autoencoder → Explicit</li>
<li>GANs → Implicit</li>
</ul>
</p></details></li>
<li>
<p>What are Variational Autoencoder (VAE)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Probabilistic extension of normal autoencoder</li>
<li>Also considered as latent variable model</li>
<li>Make it learn latent parameters that describes the probability distribution of the data<ul>
<li>We start with a standard normal distribution and a prior p_theta(z)</li>
</ul>
</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2055.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2055.png" width="50%"/></p>
<p>Source: <a href="http://kvfrans.com/content/images/2016/08/vae.jpg">http://kvfrans.com/content/images/2016/08/vae.jpg</a></p>
</p></details></li>
<li>
<p>What is the loss function of VAE?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>KL (Kullback Leibler) divergence between the learned latent distribution and its prior distribution. This forces the network to learn latent features that follow the prior distribution</li>
<li>
<p>Reconstruction loss - just as in previous autoencoders, it forces the decoder to match the input.</p>
<p><em>"Kullback–Leibler divergence (also called relative entropy) is a measure of how one probability distribution is different from a second, reference probability distribution"</em></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the key difference between Autoencoder and Variational Autoencoder?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Autoencoders learn a “compressed representation” of input (could be image,text sequence etc.) automatically by first compressing the input (encoder) and decompressing it back (decoder) to match the original input. The learning is aided by using distance function that quantifies the information loss that occurs from the lossy compression. So learning in an autoencoder is a form of unsupervised learning (or self-supervised as some refer to it) - there is no labeled data.</li>
<li>Instead of just learning a function representing the data ( a compressed representation) like autoencoders, variational autoencoders learn the parameters of a probability distribution representing the data. Since it learns to model the data, we can sample from the distribution and generate new input data samples. So it is a generative model like, for instance, GANs.</li>
</ul>
<p><a href="https://www.quora.com/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder"></a></p>
</p></details></li>
<li>
<p>Does GANs use a probability distribution?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Nope</li>
</ul>
</p></details></li>
<li>What is the idea of GANs?<details><summary><b>Answer</b></summary><ul>
<li>Train an NN that produces images based on an input of some images. This part is called Generator Network. The goal of the Generator Network is, to improve its ability to produce images so that the discriminator will fail.</li>
<li>The second actor is the Discriminator Network. Its task is to differentiate between the fake image from the generator and real images from the training set.</li>
</ul></details>
</li>
<li>
<p>How does the loss function look like?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2056.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2056.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019</p>
</p></details></li>
<li>
<p>What is a drawback of the loss function and how to solve it?</p><details><summary><b>Answer</b></summary><p>
<p>When actually training the generator, it was found that the objective function does not work very well because of gradient plateau (flat gradients means very slow if any training).</p>
<p>Solve it by:</p>
<ul>
<li>For this reason the generator objective function is “flipped”.</li>
<li>An intuitive interpretation is that now instead of minimizing the probability of the discriminator being correct, we maximize the probability of it being wrong.</li>
<li>What do Vannila GANs use?</li>
<li>Only fully connected layer but no CNN</li>
<li>When GANs use CNNs, the discriminator uses a normal cnn. What about the generator?</li>
<li>it uses an upsampling CNN with transpose convolution operations (Recall Image segmentation problems)</li>
<li>How does deepfakes video work?</li>
<li>Rely on GANs for data generation.</li>
<li>Combines existing videos with source videos to create new, almost indistinguishable videos of events that are actually completely artificial</li>
</ul>
</p></details></li>
</ol start=110>
<h2>NLP</h2>
<ol start=180>
<li>What is the idea of weight initialization?<details><summary><b>Answer</b></summary><ul>
<li>Using better weight initialization methods leads to a better gradient transport through the network and a faster convergence</li>
</ul></details>
</li>
<li>
<p>What are different approaches for weight initialization?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Weights = 0: Bad Idea - no training at all</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2057.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2057.png" width="50%"/></p>
<p>Source: Source: <a href="https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization">https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization</a></p>
</li>
<li>
<p>Random initialization with mean at 0 and small variance. Seems like it works quite well, although it heavily depends on the used distribution function</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2058.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2058.png" width="50%"/></p>
<p>Source: <a href="https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization">https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization</a></p>
</li>
<li>
<p>Problem:</p>
</li>
<li>
<p>Different variance values for weight initialization lead to vastly different activations, from vanishing to exploding gradients. Linear activation function</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2059.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2059.png" width="50%"/></p>
<p>Source: <a href="https://intoli.com/blog/neural-network-initialization/">https://intoli.com/blog/neural-network-initialization/</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>Why is it important to keep the variance of gradient and activation values of each layer constant?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Otherwise it would lead to vanishing or exploding gradients, which leads to problems in training.</li>
</ul>
</p></details></li>
<li>
<p>For linear functions it is easier to have constant variances</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2060.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2060.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019</p>
</p></details></li>
<li>
<p>Constant variances for NON linear, like relu looks like:</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2061.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2061.png" width="50%"/></p>
<p>Source: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. He et. al.</p>
</p></details></li>
<li>
<p>What is Batch Normalization and why should it be applied?</p><details><summary><b>Answer</b></summary><p>
<p>Why: Covariate Shift ends in bad performance </p>
<p>But the problem appears in the intermediate layers because the distribution of the activations is constantly changing during training. This slows down the training process because each layer must learn to adapt themselves to a new distribution in every training step. This problem is known as internal covariate shift.</p>
<p><a href="https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad">Batch normalization: theory and how to use it with Tensorflow</a></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2062.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2062.png" width="50%"/></p>
<p>That simply means that the datapoints can vary extremly which forces the intermediate layers to readjust.</p>
<p>Where and on what do we do Batch Norm?</p>
<ul>
<li>Usually for the intermediate layers, but it can also be applied on the input layer (taking the raw values). It is not applied on the values itself but on the result of the x vector times the weight vector w, which is z. Z is passed to the activation function and that's why it has to be normalized beforehand. (Sometimes the result of the activation is normalized but this doesn't make sense to me)</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2063.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2063.png" width="50%"/></p>
<p>Source: Andrew Ng, Deep Learning, Coursera</p>
</p></details></li>
<li>
<p>What is the idea of Batch Normalization?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>BN prevent a neural net from exploding or vanishing gradient and reducing learning time due to internal covariate shift. It forces the activations to have mean 0 and unit variance by standardizing it.</li>
</ul>
</p></details></li>
<li>
<p>What are the steps for Batch normalization?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2064.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2064.png" width="50%"/></p>
</p></details></li>
<li>
<p>Should Batch Normalization always be applied?</p><details><summary><b>Answer</b></summary><p>
<p>No, with the following explanation:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2065.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2065.png" width="50%"/></p>
</p></details></li>
<li>
<p>What is the idea of Dropout?</p><details><summary><b>Answer</b></summary><p>
<p>The idea is to prevent an NN from overfitting by removing a specific percentage of neurons from a layer. Often used percentage is between 0.5 and 0.25.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2066.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2066.png" width="50%"/></p>
<p>Source: <a href="https://static.commonlounge.com/fp/original/aOLPWvdc8ukd8GTFUhff2RtcA1520492906_kc">https://static.commonlounge.com/fp/original/aOLPWvdc8ukd8GTFUhff2RtcA1520492906_kc</a></p>
</p></details></li>
<li>
<p>Explain Cosine Similarity</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Measures similarity between two word vectors</li>
<li>
<p>Measures the angle of two words rather than their actual distance to each other.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2067.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2067.png" width="50%"/></p>
<p>Source: <a href="https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/cosine/">https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/cosine/</a></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2068.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2068.png" width="50%"/></p>
<p>Source: <a href="https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/eb9cd609-e44a-40a2-9c3a-f16fc4f5289a.xhtml">https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/eb9cd609-e44a-40a2-9c3a-f16fc4f5289a.xhtml</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>Explain Bleu Score an when to use it</p><details><summary><b>Answer</b></summary><p>
<p><strong>BLEU</strong> (<strong>bilingual evaluation understudy</strong>) is an algorithm for evaluating the quality of text which has been <a href="https://en.wikipedia.org/wiki/Machine_translation">machine-translated</a> from one <a href="https://en.wikipedia.org/wiki/Natural_language">natural language</a> to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" </p>
<p>Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole <a href="https://en.wikipedia.org/wiki/Text_corpus">corpus</a> to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness are not taken into account</p>
<p>BLEU's output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1 representing more similar texts. Few human translations will attain a score of 1, since this would indicate that the candidate is identical to one of the reference translations. For this reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional reference translations will increase the BLEU score</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2069.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2069.png" width="50%"/></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2070.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2070.png" width="50%"/></p>
<p>Source: <a href="https://cloud.google.com/translate/automl/docs/evaluate#:~:text=BLEU%20(BiLingual%20Evaluation%20Understudy)%20is,of%20high%20quality%20reference%20translations">https://cloud.google.com/translate/automl/docs/evaluate#:~:text=BLEU (BiLingual Evaluation Understudy) is,of high quality reference translations</a>.</p>
<ol>
<li>Berechne Precision für jedes N-Gram (wähle min von ref und cand)</li>
<li>Multipliziere über alle n-gram_precision werte</li>
<li>Wende Penalty für kurze Sätze an</li>
</ol>
</p></details></li>
<li>
<p>What are the disadvantages of the Bleu Scores?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>No distinction between content and functional words (ESA - NASA).</li>
<li>Weak in capturing the meaning and grammar of a sentence</li>
<li>Penalty for short sentences can have strong impact</li>
</ul>
</p></details></li>
<li>
<p>Why do we need RNNs and not normal Neural Nets?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Inputs and outputs can be different lengths in different examples.</li>
<li>Doesn't share features learned across different positions of text</li>
</ul>
</p></details></li>
<li>
<p>What's the architecture of a Vanilla RNN?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>a is usually referred as the hidden state, whereas I just see it as the activation from the neuron.</li>
<li>x^<t> is a word of the sentence at position t (same as at time t)</t></li>
<li>As an activation function mostly tanh/Relu is used, but only for the hidden state (a)</li>
<li>For y_hat we usually use a sigmoid function</li>
<li>a&lt;0&gt; (hidden state) is initialized with zero.</li>
<li>
<p>g ist the function (It's actually just the layer and what it does is it has a weight vector for incoming a that is multiplied with the activation value and a weight vector for the new word x that is multiplied with the new word plus a bias), this results in a<i>. g also has a weight vector  for calculating y. It takes that vector and multiplies it with the previously calculated a&lt;0&gt; plus a bias. The result is the word at position <t> bzw. y<t></t></t></i></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2071.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2071.png" width="50%"/></p>
<p>Source: Andrew Ng, Deep Learning, Coursera</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2072.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2072.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the loss of the RNN?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2073.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2073.png" width="50%"/></p>
<p>Source: Fei-Fei Li &amp; Justin Johnson &amp; Serena Yeung</p>
</p></details></li>
<li>
<p>What are problems of Vanilla RNNs?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2074.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2074.png" width="50%"/></p>
<ul>
<li>not good in learning long-term dependencies</li>
<li>What are LSTMs and why are they needed?</li>
<li>Vanilla RNNs are not good in learning long-term dependencies. LSTMs can be seen as a fancier family of RNNs.</li>
<li>Of which components is a LSTM composed?</li>
<li><strong>Cell State:</strong> C(t) Internat cell state (memory)</li>
<li><strong>Hidden State:</strong> External hidden state to calculate the predictions</li>
<li><strong>Input Gate:</strong> Determines how much of the current input is read into the cell state</li>
<li><strong>Forget Gate:</strong> Determines how much of the previous cell state is sent into the current cell state</li>
<li>
<p><strong>Output Gate:</strong> Determines how much of the cell state is output into the hidden state</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2075.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2075.png" width="50%"/></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2076.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2076.png" width="50%"/></p>
<p>Source: <a href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21">https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the sigmoid function for?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The output is a value between 0 and 1. If the network thinks something should be forgotten than it tends to be 0, if it wants to update than it is 1. This is because the output is mostly used for multiplication.</li>
</ul>
</p></details></li>
<li>What is the tanh function for?<details><summary><b>Answer</b></summary><ul>
<li>tanh function squishes values to always be between -1 and 1. This helps preventing the vectors of having too large numbers.</li>
</ul></details>
</li>
<li>What is the cell state for?<details><summary><b>Answer</b></summary><ul>
<li>Memory of the network. As the cell state goes on its journey, information gets added or removed to the cell state via gates.</li>
</ul></details>
</li>
<li>
<p>What does the forget gate do?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Input: Hidden + word (xi)</li>
<li>
<p>Based on the hidden state, the forget gate decides whether to update or if the information should be thrown away (but applied on the hidden state of the previous time stamp, so it kind of cleans up the past)</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2077.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2077.png" width="50%"/></p>
<p>Source: <a href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21">https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What does the input gate do?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Input: hidden + word (xi)</li>
<li>Pass both to sigmoid function to decide which values should be updated</li>
<li>Pass both to tanh to squish values between -1 and 1 </li>
<li>
<p>Multiply sigmoid output with tanh output, to keep important values of tanh output</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2078.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2078.png" width="50%"/></p>
<p>Source: Source: <a href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21">https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>How is the cell state computed?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Input: previous cell state + forget gate output + input gate output</li>
<li>Multiply prev cell state pointwise with forget gate output</li>
<li>Add input gate output to the result</li>
</ul>
</p></details></li>
<li>
<p>What does the output gate do?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Input: previous cell state + prev hidden state + word(xi)</li>
<li>
<p>Calculates the new hidden state by multiplying cell state with the output of sigmoid output gate output</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2079.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2079.png" width="50%"/></p>
<p>Source: <a href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21">https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>How does GRU work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Newer generation of RNN</li>
<li>got rid of cell state</li>
<li>use hidden state to transport information</li>
<li>little speedier to train then LSTM’s</li>
<li>
<p>Reset gate and update gate</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2080.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2080.png" width="50%"/></p>
<p>Source: <a href="https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21">https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21</a></p>
</li>
<li>
<p>Let the network learn when to open and close the gates, i.e. update the “highway” with new information.</p>
</li>
<li>Therefore, instead of updating the hidden state at every RNN cell, the network can learn itself when to update the hidden state with new information.</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2081.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2081.png" width="50%"/></p>
<p>Source: <a href="https://wagenaartje.github.io/neataptic/docs/builtins/gru/">https://wagenaartje.github.io/neataptic/docs/builtins/gru/</a></p>
<ol>
<li>Reset Gate: <ul>
<li>Input: Weight Matrix W, prev hidden state, and word (x<t>)</t></li>
<li>It first computes the reset value. That indicates which value of the input are important and which aren't (recall sigmoid value → 0)</li>
<li>Pass the function of multiplication of prev hidden state and reset vector to the tanh function. This deactivates some prev hidden state values</li>
</ul>
</li>
<li>
<p>Update Gate: </p>
<ul>
<li>
<p>Input: Weight, prev hidden state, x</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2082.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2082.png" width="50%"/></p>
</li>
</ul>
</li>
<li>
<p>Compute Output</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2083.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2083.png" width="50%"/></p>
</li>
</ol>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2084.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2084.png" width="50%"/></p>
<p>I like this illustration more:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2085.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2085.png" width="50%"/></p>
<p>Quote: <em>"Similar to normal RNNs the input gets multiplied by the weight matrix and is added to a hidden layer. However here the input is added to h˜ . The r is a reset switch which represents how much of the previous hidden state to use for the current prediction. Coincidentally there is also a neural network for this reset switch which learns how much of the previous state to allow for predicting the next state. z represents whether to only use the existing hidden state h or use its sum with h~(new hidden state) to predict the output character."</em></p>
<p>Source: <a href="https://medium.com/datadriveninvestor/lstm-vs-gru-understanding-the-2-major-neural-networks-ruling-character-wise-text-prediction-5a89645028f0">https://medium.com/datadriveninvestor/lstm-vs-gru-understanding-the-2-major-neural-networks-ruling-character-wise-text-prediction-5a89645028f0</a></p>
</p></details></li>
<li>
<p>What is the idea of Attention models?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The main idea of the attention mechanism is to imitate the human behaviour of text processing by "reading" a chunk of the input sentence and process it instead of the whole sentence.</li>
<li>The more words the encoder processed, the less information about the single words is contained in the vector. Attention models try to bypass this by save each output vector ht from a state. In contrast to vanilla Seq2Seq-Models, another processing layer between encoder and decoder was added which calculates a score for each ht. The scores indicate how much attention the decoder should pay on a specific ht.</li>
<li>→ Goal: Increase performance for long sentences</li>
</ul>
</p></details></li>
<li>
<p>Describe the algorithm of Attention Models</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2086.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2086.png" width="50%"/></p>
</p></details></li>
<li>
<p>Shorten the algorithm in your own words</p><details><summary><b>Answer</b></summary><p>
<ol>
<li>Choose window of words you want to incorporate its attention</li>
<li>Save the hidden states of them and compute a score for these hidden states</li>
<li>Apply softmax on the scores to get a probability distribution, resulting in attention weights</li>
<li>Compute the context vector by attention weights times hidden states</li>
<li>Take context vector and prev hidden state to compute the next word</li>
</ol>
</p></details></li>
<li>
<p>What is the advantage we have with ConvNets for NLP?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>We can compute different convolutions in parallel for our input vector</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2087.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2087.png" width="50%"/></p>
<p>Source: <a href="https://machinelearningmastery.com/best-practices-document-classification-deep-learning/">https://machinelearningmastery.com/best-practices-document-classification-deep-learning/</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>How does CNN for NLP work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Use filter for multiple words (Similar approach to n-gram)</li>
<li>Vanilla CNNs can be used in One-to-One, Many-to-One or Many-to-Many Architecture.</li>
<li>Used in Text Classification, Sentiment Analysis and Topic Categorization</li>
<li>Problem: Looses the relation information between words, caused by max pooling layers.</li>
</ul>
</p></details></li>
<li>
<p>How can CNNs be combined with Seq2Seq Models?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Each layer has two Matrices</li>
<li>W -  Kernel for convolving input</li>
<li>V - used for GRU computation</li>
<li>
<p>GRU calculation decides based on the sigmoid outcome how much of the conv output is propagated to the next level.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2088.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2088.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>Describe the basic steps of Speech Recognition</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Transform sound wave to spectrogram using fourier transformation</li>
<li>Spectogram is splitted into multiple parts, e.g. 20 ms blocks, resulting in 50 block for a second. A second usually contains between 2 and 3 words.</li>
<li>
<p>Problem: Get rid of repeated characters and separate them into words (CTC = Connectionist Temporal Classification)</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2089.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2089.png" width="50%"/></p>
<p>Source: Hannun, "Sequence Modeling with CTC", Distill, 2017.</p>
</li>
</ul>
</p></details></li>
<li>
<p>How can the problem of removing repeated characters be solved?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Apply Beam Search and take the most likely output</li>
</ul>
</p></details></li>
<li>
<p>Explain a simple approach for Speech to Text</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2090.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2090.png" width="50%"/></p>
</p></details></li>
<li>
<p>Explain Backpropagation through time (BPTT)</p><details><summary><b>Answer</b></summary><p>
<ol>
<li>
<p>Gather loss for all outputs and at every timestamp and sum them up.  (but for each state)</p>
<p>Loss = Sum (y_t - y_t_hat)</p>
</li>
<li>
<p>Apply "normal" backpropagation to each state, keep in mind the partial derivative’s </p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2091.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2091.png" width="50%"/></p>
<p>Source: <a href="http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/">http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/</a></p>
</li>
</ol>
</p></details></li>
<li>
<p>What is Truncated Backpropagation through Time?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The longer the sequence is the harder it is to calculate the gradients -&gt; Solution: truncated backprop</li>
<li>
<p>BPTT is periodically on a fixed number of timesteps applied</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2092.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2092.png" width="50%"/></p>
<p>Source: <a href="https://srdas.github.io/DLBook/RNNs.html">https://srdas.github.io/DLBook/RNNs.html</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is a Language Model?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It gives you a probability of a following sequence of words/characters for a given word. P(Y) = p(y1 ,y2 ,y3 ,...,yn )</li>
<li>This effectively estimates the likelihood of different phrases (in a language).</li>
</ul>
<p>→ Simpler models may look at a context of a short sequence of words, whereas larger models may work at the level of sentences or paragraphs. </p>
<p>Useful for:</p>
<ul>
<li>Speech Recognition</li>
<li>Machine translation</li>
<li>Part-Of-Speech (POS) tagging</li>
<li>Optical Character Recognition (OCR)</li>
</ul>
</p></details></li>
<li>
<p>Explain the idea of N-Grams</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Coalition of n words to "one"</li>
<li>1-gram = Unigram</li>
<li>2-gram = Bigram</li>
<li>Can be used to calculate the probability of a given word being followed by a particular one.</li>
<li>P(x1, x2, ..., xn) = P(x1)P(x2|x1)...P(xn|x1,...xn-1)</li>
</ul>
</p></details></li>
<li>
<p>Define One-Hot Encoding</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Encoding categorical variable so that NN can handle it. We need a vocabulary V with all words in it.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2093.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2093.png" width="50%"/></p>
<p>Source: <a href="https://tensorflow.rstudio.com/guide/tfdatasets/feature_columns/">https://tensorflow.rstudio.com/guide/tfdatasets/feature_columns/</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What are pros and cons of one hot encoding for text data?</p><details><summary><b>Answer</b></summary><p>
<p><strong>Pros:</strong></p>
<ul>
<li>Can be used on different stages for example words or letters</li>
<li>Can represent every text with a single vector</li>
</ul>
<p><strong>Cons:</strong> There are a few problems with the one-hot approach for encoding:</p>
<ul>
<li>The number of dimensions (columns in this case), increases linearly as we add words to the vocabulary. For a vocabulary of 50,000 words, each word is represented with 49,999 zeros, and a single “one” value in the correct location. As such, memory use is prohibitively large.</li>
<li>The embedding matrix is very sparse, mainly made up of zeros.</li>
<li>There is no shared information between words and no commonalities between similar words. All words are the same “distance” apart in the n-dimensional embedding space.</li>
<li>Example of one hot encoding</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2094.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2094.png" width="50%"/></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2095.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2095.png" width="50%"/></p>
<p>Source: <a href="https://programmersought.com/article/64902775751/">https://programmersought.com/article/64902775751/</a></p>
</p></details></li>
<li>
<p>What is the idea of word embeddings?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Encoding for meaning and similarity of words in form of a vector that exists of latent dimensions (latent feature space)</li>
<li>Main motivation: we want to capture some intrinsic “meaning” of a word and be able to draw comparisons between words</li>
<li>
<p>Example of latent feature space:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2096.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2096.png" width="50%"/></p>
<p>Source: <a href="https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf">https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What are word embeddings used for?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Semantic Analysis</li>
<li>Named Entity Recognition</li>
<li>Weights initilization</li>
</ul>
</p></details></li>
<li>What is a problem with word embeddings?<details><summary><b>Answer</b></summary><ul>
<li>Need huge corpora to train, that's why mostly pre-trained embeddings are used</li>
</ul></details>
</li>
<li>What is the basic approach of word embeddings?<details><summary><b>Answer</b></summary><ul>
<li>Words that appear in similar contexts (close to each other) are likely to be semantically related.</li>
<li>If we want to “understand” the meaning of a word, we take information about the words it coappears with.</li>
</ul></details>
</li>
<li>
<p>How does the training process look like?</p><details><summary><b>Answer</b></summary><p>
<p>We want to have p(context|wt), where wt is the word we are looking at and context the surrounding words.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2097.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2097.png" width="50%"/></p>
<ol>
<li>Used Loss function: -log[p(context|wt)] <a href="//somewhat">//Somewhat</a> like cross entropy loss function</li>
<li>Optimize NN based on Used Loss Function</li>
<li>If we can predict context we have good embedding</li>
<li>delete classifier keep embeddings</li>
</ol>
<p>Source Illustration: <a href="https://laptrinhx.com/get-busy-with-word-embeddings-an-introduction-3637999906/">https://laptrinhx.com/get-busy-with-word-embeddings-an-introduction-3637999906/</a></p>
</p></details></li>
<li>
<p>What was an early attempt of word-embeddings?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Neural probabilistic Language Models</li>
<li>→ Generate joint probability functions of word sequences</li>
</ul>
</p></details></li>
<li>What were shortcomings of Neural Probabilistic?<details><summary><b>Answer</b></summary><ul>
<li>It was only looking at words before but not at words after the predicted one</li>
</ul></details>
</li>
<li>
<p>How does Skip-Gram work?</p><details><summary><b>Answer</b></summary><p>
<p>blue is the given word to predict the surroundings → + + + C + + + </p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2098.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2098.png" width="50%"/></p>
<p>Source: <a href="http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/">http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/</a></p>
</p></details></li>
<li>
<p>How does word2vec work?</p><details><summary><b>Answer</b></summary><p>
<p>Idea relies on estimating word co-occurences.</p>
<p>For Skip-gram:</p>
<ol>
<li>Select Windows size</li>
<li>Select a word wt</li>
<li>For a selected word wt in our training corpus look a randomly selected word within m positions of wt.</li>
<li>train to predict correct context words</li>
<li>Predict by softmax classifier</li>
</ol>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2099.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2099.png" width="50%"/></p>
</p></details></li>
<li>
<p>What is a problem with word2vec?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>At each window the normalization factor over the whole vocabulary has to be trained (e.g. 10.000 dot products) → computationally too expensive</li>
</ul>
</p></details></li>
<li>How can the problems of word2vec be solved?<details><summary><b>Answer</b></summary><ul>
<li>Applying negative sampling</li>
</ul></details>
</li>
<li>
<p>What is negative sampling</p><details><summary><b>Answer</b></summary><p>
<p>Main idea: we want to avoid having to train the full softmax at each prediction. “A good model should be able to differentiate data from noise by means of logistic regression”.</p>
<ul>
<li>Train two binary logistic regressions<ol>
<li>One for the true pair (center word, output)</li>
<li>One for random pairs (center word, random word)</li>
</ol>
</li>
</ul>
<p>Exaxmple:</p>
<ul>
<li>
<p><em>"I want a glass of orange juice to go along with my cereal."</em> 
For every positive example, we have k-1 negative examples. Here k = 4.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20100.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20100.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>How does CBOW work?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Target is a single word → C C C + C C C</li>
<li>Context is now all words within the context window of size m.</li>
<li>Average embeddings of all context words</li>
</ul>
</p></details></li>
<li>
<p>Differentiate CBOW from Skip-gram</p><details><summary><b>Answer</b></summary><p>
<p>According to the author, CBOW is faster but skip-gram but does a better job for infrequent words.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20101.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20101.png" width="50%"/></p>
<p>Source: <a href="https://www.knime.com/blog/word-embedding-word2vec-explained">https://www.knime.com/blog/word-embedding-word2vec-explained</a></p>
</p></details></li>
<li>
<p>What does GloVe?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Calculate co-occurrence directly for the whole corpus</li>
<li>We go through the entire corpus and update the co-occurrence counts.</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20102.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20102.png" width="50%"/></p>
<ul>
<li>Go through all pairs of words in the co-occurrence matrix.</li>
<li>Minimize distance between dot product of 2 word embeddings</li>
<li>Function <em>f</em>(Pij) - allows one to weight the co-occurrences. E.g. give a lower weight to frequent co-occurrences.</li>
<li>Trains much more quickly and with small corpora</li>
<li>How can word embeddings be evaluated?</li>
<li>
<p>Intrinsic Evaluation</p>
<ol>
<li>Evaluate on a specifically created subtask</li>
</ol>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20103.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20103.png" width="50%"/></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20104.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20104.png" width="50%"/></p>
<p>Source Illustration: Pennington, Socher, Manning: <a href="https://nlp.stanford.edu/projects/glove/">https://nlp.stanford.edu/projects/glove/</a></p>
<ol>
<li>Another intrinsic evaluation approach is comparing with human judgments.</li>
</ol>
</li>
<li>
<p>Extrinsic Evaluation</p>
<ul>
<li>How good are our embeddings for solving actual tasks?</li>
<li>Example of such task is named entity recognition.</li>
</ul>
</li>
<li>Information on dependence on Hyperparameters</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20105.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20105.png" width="50%"/></p>
<p>Source Slides: Socher, Manning</p>
</p></details></li>
<li>
<p>What are advantages of Character Level Models?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Vocabulary is just of the size of the alphabet + punctuation</li>
<li>No need for tokenization</li>
<li>No unknown word (out of vocabulary)</li>
<li>Can learn structures data encoded by characters and only plain text (For example gen informations)</li>
</ul>
</p></details></li>
<li>What are disadvantages of Character Level Models?<details><summary><b>Answer</b></summary><ul>
<li>Require deeper RNN's because there are more characters in a sentence than words</li>
<li>Generally worse capturing long-distance dependencies</li>
<li>In order to model long term dependencies, hidden layers need to be larger</li>
</ul></details>
</li>
<li>
<p>What are different types of NNs?</p><details><summary><b>Answer</b></summary><p>
<p>One to One: Image Classification</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20106.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20106.png" width="50%"/></p>
<p>Illustration: Andrey Karpathy. The Unreasonable Effectiveness of Recurrent Neural Networks</p>
<p>One to Many: Generating caption for an image</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20107.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20107.png" width="50%"/></p>
<p>Source: <a href="https://medium.com/swlh/automatic-image-captioning-using-deep-learning-5e899c127387">https://medium.com/swlh/automatic-image-captioning-using-deep-learning-5e899c127387</a></p>
<p>Many to One: Sentiment Classification</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20108.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20108.png" width="50%"/></p>
<p>Source:</p>
<p>Many to Many (Sequence2Sequence): Machine Translation where length of input and output can be different or Entity Recognition</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20109.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20109.png" width="50%"/></p>
<p>Source: <a href="https://confusedcoders.com/data-science/deep-learning/how-to-build-deep-neural-network-for-custom-ner-with-keras">https://confusedcoders.com/data-science/deep-learning/how-to-build-deep-neural-network-for-custom-ner-with-keras</a></p>
</p></details></li>
<li>
<p>Of what does the Sequence2Sequence architecture exist?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20110.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20110.png" width="50%"/></p>
<p>Source: <a href="https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346">https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346</a></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20111.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20111.png" width="50%"/></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20112.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20112.png" width="50%"/></p>
</p></details></li>
<li>
<p>How can the process of predicting in the decoder part be improved?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li><strong>Further Improvements</strong>
Remember, we are creating our word vector based on a probabilistic model.
Predicting the output vector using a greedy approach may not be the best solution.</li>
</ul>
</p></details></li>
<li>
<p>Describe Beam Search</p><details><summary><b>Answer</b></summary><p>
<p>Beam search is an approximate search algorithm, so no guarantee to find a global maximum. Beam Search is a heuristic algorithm - it does not guarantee the optimal solution.</p>
<p>Instead of selecting the first best word, we take the k best matching words and calculate the probabilities for all the next words. Out of these, choose again the top b matching samples and repeat. If we choose b=1, than beam seach is a greedy search.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20113.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20113.png" width="50%"/></p>
<p>Source: <a href="https://d2l.ai/chapter_recurrent-modern/beam-search.html">https://d2l.ai/chapter_recurrent-modern/beam-search.html</a></p>
</p></details></li>
<li>
<p>What is a painpoint of beam search?</p><details><summary><b>Answer</b></summary><p>
<p>The probability of a sentence is defined as product of probabilities of words conditioned on previous words. Due to compounding probabilities, longer sentences
have a lower probability by definition. Shorter sentences can have higher probability even if they make less sense.</p>
</p></details></li>
<li>
<p>How can the painpoints of beam search be solved?</p><details><summary><b>Answer</b></summary><p>
<p>We need to normalize sentences by their length multiply the probability of a sequence by its inverse length.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20114.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20114.png" width="50%"/></p>
</p></details></li>
<li>
<p>Describe the light normalization implementation</p><details><summary><b>Answer</b></summary><p>
<ol>
<li>run, select and save top 3 most likely sequences for every sequence length (e.g. Defined max sequence length = 5 → 1,2,3,4,5)</li>
<li>
<p>This results in a set of size 3 containing</p>
<ul>
<li>1 sequences (single words, e.g.: ("Oh","donald","trump"), ("Go","Barack","Obama"), (...) )</li>
<li>2 sequences (two word sequences)</li>
<li>Up to N (here 5) sequences</li>
</ul>
</li>
<li>
<p>For every sequence length run a normalized probability score calculation and assign it to every sequence.</p>
</li>
<li>
<p>Select the sequence with highest probability of all</p>
</li>
</ol>
</p></details></li>
<li>
<p>Which two errors can arise with Beam Search?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Model errors from our RNN</li>
<li>
<p>Search errors from our beam search hyperparameter</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20115.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20115.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
</ol start=180>
<h2>Reinforcement Learning</h2>
<ol start=245>
<li>Describe the idea of RL<details><summary><b>Answer</b></summary><ul>
<li>The idea of RL is that an algorithm learns a task by itself without precise instructions from a human. A developer defines the goal and provides it with a reward. In addition, it must be defined which actions the model is allowed to perform (actions). The model explores several strategies over a period of time and acquires the best one. This is also known as trial-and-error search.</li>
</ul></details>
</li>
<li>Describe the terminology of RL<details><summary><b>Answer</b></summary><ul>
<li><strong>Environment</strong>: The abstract world that everything happens in.</li>
<li><strong>Agent</strong>: The Algorithm which interacts with the environment</li>
<li><strong>State</strong>: Part of the environment. Every action of the agent with the Env changes their state</li>
<li><strong>Reward</strong>: Every interaction of an agent with the environment produces a reward. Reward is a signal of how good or bad the interaction was in terms of some goal. At some point, the interaction is terminate. The final goal of the agent is to maximize the cumultative reward</li>
<li><strong>Policy</strong>: The rules by which the agent acts (π)</li>
</ul></details>
</li>
<li>When to use Reinforcement Learning<details><summary><b>Answer</b></summary><ul>
<li>If the problem is an inherently sequential decision-making problem i.e. a board game like chess or games in general.</li>
<li>If the optimal behaviour is not known in advance (unsolved games without "perfect strategy")</li>
<li>When we can simulate the system behaviour as good as possible to imitate the real world</li>
<li>Solving problems with a non-differentiable loss function</li>
</ul></details>
</li>
<li>
<p>RL is separated into Model-Free RL and Model-Based RL. Name subcategories and associated algorithms</p><details><summary><b>Answer</b></summary><p>
<p>Model-Free RL:</p>
<ul>
<li>Policy Optimization (DDPG, Policy Gradient)</li>
<li>Q-Learning (DQN, DDPG)</li>
</ul>
<p>Model Based RL:</p>
<ul>
<li>Learn the Model (World Models)</li>
<li>Given the Model (AlphaZero)</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20116.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20116.png" width="50%"/></p>
</p></details></li>
<li>
<p>Tasks of the Environment</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Interface: An Env need to provide an interface with which the agent could interact</li>
<li>Calculating Effects: For an action performed by the agent, the Env needs to calculate the:<ul>
<li>next state</li>
<li>reward for this action</li>
</ul>
</li>
</ul>
</p></details></li>
<li>Describe Model Free approaches<details><summary><b>Answer</b></summary><ul>
<li>The focus is on finding out the value functions directly by interactions with the environment. All Model Free algorithms learn the value function directly from the environment. That's not entirely true. Looking at Policy Gradient approaches we do not learn the value function.</li>
</ul></details>
</li>
<li>
<p>Define the Markov Decision Problem (MDP)</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20117.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20117.png" width="50%"/></p>
<ul>
<li>The MDP is inspired by the theorem "The future is independent of the past given the present". (Markov Property) That means, to calculate values for the future state $S_{t+1}$, only the state $S_t$ counts. It should keep all the information from already passed states.</li>
</ul>
</p></details></li>
<li>
<p>Describe the Markov Property</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>A state contains the Markov property. This gives the probabilities of all possible next states under the condition of the current state in a matrix P. (Markov Property: Capture all relevant information from the past)</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20118.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20118.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the discounted reward?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It is the discounted sum over all rewards that happened in the future. Intuitively speaking it is something like a measurement how efficient the agent solved an episode. E.g.: An agent decided to walk loops in a  GridWorld scenario and after a few loops he changed his route towards the goal and reached it. The discounted is still good but it would have been better if he had taken the direct path, because reward further in the future are discounted the most. Hence, the final is always discounted most.</li>
<li>Recall that return is the total discounted reward in an episode:<ul>
<li>G_t = R_t+1 + γR_t+2 + ... + γT-1RT</li>
</ul>
</li>
</ul>
</p></details></li>
<li>
<p>What is a policy?</p><details><summary><b>Answer</b></summary><p>
<p>A policy is the agent’s behaviour. It is a map from state to action.</p>
<ul>
<li>Deterministic policy: a=π(s).</li>
<li>Stochastic policy: π(a|s) =P[A=a|S=s].</li>
</ul>
</p></details></li>
<li>
<p>What is the State Value Function?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Value function for a state is the reward under the assumption of taking the best action/policy from there. → Kind of the best Q Value</li>
<li>
<p>The Value function takes the reward of all possible next states into account (and the next rewards of those states, which is G) and calculates the value by multiplying the action probability (policy) by the reward for the particular state.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20119.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20119.png" width="50%"/></p>
</li>
</ul>
<p>However the description above can be written differently (much smarter so that we can apply dynamic programming on it)!</p>
<p>Explanation of the belman equation:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20120.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20120.png" width="50%"/></p>
<p>Discounted Reward: That's just the sum of all rewards (discounted with gamma, but this isn't important to understand the concept here). </p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20121.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20121.png" width="50%"/></p>
<p>Defining G_t brings us to the Expectation_policy </p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20122.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20122.png" width="50%"/></p>
<p>of </p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20123.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20123.png" width="50%"/></p>
<p>The expectation is mathematically defined by the probability (here Pi/policy) times over all elements (here G_t). The policy/probability is the probability of a specific action under state s.</p>
<p>DP, MC and TD Learning methods are value-based methods (Learnt Value Function, Implicit policy).</p>
</p></details></li>
<li>
<p>What is the Action Value Function?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Equation almost the same as State Value Function</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20124.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20124.png" width="50%"/></p>
</p></details></li>
<li>
<p>Why do we need the bellman equation and how to solve it?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Reason: Computing entire value function for some RL use case is too complex/time consuming, therefore, breaking it down to sub equations.</li>
<li>
<p>Usually solved by using backup methods, we transfer then the information back to a state from its successor state. Therefore we can calculate paths separately?</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20125.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20125.png" width="50%"/></p>
<p>Source: <a href="https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/">https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/</a></p>
</li>
</ul>
<p>Can also be applied to q-values!</p>
</p></details></li>
<li>
<p>How is optimal policy defined?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Optimal policy is defined as that policy which has the greatest expected return for <em>all states</em></li>
<li>
<p>We can explicitly write out this requirement for state-value functions and state-action value functions.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20126.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20126.png" width="50%"/></p>
<p>Source: <a href="https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/">https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>The optimal policy and the bellman equation leads to the Bellman optimality equation</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>By that we formulate the equation without referring to a specific policy (that's why a star is beneath v)</li>
<li>"Intuitively, this says that the value of a state under an optimal policy must equal the expected return for the best action from that state."</li>
<li>
<p>The next image shows this for q values. The order would be that we select the best action at the current state s, following the optimal policy which results in the best value function value.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20127.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20127.png" width="50%"/></p>
<p>Source: <a href="https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/">https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the Markov Reward Process (MRP)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>MRP is a tuple of (S,P,R, gamma)</p>
<ul>
<li>S is a state space</li>
<li>P is the state transaction function (probability function)</li>
<li>R is the reward function with E(R_t+1|S), i.e. how much immediate reward we would get for a state S.</li>
<li>
<p>Everything results in G_t, which is the total discounted reward for a time step t. The goal is to maximize this function:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20128.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20128.png" width="50%"/></p>
</li>
<li>
<p>Gamma is the discount factor which is applied to future rewards. This should play in less than the "closer". This also leads us to avoid infinite returns in Markov processes.</p>
<ul>
<li>Another important point is the value function V(S) which can be adapted to different use cases. The original MDP uses the Bellman Equation:</li>
</ul>
</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20129.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20129.png" width="50%"/></p>
<p>For gamma = 1 we aim at long term rewards and for gamma = 0 we tend to short term profit.</p>
<p>In the following example, a discount of 0.9 is used from the final state reward(Peach) to the start state of the agent (Mario).</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20130.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20130.png" width="50%"/></p>
<p>However, to solve the Bellman equation we need almost all the information of our environment: Markov property and calculate v* for all other states. Since this rarely works, many RL methods try to approximate the Bellman function with the last done state transition instead of having a complete knowledge of all transitions.</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the difference between State Value V(s) and Action Values Q(s,a)?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Basically two different ways to solve reinforcement learning problems.</li>
<li>V_Pi(s) describes the expected value of following the policy Pi when the agent starts at the state S. Basically, the action is greedy.</li>
<li>Q_Pi (s,a) describes the expected value when an action a is taken from state S under policy Pi.</li>
</ul>
</p></details></li>
<li>
<p>What is the relationship between Q-Values (Q_Pi) and the Value function (V_Pi)?</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20131.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20131.png" width="50%"/></p>
<ul>
<li>You sum every action-value multiplied by the probability to take that action (the policy 𝜋(𝑎|𝑠)*π(a|s)).</li>
<li>
<p>If you think of the grid world example, you multiply the probability of (up/down/right/left) with the one step ahead state value of (up/down/right/left).</p>
<p>Source: <a href="https://datascience.stackexchange.com/questions/9832/what-is-the-q-function-and-what-is-the-v-function-in-reinforcement-learning">https://datascience.stackexchange.com/questions/9832/what-is-the-q-function-and-what-is-the-v-function-in-reinforcement-learning</a></p>
</li>
</ul>
</p></details></li>
<li>
<p>Describe the idea of Q-Learning</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances.</li>
<li>Approximation of the MDP with the approach of action states without solving the Bellman equation. It is a MDP with finite and small enough state space and actions</li>
<li>
<p>How can a grid world problem be solved with Q-learning?</p>
<ol>
<li>Initialize a table for each state-action pair</li>
<li>Fill in the table step by step while interacting with the environment</li>
<li>
<p>Calculate the Q-Values</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20132.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20132.png" width="50%"/></p>
<p>Source: <a href="http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf">http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf</a></p>
</li>
<li>
<p>Repeat step 2 + 3 for N epochs</p>
</li>
<li>
<p>Check how well the learned table approximated the State Values (V(s)) for the Bellmann equation. This could look like the following:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20133.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20133.png" width="50%"/></p>
<p>Source: <a href="https://towardsdatascience.com/qrash-course-deep-q-networks-from-the-ground-up-1bbda41d3677">https://towardsdatascience.com/qrash-course-deep-q-networks-from-the-ground-up-1bbda41d3677</a></p>
</li>
</ol>
</li>
</ul>
</p></details></li>
<li>
<p>What is Policy Iteration?</p><details><summary><b>Answer</b></summary><p>
<p><strong>Policy iteration</strong> includes: <strong>policy evaluation</strong> + <strong>policy improvement</strong>, (both are iteratively)</p>
<ul>
<li>Uses also bellman equation but without maximum.</li>
<li>For each state S we check if the Q-value is larger than the value value: q(s,a) &gt; v(s) under the same policy (test different policies). If the Q-Value returns a larger value, the strategy performs better and we switch to it.</li>
</ul>
<p>Sometimes it only finds local minima</p>
<ol>
<li>Choose a policy randomly (e.g. go in each direction with prob = 0.25)</li>
<li>Do policy evaluation: Calculate the Value for each state</li>
<li>Do policy improvement: Change an action a (that results in a new policy pi') and see if the values are better. If so, switch to the new policy,. (We could use the Value function to do this successively for each state</li>
<li>
<p><a href="https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/">https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/</a></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20134.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20134.png" width="50%"/></p>
</li>
</ol>
</p></details></li>
<li>
<p>Example of Policy Iteration for Monte Carlo</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20135.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20135.png" width="50%"/></p>
<p>Source: <a href="https://www.programmersought.com/article/39135619445/">https://www.programmersought.com/article/39135619445/</a></p>
</p></details></li>
<li>
<p>What is Value iteration?</p><details><summary><b>Answer</b></summary><p>
<p><strong>Value iteration</strong> includes: <strong>finding optimal value function</strong> + one <strong>policy extraction</strong></p>
<p><strong>Usually finds global minima</strong></p>
<p>It is similar to policy iteration, although it differs in the following:</p>
<ol>
<li>Value Iteration tries to find the optimal value function rather than first the optimal policy. Once the best value function was found, the policy can be derived.</li>
<li>It shares the policy improvement function/approach and a truncated policy evaluation</li>
</ol>
<p>For value iteration it can be less iterations but for one iteration there can be so much of work, because it uses bellman maximum equation. For the policy iteration more iterations.</p>
</p></details></li>
<li>
<p>Compare Policy Iteration with Value Iteration</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p><em>"In my experience, policy iteration is faster than value iteration, as a policy converges more quickly than a value function. I remember this is also described in the book."</em></p>
<p>Source: <a href="https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration">https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration</a></p>
</li>
<li>
<p>Policy Iteration: Sometimes it only finds local minima</p>
</li>
<li>Value Iteration: Finds global minima</li>
</ul>
</p></details></li>
<li>
<p>What is a disadvantage of policy iteration?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Can get stuck in a local minimum</li>
</ul>
</p></details></li>
<li>
<p>How can the disadvantages of policy iteration be compensated?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Early stopping</li>
</ul>
</p></details></li>
<li>
<p>What is the idea of the Monte Carlo method?</p><details><summary><b>Answer</b></summary><p>
<p>In General: A statistical approach that averages the mean for a state over an episode.</p>
<ol>
<li>Initialize rewards randomly</li>
<li>Run through a specific number of episodes (e.g. 1000)</li>
<li>~~Calculate the discounted reward for each state~~</li>
<li>Apply either:<ol>
<li>First visit method: Average returns only for first time s is visited in an episode.</li>
<li>Every visit method: Average returns for every time s is visited in an episode.</li>
</ol>
</li>
</ol>
<p>(Check again if example is correct (might be wrong))</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20136.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20136.png" width="50%"/></p>
<p>Source: <a href="https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/">https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/</a></p>
</p></details></li>
<li>
<p>Give a summary of Monte Carlo</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>MC methods learn directly from episodes of experience.</li>
<li>MC is model-free: no knowledge of MDP transitions / rewards.</li>
<li>MC uses the simplest possible idea: value = mean return.</li>
<li>Episode must terminate before calculating return.</li>
<li>Average return is calculated instead of using true return G.</li>
<li>First Visit MC: The first time-step t that state s is visited in an episode.</li>
<li>Every Visit MC: Every time-step t that state s is visited in an episode.</li>
<li><a href="https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/README.md#MonteCarlo">https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/README.md#MonteCarlo</a></li>
</ul>
</p></details></li>
<li>
<p>What is the idea of Temporal Difference Learning?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Temporal Difference Learning (also TD learning) is a method of reinforcement learning. In reinforcement learning, an agent receives a reward after a series of actions and adjusts its strategy to maximize the reward. An agent with a TD-learning algorithm makes the adjustment not when it receives the reward, but after each action based on an estimated expected reward.</p>
<p>Source: <a href="https://de.wikipedia.org/wiki/Temporal_Difference_Learning">https://de.wikipedia.org/wiki/Temporal_Difference_Learning</a></p>
</li>
<li>
<p>Is an on-policy learning approach</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20137.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20137.png" width="50%"/></p>
<p>Source: <a href="https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo">https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo</a></p>
</li>
<li>
<p>Compared to DP and MC, which are both offline because they have to finish the epoch first, TD is a mix of the two. Monte Carlo has to run a branch of States, DP even has to run all of them.</p>
</li>
<li>Summary Temporal Difference Learning</li>
<li>TD methods learn directly from episodes of experience.</li>
<li>TD updates a guess towards a guess</li>
<li>TD learns from incomplete episodes, by bootstrapping.</li>
<li>TD uses bootstrapping like DP, TD learns experience like MC (combines MC and DP).</li>
<li>MC-TD Difference</li>
<li>MC and TD learn from experience.</li>
<li>TD can learn before knowing the final outcome.</li>
<li>TD can learn online after every step. MC must wait until end of episode before return is known.</li>
<li>TD can learn without the final outcome.</li>
<li>TD can learn from incomplete sequences. MC can only learn from complete sequences.</li>
<li>TD works in continuing environments. MC only works for episodic environments.</li>
<li>MC has high variance, zero bias. TD has low variance, some bias.</li>
<li>What is the idea of Dynamic Programming?</li>
<li>
<p>DP is a general framework concept. It is derived from the divide and conquer principle, where DP tries to break down a complex problem into sub-problems that are easier to solve. In the end, the solutions of the subproblems are merged to solve the main problem.</p>
<p>In order to apply it, the following properties must be given:</p>
<ul>
<li>The optimal solution can be divided into subproblems</li>
<li>Subproblems occur more often</li>
<li>Subproblems solutions can be cached</li>
<li>Are the dynamic programming requirements met for a MDP?<ul>
<li>Bellman equation (is a DP approach) gives us a recursive decomposition</li>
<li>The value function stores the action (serves as a cache) and can thus be used for DP</li>
</ul>
</li>
<li>What is meant by full backups vs sample backups and shallow vs deep?<ul>
<li>The history of the learned. Full backup utilizes a lot of paths and states, whereas sample backup is a branch in the tree diagram.</li>
<li>Shallow means that it does not go very deep into the tree before making an update (fast), whereas deep waits for a complete episode.</li>
</ul>
</li>
</ul>
</li>
</ul>
<p><a href="http://gki.informatik.uni-freiburg.de/teaching/ws0607/advanced/recordings/reinforcement.pdf"></a></p>
</p></details></li>
<li>
<p>What is the difference between on-policy and off-policy?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Online: during the job: E.g TD Algorithm</li>
<li>
<p>On policy learning (Monte Carlo): Improve the same policy that is used to make decisions (generate experience).</p>
</li>
<li>
<p>Off policy learning (Q-Learning (TD-Algorithm)):  Improve policy different than the one used to generate experience. Here, improve target policy while following behaviour.</p>
<ul>
<li>Target policy - what we want to learn. Therefore, we evaluate and improve upon.</li>
<li>Behaviour policy - what we use to generate data.</li>
</ul>
<p>Motivation for having 2 policies is the trade-off between exploration and exploitation. In order to know what actions are best, we need to explore as much as possible - i.e. take suboptimal actions as well.</p>
<p>Target policy π(St) is greedy w.r.t. Q(s,a),  because we want our target policy to provide the best action available.</p>
<p>Behavior policy μ(St) is often 𝜀-greedy w.r.t. Q(s,a).</p>
<ul>
<li>𝜀-greedy policy is an extension of greedy policy<ul>
<li>With probability 𝜀 it will output a random action.</li>
<li>Otherwise, it output is the best action.</li>
<li>𝜀 controls how much exploration we want to allow.</li>
</ul>
</li>
</ul>
<p>This combination of two policies let’s us combine “the best of both worlds”. Our behavior policy gives enough randomness to explore all possible state-action pairs. Our target policy still learns to output the best action from each state.</p>
</li>
</ul>
</p></details></li>
<li>
<p>What is the difference between Episodic and Continuous Tasks?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Episodic task: A task which can last a finite amount of time is called Episodic task ( an episode )</li>
<li>Continuous task: A task which never ends is called Continuous task. For example trading in the cryptocurrency markets or learning Machine learning on internet.</li>
</ul>
</p></details></li>
<li>
<p>Differentiate Exploitation from Exploration</p><details><summary><b>Answer</b></summary><p>
<ul>
<li><strong>Exploitation</strong>: Following a strict policy by exploiting the known information from already known states to maximize the reward.</li>
<li><strong>Exploration</strong>: Finding more information about the environment. Exploring a lot of states and actions in the environment by avoid a greedy policy every action.</li>
</ul>
</p></details></li>
<li>
<p>What is the sample efficiency problem?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>After improving our policy by an experience we do not use this information anymore, but with future experiences, which is not sample efficient.</li>
</ul>
</p></details></li>
<li>Define Experience Replay<details><summary><b>Answer</b></summary><ul>
<li>Another human behaviour imitation technique is called experience replay. The idea behind experience replay is, that the agent builds a huge memory with already experienced events.</li>
<li>For this, a memory M is needed which keeps experiences e. We store the agents experience et=(st,at,rt,st+1)e_{t}=\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)et=(st,at,rt,st+1) there.
This means instead of running Q-learning on state/action pairs as they occur during simulation or actual experience, the system stores the data discovered for [state, action, reward, next_state] - typically in a large table. Note this does not store associated values - this is the raw data to feed into action-value calculations later.</li>
<li>The learning phase is then logically separated from gaining experience, and based on taking random samples from this table which does not stand in a directional relationship. You still want to interleave the two processes - acting and learning - because improving the policy will lead to different behaviour that should explore actions closer to optimal ones, and you want to learn from those.</li>
</ul></details>
</li>
<li>What are the advantages of experience replay?<details><summary><b>Answer</b></summary><ul>
<li>More efficient use of previous experience, by learning with it multiple times. This is key when gaining real-world experience is costly, you can get full use of it. The Q-learning updates are incremental and do not converge quickly, so multiple passes with the same data is beneficial, especially when there is low variance in immediate outcomes (reward, next state) given the same state, action pair.</li>
<li>Better convergence behaviour when training a function approximator. Partly this is because the data is more like i.i.d. data assumed in most supervised learning convergence proofs.</li>
</ul></details>
</li>
<li>
<p>What are advantages of Experience Replay?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>This turns our RL problem into a supervised learning problem!</li>
<li>Now we reuse our data, which is far more sample efficient.</li>
<li>
<p>Before, we worked with non i.i.d., strongly correlated data - we always saw states</p>
<p>one after the other in the trajectory (now we randomly sample) </p>
<ul>
<li>We break the correlation by presenting (state,value) pairs in a random order.</li>
<li>This greatly increases the convergence rate (or even makes possible).</li>
</ul>
</li>
</ul>
</p></details></li>
<li>
<p>What are the disadvantages of experience replay?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>It is harder to use multi-step learning algorithms, such as $Q(\lambda)$ , which can be tuned to give better learning curves by balancing between bias (due to bootstrapping) and variance (due to delays and randomness in long-term outcomes). Multi-step DQN with experience-replay DQN is one of the extensions explored in the paper Rainbow: Combining Improvements in Deep Reinforcement Learning.</li>
</ul>
</p></details></li>
<li>
<p>How can we re-use the data?</p><details><summary><b>Answer</b></summary><p>
<ol>
<li>Saving experiences (Gather data)</li>
</ol>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20138.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20138.png" width="50%"/></p>
<ol>
<li>Apply Experience Replay<ol>
<li>Define cost function, such as Least Square</li>
<li>Random sample (state, value) pair from experience</li>
<li>Apply SGD Update</li>
</ol>
</li>
<li>What is the idea of Deep Q-Learning (DQN)?</li>
</ol>
<p>DQN is quite young and was introduced only in 2013. It is an end-to-end learning for the Q(s,a) values for pixels with which well-known Atari games can be solved. Two main ideas are followed:</p>
<ul>
<li>Experience Replay (kind of generator for the data generation of the training process).</li>
<li>Fixed Q-Targets:<ul>
<li>Form of off-policy method</li>
<li>Direct manipulation of a used Q Value leads to an unstable NN.</li>
<li>To solve this two parameters are needed:<ol>
<li>current Q values</li>
<li>target Q values</li>
<li>During training, we update the target Q values with Stochastic Gradient Descent. After the epoch, we replace our current weights with the target weights and start a new epoch.</li>
</ol>
</li>
</ul>
</li>
</ul>
</p></details></li>
<li>
<p>So far we have seen Value Based RL which relies on improving the model by the state value function, but there is also the possibility to learn a policy directly. What are the advantages?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Better convergence properties</li>
<li>When the space is large, the usage of memory and computation consumption grows rapidly. The policy based RL avoids this because the objective is to learn a set of parameters that is far less than the space count.</li>
<li>Can learn stochastic policies. Stochastic policies are better than deterministic policies, especially in 2 players game where if one player acts deterministically the other player will develop counter measures in order to win.</li>
</ul>
</p></details></li>
<li>What are disadvantages of Policy Based RL?<details><summary><b>Answer</b></summary><ul>
<li>Typically converge to a local rather than global optimum</li>
<li>Evaluating a policy is typically inefficient and high variance policy based RL has high variance, but there are techniques to reduce this variance.</li>
</ul></details>
</li>
<li>
<p>How can we fix a RL case where the problem is too large (keeping track of every single state or state-action pair becomes infeasible)</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>Value Function Approximation. Therefore we extend the original Value function of a state and the q- values with a weight parameter.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20139.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20139.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>What kind of function can be used for Value Function approximation? Elaborate.</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p><strong>Linear combinations of features:</strong></p>
<ul>
<li>
<p>In this case we represent the value function by a linear combination of the features and weights.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20140.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20140.png" width="50%"/></p>
<ul>
<li>This lets us generalize from seen states to unseen states, by optimizing the function to already seen value function for a state and apply the parameters for unseen states.</li>
</ul>
</li>
</ul>
<p>Big advantage of this approach - it converges on a global optimum.</p>
</li>
<li>
<p><strong>Neural networks:</strong></p>
<ul>
<li>Most general goal: find optimal parameter w by finding its (local) minimum. Cost function: mean squared error.</li>
</ul>
</li>
<li>~~Monte Carlo Method:~~</li>
<li>In which step during the value function approximation is the approximation checked?</li>
<li>Policy evaluation</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20141.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20141.png" width="50%"/></p>
<p>Source: <a href="https://dnddnjs.gitbook.io/rl/chaper-8-value-function-approximatiobn/learning-with-function-approximator">https://dnddnjs.gitbook.io/rl/chaper-8-value-function-approximatiobn/learning-with-function-approximator</a></p>
<p>Slide: D. Silver 2015.</p>
</p></details></li>
<li>
<p>What are Policy Gradient Methods?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>Return a matrix of probabilities for each action via softmax</li>
<li>Exploration is handled automatically</li>
<li>We don't care about calculating accurate state values</li>
<li>Adjust the action probabilities depending on rewards</li>
</ul>
</p></details></li>
<li>What is a parametrized policy (belongs to Policy Gradient Methods)?<details><summary><b>Answer</b></summary><ul>
<li>Assign a probability density to each action</li>
<li>Such policies can select actions without consulting a value function</li>
<li>As we will see, the most sophisticated methods still utilize value functions to speed up learning, <em>but not for action selection (but this is considered as Actor-Critic)</em></li>
</ul></details>
</li>
<li>
<p>A stochastic Policy is a Policy Based RL. Why would we choose that?</p><details><summary><b>Answer</b></summary><p>
<p>At first it is important to note that stochastic does not mean randomness in all states, but it can be stochastic in some states where it makes sense. Usually maximizing reward leads to deterministic policy. But in some cases deterministic policies are not a good fit for the problem, for example in any two player game, playing deterministically means the other player will be able to come with counter measures in order to win all the time. For example in Rock-Cissors-Paper game, if we play deterministically meaning the same shape every time, then the other player can easily counter our policy and wins every game.</p>
<ul>
<li>I.e. at every point select one of the three actions with equal probability (0.33).</li>
<li>Therefore, a stochastic policy is desirable!</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20142.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20142.png" width="50%"/></p>
<p>Source: <a href="https://en.wikipedia.org/wiki/Rock_paper_scissors">https://en.wikipedia.org/wiki/Rock_paper_scissors</a></p>
</p></details></li>
<li>
<p>Another Policy Based Algorithm is REINFORCE with Monte Carlo Policy Gradient, explain it. (Also called Monte-Carlo Policy Gradient)</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20143.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20143.png" width="50%"/></p>
</p></details></li>
<li>
<p>REINFORCE, also called Vanilla Policy Gradient (VPG) can also work with Baseline, explain.</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>With the baseline, the variance is mitigated</li>
</ul>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20144.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20144.png" width="50%"/></p>
<p>Source Illustration: <a href="https://papoudakis.github.io/announcements/policy_gradient_a3c/">https://papoudakis.github.io/announcements/policy_gradient_a3c/</a></p>
</p></details></li>
<li>
<p>What are disadvantages of REINFORCE?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>REINFORCE is unbiased and will converge to a local minimum.</li>
<li>However, it is a Monte Carlo method which has its disadvantages.</li>
<li>Slow learning because of high variance (partially solved by baseline).</li>
<li>Impossible to implement online because of return.</li>
<li>we need to generate a full episode to perform any learning.</li>
<li>Therefore, can only apply to episodic tasks (needs to terminate).</li>
</ul>
</p></details></li>
<li>
<p>Provide a intersection diagram to show what kind of learning a RL Algo can have</p><details><summary><b>Answer</b></summary><p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20145.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20145.png" width="50%"/></p>
<p>Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 </p>
</p></details></li>
<li>
<p>Explain the Actor-Critic Method</p><details><summary><b>Answer</b></summary><p>
<p>Actor-Critic methods are a buildup on REINFORCE-with-baseline.</p>
<p>Two parts of the algorithm:</p>
<ul>
<li>Actor - updates the action selection probabilities</li>
<li>Critic - provides a baseline for actions’ value</li>
</ul>
<p>Actor Critic Method has the same goal as the Baseline for REINFORCE: It should remove the high variance in predicting an action. Instead of applying a Monte Carlo method for value function approximation, we perform an one-step Temporal Difference algorithm. </p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20146.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20146.png" width="50%"/></p>
<p>Source Illustration: Reinforcement Learning: An Introduction. A. Barto, R. Sutton. Page 325</p>
</p></details></li>
<li>
<p>What is the motivation for Monte Carlo Tree search rather than use Depth-First search?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>DFS fails with more complicated games (too many states and branches to go through)</li>
</ul>
</p></details></li>
<li>
<p>Explain Monte Carlo Tree search</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>TLDR: Random moves is the main idea behind MCTS.</li>
<li>Monte Carlo Tree Search is an efficient reinforcement learning approach to lookup for best path in a tree without performing a depth search over each path. Imagine an easy game like tik tak toe, where we have only 9 field so a game definitively ends after 9 actions. A depth-first search can easy find the best path through all actions. But for games like go or chess, where wie have a huge amount of states (1012010^12010120), it is impossible to perform a normal depth-first search. This is where the MCTS comes in.</li>
</ul>
<h2>Minimax and Game Trees</h2>
<ul>
<li>
<p>A game tree is a directed graph whose nodes are positioned in a game. The following picture is a game tree for the game tic tac toe.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20147.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20147.png" width="50%"/></p>
<p>Source: <a href="https://upload.wikimedia.org/wikipedia/commons/1/1f/Tic-tac-toe-full-game-tree-x-rational.jpg">https://upload.wikimedia.org/wikipedia/commons/1/1f/Tic-tac-toe-full-game-tree-x-rational.jpg</a></p>
</li>
<li>
<p>The main idea behind MCTS is to make random moves in the graph. For any given state we are no longer looking through every future combinations of states and actions. Instead, we use random exploration to estimate a state’s value. If we simulate 1000 games from a state s with random move selection, and see that on average P1 wins 75% of the time, we can be fairly certain that this state is better for P1 than P2.</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20148.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20148.png" width="50%"/></p>
</li>
</ul>
</p></details></li>
<li>
<p>Vanilla MCTS has a problem with the mere number of possible next states. How could it be improved?</p><details><summary><b>Answer</b></summary><p>
<ul>
<li>
<p>The Vanilla MCTS suffers in the point, that if there are a lot of possible actions in a state, only a few would lead to a good result. The chance to hit one of these good states is really low. An improvement for this could be to change these with something more intelligent. This is where <strong>MCTS Upper-Confidence Bound</strong> comes in. We can use the information after our backpropagation. The states currently keep information about their winning chance. We do actual move selection by visit count from root (max or as proportional probability). Idea is that by UCT "good" nodes were visited more often.</p>
</li>
<li>
<p>But of course, a balance between exploration and exploitation is quite important. We need to make sure that we explore as many states as possible, especially in the first few rounds. The following formula can be used to calculate the selection of the next node:</p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20149.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20149.png" width="50%"/></p>
</li>
<li>
<p>This approach can be leveraged by using a neural net which does value function approximation. So we do not need to explore a lot of states because the NN tells us, how a good a state is.</p>
</li>
</ul>
</p></details></li>
<li>
<p>How does AlphaGo Zero work?</p><details><summary><b>Answer</b></summary><p>
<p><a href="https://www.youtube.com/watch?v=MgowR4pq3e8">How AlphaGo Zero works - Google DeepMind</a></p>
<p><img alt="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20150.png" height="50%" src="data/5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20150.png" width="50%"/></p>
<p>Source Illustrations: David Silver</p>
</p></details></li>
<li>
<p>Shorten the algorithm of AlphaGo Zero</p><details><summary><b>Answer</b></summary><p>
<ol>
<li>Use current network and play some games to create training data. use MCTS (Monte Carlo Tree Search) for that. Completly focused on self-training.</li>
<li>Use this data to retrain the network (Fix and target values). We train a policy and a value network.</li>
<li>Now we have a slightly better network, repeat 1 + 2.</li>
</ol>
<p>Notes: </p>
<ul>
<li>AlphaGo Zero uses ResNet</li>
<li>Takes the last 8 images of the board into account</li>
</ul>
<p>Output of the feature vector:</p>
<ul>
<li>Value vector:</li>
<li>Policy vector: Probability distribution for all actions</li>
<li>Define Actor-Critic</li>
<li>Learnt value function</li>
<li>Learnt policy</li>
<li>Value function “helps” learn the policy</li>
</ul>
</p></details></li>
</ol start=245>