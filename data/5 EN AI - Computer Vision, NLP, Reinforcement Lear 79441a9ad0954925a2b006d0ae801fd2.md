# 5 EN: AI - Computer Vision, NLP, Reinforcement Learning

## Computer Vision

- What are difficulties about classifying an image
    - Intra-class variation (Different breed of dogs)
    - Shape Variation (Size of dogs)
    - Illumination variation
- What kind of feature did people use before 2012?
    - Color Histogram
    - Key Point descriptor

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled.png)

    - Histogram of gradients

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%201.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%201.png)

    - Bag of words model

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%202.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%202.png)

- Name popular benchmark datasets
    - MNIST
    - CIFAR-10 (Canadian Institute for Advanced Research)
    - IMAGENET (22 Categories)
- Why haven't been CNNs used before?
    - Requires huge datasets
    - and immense computational power
- Name the most popular loss functions for regression and classification
    - Regression: L2 loss (aka mean squared error), L1 loss (aka mean absolute error)
    - Classification: cross-entropy loss, hinge loss (used in SVMs)
- Explain cross-entropy loss

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%203.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%203.png)

    - M - number of classes
    - y_i,c - binary indicator (0 or 1) if class label c is the correct classification for observation o
    - (p_i,c) - the predicted probability that observation i is of class c

    Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 

    Actually the log function would be mirrored on the x axis and a small value of the predicted probability would be < 0, but since our goal is to minimize it we have to turn the function around and we do that by multiplying -1.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%204.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%204.png)

    Source: [ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

    Therefore, eventually the network learns to output high probabilities for the correct class and low probabilities for the incorrect ones.

- What is a dense layer?
    - A dense layer (also called a fully connected layer) is a layer where each input is connected to every output, i.e. what we modelled
- What is the standard activation function to use?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%205.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%205.png)

    Source: [https://cdn-images-1.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png](https://cdn-images-1.medium.com/max/1200/1*ZafDv3VUm60Eh10OeJu1vw.png)

- What is the output of the conv filter when it slides over an image?
    - Its dot product
- What does a pooling layer?
    - Used for dimensionality reduction
    - Slide with a stride > 1
    - Instead of a dot product of a conv filter the ouput is the max (there are others like average)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%206.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%206.png)

        Source Illustration: Fei-Fei Li & Justin Johnson & Serena Yeung

- Impression of a ConvNet from the inside

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%207.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%207.png)

    Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 

- What kind of level feature does a ConvNet encode in the beginning and at the end?
    - The early layers encode most basic feature, whereas deeper layers pick up more specfific ones (e.g. eyes of a face)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%208.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%208.png)

    Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 

- How did they improve neural networks further?
    - Smaller conv filter but deeper networks
- What is the idea of Inception Modules?

    Create a good local network in a module and stack the modules on top of each other.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%209.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%209.png)

    Source Illustration : Fei-Fei Li & Justin Johnson & Serena Yeung

- How do Inception Modules work?
    - Different techniques are applied independently and concatenated afterwards. It is not clear in the beginning which of the approaches will work, but one of these just might.
- What is the problem of Inception Modules?
    - It creates a huge computational complexity due to a huge increase of the ouput depth
    - Example:

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2010.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2010.png)

        Source Illustration : Fei-Fei Li & Justin Johnson & Serena Yeung

- How can the problem of Inception Modules be solved?
    - By applying 1x1 convolutions at the right position, such as prior to a 5x5 or as successor of a 3x3 max pooling

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2011.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2011.png)

        Source Illustration : Fei-Fei Li & Justin Johnson & Serena Yeung

    Applying 1x1, leads to:

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2012.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2012.png)

    Source Illustration : Fei-Fei Li & Justin Johnson & Serena Yeung

- What is the goal/purpose of a 1x1 conv filter?
    - It reduces the depth by combining feature maps

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2013.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2013.png)

    Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- What are the three parts of the GoogleNet?
    - Stem Network
    - Stacked Inception Modules
    - Classifier Outputs
- What is the idea of Deep Residual Learning?
    - Learn difference to the identity map instead of map itself

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2014.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2014.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- What is Transfer Learning?
    - Transfer learning focuses on storing knowledge gained for a specific problem and applying it to a different but related problem”

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2015.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2015.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- What different challenges are solvable with computer vision?
    - Object localization: output the class of an image (object) but also output the position of the object (e.g. in form of a bounding box)
    - Object detection: classify multiple objects on an image as well as their locations (bounding boxes)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2016.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2016.png)

    - Semantic segmentation: label the individual pixels that belong to different categories (don’t differentiate between instances). Widely used in self-driving cars.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2017.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2017.png)

    - Instance segmentation: detect multiple objects, then label the pixels belonging to each of the detected objects. In contrary to Semantic segmentation: Differentiate between instances!

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2018.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2018.png)

        Source Illustration: Fei-Fei Li & Justin Johnson & Serena Yeung

- What is the idea for object localization?
    - We provide the class we are looking for and the NN tries to locate it. We still need to train a classifier for scoring the different classes, because there could be more than one class in the picture.  If the picture showed a dog and cat, and we specify the class as cat, we would only get the bounding boxes for the cat. (or all cats)
    - train a regression output for the box coordinates

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2019.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2019.png)

        Source Illustration: Fei-Fei Li & Justin Johnson & Serena Yeung

- What are the different loss functions for object localization?
    - Find every object and its bounding boxes in the image.
    - softmax loss for classification
    - L2 loss (MSE) for bounding boxes

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2020.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2020.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- What is pose estimation?
    - The human body gets split up in different parts, such as left hand, head, right foot. The NN tries to figure out these parts in the image.
- What's the ouput of a NN doing pose estimation?
    - A vector holding different parts and their coordinates.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2021.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2021.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- What is the loss of pose estimation?
    - L2 loss

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2022.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2022.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- What are some approaches to do object detection (which is why harder than object localization?)
    - We do not know beforehand how many object there will be
    - Therefore, we cannot use the same approach as in localization (where only one object was to be found and classified).
    - A brute-force approach: apply a CNN to many different subsets of an image and classify each as an object or background.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2023.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2023.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- Why is the brute force approach bad?
    - checking every crop is computationally costly
- What are other algorithms to do object detection?
    - Selective Search for Object Recognition, which does a region proposal for, for example, 2000 parts of the image. The regions are then passed to a normal cnn. This cnn is called Region-CNN (R-CNN)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2024.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2024.png)

        Source: [https://cdn-images-1.medium.com/max/1000/1*REPHY47zAyzgbNKC6zlvBQ.png](https://cdn-images-1.medium.com/max/1000/1*REPHY47zAyzgbNKC6zlvBQ.png)

- R-CNNs are slow, thats why Fast R-CNN were born. What's the difference?
    - Fast R-CNN make use of the feature map that is created when an image is fed to the NN. Based on this feature map, regions are derived, they are wrapped into identical size by using pooling operations and then passed to a fully connected layer.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2025.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2025.png)

        Source: [https://cdn-images-1.medium.com/max/1000/1*0pMP3aY8blSpva5tvWbnKA.png](https://cdn-images-1.medium.com/max/1000/1*0pMP3aY8blSpva5tvWbnKA.png)

- What does Fast R-CNN do?
    - Uses a predefined algorithm for creating region proposals, and a Region Proposal Network (RPN) to predict proposals from features.
- What are MobileNets and what do they use?
    - On mobile we require nn to process in real time, therefore the number of computations has to be reduced. The engine for this is called depthwise separable convolutions.
- Explain the idea of Depthwise Separable Convolutions

    Key to understand is to know that the dot product calculation is expensive.

    - A standard convolution does a dot product of all channels

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2026.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2026.png)

        Source: [https://cdn-images-1.medium.com/max/800/0*rbWRzjKvoGt9W3Mf.png](https://cdn-images-1.medium.com/max/800/0*rbWRzjKvoGt9W3Mf.png)

    - The idea of depthwise is to reduce the dimension of the channels before the dot product is applied.

        1.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2027.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2027.png)

        Source: [https://machinethink.net/images/mobilenets/DepthwiseConvolution@2x.png](https://machinethink.net/images/mobilenets/DepthwiseConvolution@2x.png)

        2.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2028.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2028.png)

        Source: [https://machinethink.net/images/mobilenets/PointwiseConvolution@2x.png](https://machinethink.net/images/mobilenets/PointwiseConvolution@2x.png)

- What impact do the two parameters have on MobileNets?

    The parameters can be used to decrease latency with the drawback to sacrifice accuracy.

    - Width multiplier: How thin or thick the feature map gets. (Must be dimensions going into the back)
    - Resolution multiplier: Parameter for the size of the resolution of input and feature map.
- Image showing the dropping accuracy with less computation steps

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2029.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2029.png)

    Source: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, A. Howard et al.

    Figure 5 shows the trade off between ImageNet Accuracy and number of parameters for the 16 models made from the cross product of width multiplier α∈{1,0.75,0.5,0.25} and resolutions {224,192,160,128}

- What's the difference between semantic segmentation and instance segmentation?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2030.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2030.png)

    Source: [http://vladlen.info/wp-content/uploads/FSO-1.jpg](http://vladlen.info/wp-content/uploads/FSO-1.jpg)

- What is Panoptic Segmentation?
    - Combines segmenting  stuff (background regions, e.g. grass, sky) and Things (Objects with well defined shape, e.g. person, car). Therefore it combines Semantic and Instance Segmentation

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2031.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2031.png)

        Source: Panoptic Segmentation A. Kirillov et al.

- How does Panoptic Segmentation work?
    - Algorithm must label every pixel in the found objects but also label every pixel in the background "stuff". We map every pixel i to a tupel:

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2032.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2032.png)

        Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 

- Name datasets to train Semantic or Instance Segmentation models
    - COCO (Common Objects in Context)
    - BDD100K (A Large-scale Diverse Driving Video Database)
    - Mapillary Vistas Dataset (Data from cities all over the world)
    - Cityscapes (50 cities (49 german cities + Zurich), various seasons (spring, summer, fall))
    - The KITTI Vision Benchmark Suite
- What is the Loss Function for Semantic Segmentation?
    - Cross-entropy on a per-pixel basis
    - Main idea: evaluate prediction loss for every individual pixel in an image and then take the average of these values for a per-image loss. This is called Pixel-Wise Cross Entropy Loss.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2033.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2033.png)

        Source: [https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-24-at-10.46.16-PM.png](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-24-at-10.46.16-PM.png)

- What is the problem for the Loss Function for Semantic Segmentation?
    - Here: every pixel is treated equally (because of the averaging). This can lead to some issues related to imbalanced classes.
- How can you evaluate the performance of semantic segmentation?
    - For image segmentation we have Intersection
    over Union (IoU) metric.
    - IoU measures what percentage of pixels in
    target and prediction images overlap.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2034.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2034.png)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2035.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2035.png)

        Source: [https://www.jeremyjordan.me/evaluating-image-segmentation-models/](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)

- What approaches exist for Semantic Segmentation?
    - Sliding Windows (Inefficient and unusable in practice)
    - Fully Convolutional Network for Segmentation (still inefficent, because no dimensionality reduction is applied) //Recall: When a CNN gets deeper we want to have more features but of lower dimension.
    - Image Segmentation Network

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2036.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2036.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- How can we incorporate dimensionality reduction to a CNN for Semantic Segmentation but still have the output of the original size?
    - Design network with downsampling and upsampling (same principal as autoencoder)
- How can downsampling be done?
    - Pooling or Strided Convolutions

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2037.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2037.png)

        Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019

- How can upsampling be done?
    - Unpooling (parameter-free), e.g. "max-unpooling"
    - Transpose Convolution (trainable upsampling)
- What types of Unpooling do you know?
    - 1:

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2038.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2038.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

    - **Bed of Nails** (Fill rest of the squares with zeroes)
    - **Nearest Neighbour** Unpooling. A KIND OF INVERSION OF AVERAGE POOLING OPERATION (Fill rest of the squares with the same number)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2039.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2039.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

    - **Max Unpooling as Max Pooling Inverse**
        - We remember which element was the max value and recover this in the new output

            ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2040.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2040.png)

            ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2041.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2041.png)

            Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- Why is the value of the unpooled output slighlty different than it was before?
    - For this a transpose convolution is used and the scalar of the reduced output is multiplied by the filter. Important: The values in the filter are trained by the network and not predefined.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2042.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2042.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- How does the checkerboard artifact evolve?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2043.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2043.png)

    Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- Convolution and its transpose side-by-side

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2044.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2044.png)

    Source: [https://github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic)

- What are other names for transpose convolution?
    - Upconvolution
    - Backward strided convolution
    - Fractionally strided convolution
    - Deconvolution - this one is particularly dangerous because deconvolution is a well-defined mathematical operation which **is not the same** as transpose convolution.
- What is the architecture idea for Instance segmentation?
    - There are two independent branches trying to detect a) the categories and the bounding boxes and b) the second branch classifies each pixel whether its an object or not (also called mask prediction).
    - ⇒ This procedure happens due to performance reason to the region proposals.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2045.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2045.png)

        Source: [https://github.com/vdumoulin/conv_arithmetic](https://github.com/vdumoulin/conv_arithmetic)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2046.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2046.png)

        Source: [https://cdn-images-1.medium.com/max/2000/1*lMEd6AcDmpH0mDzBHyiERw.png](https://cdn-images-1.medium.com/max/2000/1*lMEd6AcDmpH0mDzBHyiERw.png)

    - Note: The idea of mask prediction can also be used for Pose Detection?
- Give a summary of Encoder-Decoder Networks

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2047.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2047.png)

    Source: [https://saytosid.github.io/images/segnet/Complete architecture.png](https://saytosid.github.io/images/segnet/Complete%20architecture.png)

- What is the motivation for Autoencoders?

    We want them to learn efficient and pertinent data encodings of the data.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2048.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2048.png)

    Source: [https://cdn-images-1.medium.com/max/1600/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png](https://cdn-images-1.medium.com/max/1600/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png)

- Give a definition of Autoencoders
    - Autoencoder is a type of an encoder-decoder network where target space is the same as the input space.
- Provide Characteristics of Autoencoders
    - Data does not have to be labelled
    - Latent feature space is usually of lower dimension than the input feature space
    - L2 as loss function
- Where are autoencoders used?
    - All applications of AEs utilize the learned latent features for further purposes.
    - Use for dimensionality reduction
    - Important: AEs are quite data-specific. They will only work well on data that is similar to that on which they were trained. Extreme example: an AE trained on images will not work well on time-series data.
    - Not really good for compression of data
    - Denoising of Input data

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2049.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2049.png)

        Source: [https://cdn-images-1.medium.com/max/1600/1*G0V4dz4RKTKGpebeoSWB0A.png](https://cdn-images-1.medium.com/max/1600/1*G0V4dz4RKTKGpebeoSWB0A.png)

    - Watermark Removal

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2050.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2050.png)

        Source: [https://www.edureka.co/blog/autoencoders-tutorial/](https://www.edureka.co/blog/autoencoders-tutorial/)

    - Image Coloring

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2051.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2051.png)

        Source: [https://www.edureka.co/blog/autoencoders-tutorial/](https://www.edureka.co/blog/autoencoders-tutorial/)

- How could we initialize our neural network quite well?
    - By passing the features of the autoencoder to the weight initialization

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2052.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2052.png)

        Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019

- How can Autoencoders be used for Anomaly detection?
    - Main idea: let’s say we have a well-trained autoencoder (which means it can reconstruct the data it was trained on without too much error):
        - If it works well on the new input, i.e. reconstruction error is low, we can assume that the input is of the normal class.
        - If the reconstruction error is high, we should think that the input is of an ‘unseen’ class, i.e. it is an anomaly.
- Potential applications for Autoencoders
    - Detect if input data is of the class that our model was trained on or not.
    - Detect outliers.
    - Fraud analytics.
    - Monitoring sensor data
- Give an idea of Generative Models
    - It’s not about predicting and classifying things, it’s

        about learning some underlying hidden structure of the
        data at hand.

        We want to train a model distribution pθ ,parameterized by our choice, that would fit or resemble pdata.

        If we can successfully obtain this trained model pθ, it in fact means that we know the underlying structure of the data and we can do a lot of interesting things with it.

        → We start by an empirical data distribution

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2053.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2053.png)

        Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- Which famous models belong to generative models?
    - Variational Autoencoder
    - Markov Chain (e.g. Boltzmann Machine)
    - GANs
- What are the flavours of Generative Models?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2054.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2054.png)

    Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- On which flavour focuses VAEs and GANs?
    - Variational Autoencoder → Explicit
    - GANs → Implicit
- What are Variational Autoencoder (VAE)?
    - Probabilistic extension of normal autoencoder
    - Also considered as latent variable model
    - Make it learn latent parameters that describes the probability distribution of the data
        - We start with a standard normal distribution and a prior p_theta(z)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2055.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2055.png)

    Source: [http://kvfrans.com/content/images/2016/08/vae.jpg](http://kvfrans.com/content/images/2016/08/vae.jpg)

- What is the loss function of VAE?
    - KL (Kullback Leibler) divergence between the learned latent distribution and its prior distribution. This forces the network to learn latent features that follow the prior distribution
    - Reconstruction loss - just as in previous autoencoders, it forces the decoder to match the input.

        *"Kullback–Leibler divergence (also called relative entropy) is a measure of how one probability distribution is different from a second, reference probability distribution"*

- What is the key difference between Autoencoder and Variational Autoencoder?
    - Autoencoders learn a “compressed representation” of input (could be image,text sequence etc.) automatically by first compressing the input (encoder) and decompressing it back (decoder) to match the original input. The learning is aided by using distance function that quantifies the information loss that occurs from the lossy compression. So learning in an autoencoder is a form of unsupervised learning (or self-supervised as some refer to it) - there is no labeled data.
    - Instead of just learning a function representing the data ( a compressed representation) like autoencoders, variational autoencoders learn the parameters of a probability distribution representing the data. Since it learns to model the data, we can sample from the distribution and generate new input data samples. So it is a generative model like, for instance, GANs.

    [](https://www.quora.com/Whats-the-difference-between-a-Variational-Autoencoder-VAE-and-an-Autoencoder)

- Does GANs use a probability distribution?
    - Nope
- What is the idea of GANs?
    - Train an NN that produces images based on an input of some images. This part is called Generator Network. The goal of the Generator Network is, to improve its ability to produce images so that the discriminator will fail.
    - The second actor is the Discriminator Network. Its task is to differentiate between the fake image from the generator and real images from the training set.
- How does the loss function look like?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2056.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2056.png)

    Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019

- What is a drawback of the loss function and how to solve it?

    When actually training the generator, it was found that the objective function does not work very well because of gradient plateau (flat gradients means very slow if any training).

    Solve it by:

    - For this reason the generator objective function is “flipped”.
    - An intuitive interpretation is that now instead of minimizing the probability of the discriminator being correct, we maximize the probability of it being wrong.
- What do Vannila GANs use?
    - Only fully connected layer but no CNN
- When GANs use CNNs, the discriminator uses a normal cnn. What about the generator?
    - it uses an upsampling CNN with transpose convolution operations (Recall Image segmentation problems)
- How does deepfakes video work?
    - Rely on GANs for data generation.
    - Combines existing videos with source videos to create new, almost indistinguishable videos of events that are actually completely artificial

## NLP

- What is the idea of weight initialization?
    - Using better weight initialization methods leads to a better gradient transport through the network and a faster convergence
- What are different approaches for weight initialization?
    - Weights = 0: Bad Idea - no training at all

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2057.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2057.png)

        Source: Source: [https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization](https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization)

    - Random initialization with mean at 0 and small variance. Seems like it works quite well, although it heavily depends on the used distribution function

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2058.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2058.png)

        Source: [https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization](https://github.com/Intoli/intoli-article-materials/tree/master/articles/neural-network-initialization)

    - Problem:
    - Different variance values for weight initialization lead to vastly different activations, from vanishing to exploding gradients. Linear activation function

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2059.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2059.png)

        Source: [https://intoli.com/blog/neural-network-initialization/](https://intoli.com/blog/neural-network-initialization/)

- Why is it important to keep the variance of gradient and activation values of each layer constant?
    - Otherwise it would lead to vanishing or exploding gradients, which leads to problems in training.
- For linear functions it is easier to have constant variances

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2060.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2060.png)

    Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019

- Constant variances for NON linear, like relu looks like:

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2061.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2061.png)

    Source: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. He et. al.

- What is Batch Normalization and why should it be applied?

    Why: Covariate Shift ends in bad performance 

    But the problem appears in the intermediate layers because the distribution of the activations is constantly changing during training. This slows down the training process because each layer must learn to adapt themselves to a new distribution in every training step. This problem is known as internal covariate shift.

    [Batch normalization: theory and how to use it with Tensorflow](https://towardsdatascience.com/batch-normalization-theory-and-how-to-use-it-with-tensorflow-1892ca0173ad)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2062.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2062.png)

    That simply means that the datapoints can vary extremly which forces the intermediate layers to readjust.

    Where and on what do we do Batch Norm?

    - Usually for the intermediate layers, but it can also be applied on the input layer (taking the raw values). It is not applied on the values itself but on the result of the x vector times the weight vector w, which is z. Z is passed to the activation function and that's why it has to be normalized beforehand. (Sometimes the result of the activation is normalized but this doesn't make sense to me)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2063.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2063.png)

    Source: Andrew Ng, Deep Learning, Coursera

- What is the idea of Batch Normalization?
    - BN prevent a neural net from exploding or vanishing gradient and reducing learning time due to internal covariate shift. It forces the activations to have mean 0 and unit variance by standardizing it.
- What are the steps for Batch normalization?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2064.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2064.png)

- Should Batch Normalization always be applied?

    No, with the following explanation:

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2065.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2065.png)

- What is the idea of Dropout?

    The idea is to prevent an NN from overfitting by removing a specific percentage of neurons from a layer. Often used percentage is between 0.5 and 0.25.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2066.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2066.png)

    Source: [https://static.commonlounge.com/fp/original/aOLPWvdc8ukd8GTFUhff2RtcA1520492906_kc](https://static.commonlounge.com/fp/original/aOLPWvdc8ukd8GTFUhff2RtcA1520492906_kc)

- Explain Cosine Similarity
    - Measures similarity between two word vectors
    - Measures the angle of two words rather than their actual distance to each other.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2067.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2067.png)

        Source: [https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/cosine/](https://neo4j.com/docs/graph-algorithms/current/labs-algorithms/cosine/)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2068.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2068.png)

        Source: [https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/eb9cd609-e44a-40a2-9c3a-f16fc4f5289a.xhtml](https://www.oreilly.com/library/view/statistics-for-machine/9781788295758/eb9cd609-e44a-40a2-9c3a-f16fc4f5289a.xhtml)

- Explain Bleu Score an when to use it

    **BLEU** (**bilingual evaluation understudy**) is an algorithm for evaluating the quality of text which has been [machine-translated](https://en.wikipedia.org/wiki/Machine_translation) from one [natural language](https://en.wikipedia.org/wiki/Natural_language) to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" 

    Scores are calculated for individual translated segments—generally sentences—by comparing them with a set of good quality reference translations. Those scores are then averaged over the whole [corpus](https://en.wikipedia.org/wiki/Text_corpus) to reach an estimate of the translation's overall quality. Intelligibility or grammatical correctness are not taken into account

    BLEU's output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1 representing more similar texts. Few human translations will attain a score of 1, since this would indicate that the candidate is identical to one of the reference translations. For this reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional reference translations will increase the BLEU score

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2069.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2069.png)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2070.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2070.png)

    Source: [https://cloud.google.com/translate/automl/docs/evaluate#:~:text=BLEU (BiLingual Evaluation Understudy) is,of high quality reference translations](https://cloud.google.com/translate/automl/docs/evaluate#:~:text=BLEU%20(BiLingual%20Evaluation%20Understudy)%20is,of%20high%20quality%20reference%20translations).

    1. Berechne Precision für jedes N-Gram (wähle min von ref und cand)
    2. Multipliziere über alle n-gram_precision werte
    3. Wende Penalty für kurze Sätze an

- What are the disadvantages of the Bleu Scores?
    - No distinction between content and functional words (ESA - NASA).
    - Weak in capturing the meaning and grammar of a sentence
    - Penalty for short sentences can have strong impact

- Why do we need RNNs and not normal Neural Nets?
    - Inputs and outputs can be different lengths in different examples.
    - Doesn't share features learned across different positions of text
- What's the architecture of a Vanilla RNN?
    - a is usually referred as the hidden state, whereas I just see it as the activation from the neuron.
    - x^<t> is a word of the sentence at position t (same as at time t)
    - As an activation function mostly tanh/Relu is used, but only for the hidden state (a)
    - For y_hat we usually use a sigmoid function
    - a<0> (hidden state) is initialized with zero.
    - g ist the function (It's actually just the layer and what it does is it has a weight vector for incoming a that is multiplied with the activation value and a weight vector for the new word x that is multiplied with the new word plus a bias), this results in a<i>. g also has a weight vector  for calculating y. It takes that vector and multiplies it with the previously calculated a<0> plus a bias. The result is the word at position <t> bzw. y<t>

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2071.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2071.png)

        Source: Andrew Ng, Deep Learning, Coursera

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2072.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2072.png)

- What is the loss of the RNN?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2073.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2073.png)

    Source: Fei-Fei Li & Justin Johnson & Serena Yeung

- What are problems of Vanilla RNNs?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2074.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2074.png)

    - not good in learning long-term dependencies
- What are LSTMs and why are they needed?
    - Vanilla RNNs are not good in learning long-term dependencies. LSTMs can be seen as a fancier family of RNNs.
- Of which components is a LSTM composed?
    - **Cell State:** C(t) Internat cell state (memory)
    - **Hidden State:** External hidden state to calculate the predictions
    - **Input Gate:** Determines how much of the current input is read into the cell state
    - **Forget Gate:** Determines how much of the previous cell state is sent into the current cell state
    - **Output Gate:** Determines how much of the cell state is output into the hidden state

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2075.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2075.png)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2076.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2076.png)

        Source: [https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

- What is the sigmoid function for?
    - The output is a value between 0 and 1. If the network thinks something should be forgotten than it tends to be 0, if it wants to update than it is 1. This is because the output is mostly used for multiplication.
- What is the tanh function for?
    - tanh function squishes values to always be between -1 and 1. This helps preventing the vectors of having too large numbers.
- What is the cell state for?
    - Memory of the network. As the cell state goes on its journey, information gets added or removed to the cell state via gates.
- What does the forget gate do?
    - Input: Hidden + word (xi)
    - Based on the hidden state, the forget gate decides whether to update or if the information should be thrown away (but applied on the hidden state of the previous time stamp, so it kind of cleans up the past)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2077.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2077.png)

        Source: [https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

- What does the input gate do?
    - Input: hidden + word (xi)
    1. Pass both to sigmoid function to decide which values should be updated
    2. Pass both to tanh to squish values between -1 and 1 
    3. Multiply sigmoid output with tanh output, to keep important values of tanh output

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2078.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2078.png)

        Source: Source: [https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

- How is the cell state computed?
    - Input: previous cell state + forget gate output + input gate output
    1. Multiply prev cell state pointwise with forget gate output
    2. Add input gate output to the result
- What does the output gate do?
    - Input: previous cell state + prev hidden state + word(xi)
    - Calculates the new hidden state by multiplying cell state with the output of sigmoid output gate output

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2079.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2079.png)

        Source: [https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

- How does GRU work?
    - Newer generation of RNN
    - got rid of cell state
    - use hidden state to transport information
    - little speedier to train then LSTM’s
    - Reset gate and update gate

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2080.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2080.png)

        Source: [https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

    - Let the network learn when to open and close the gates, i.e. update the “highway” with new information.
    - Therefore, instead of updating the hidden state at every RNN cell, the network can learn itself when to update the hidden state with new information.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2081.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2081.png)

    Source: [https://wagenaartje.github.io/neataptic/docs/builtins/gru/](https://wagenaartje.github.io/neataptic/docs/builtins/gru/)

    1. Reset Gate: 
        - Input: Weight Matrix W, prev hidden state, and word (x<t>)
        1. It first computes the reset value. That indicates which value of the input are important and which aren't (recall sigmoid value → 0)
        2. Pass the function of multiplication of prev hidden state and reset vector to the tanh function. This deactivates some prev hidden state values
    2. Update Gate: 
        - Input: Weight, prev hidden state, x

            ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2082.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2082.png)

    3. Compute Output

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2083.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2083.png)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2084.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2084.png)

    I like this illustration more:

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2085.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2085.png)

    Quote: *"Similar to normal RNNs the input gets multiplied by the weight matrix and is added to a hidden layer. However here the input is added to h˜ . The r is a reset switch which represents how much of the previous hidden state to use for the current prediction. Coincidentally there is also a neural network for this reset switch which learns how much of the previous state to allow for predicting the next state. z represents whether to only use the existing hidden state h or use its sum with h~(new hidden state) to predict the output character."*

    Source: [https://medium.com/datadriveninvestor/lstm-vs-gru-understanding-the-2-major-neural-networks-ruling-character-wise-text-prediction-5a89645028f0](https://medium.com/datadriveninvestor/lstm-vs-gru-understanding-the-2-major-neural-networks-ruling-character-wise-text-prediction-5a89645028f0)

- What is the idea of Attention models?
    - The main idea of the attention mechanism is to imitate the human behaviour of text processing by "reading" a chunk of the input sentence and process it instead of the whole sentence.
    - The more words the encoder processed, the less information about the single words is contained in the vector. Attention models try to bypass this by save each output vector ht from a state. In contrast to vanilla Seq2Seq-Models, another processing layer between encoder and decoder was added which calculates a score for each ht. The scores indicate how much attention the decoder should pay on a specific ht.
    - → Goal: Increase performance for long sentences
- Describe the algorithm of Attention Models

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2086.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2086.png)

- Shorten the algorithm in your own words
    1. Choose window of words you want to incorporate its attention
    2. Save the hidden states of them and compute a score for these hidden states
    3. Apply softmax on the scores to get a probability distribution, resulting in attention weights
    4. Compute the context vector by attention weights times hidden states
    5. Take context vector and prev hidden state to compute the next word

- What is the advantage we have with ConvNets for NLP?
    - We can compute different convolutions in parallel for our input vector

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2087.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2087.png)

        Source: [https://machinelearningmastery.com/best-practices-document-classification-deep-learning/](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

- How does CNN for NLP work?
    - Use filter for multiple words (Similar approach to n-gram)
    - Vanilla CNNs can be used in One-to-One, Many-to-One or Many-to-Many Architecture.
    - Used in Text Classification, Sentiment Analysis and Topic Categorization
    - Problem: Looses the relation information between words, caused by max pooling layers.
- How can CNNs be combined with Seq2Seq Models?
    - Each layer has two Matrices
    - W -  Kernel for convolving input
    - V - used for GRU computation
    - GRU calculation decides based on the sigmoid outcome how much of the conv output is propagated to the next level.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2088.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2088.png)

- Describe the basic steps of Speech Recognition
    - Transform sound wave to spectrogram using fourier transformation
    - Spectogram is splitted into multiple parts, e.g. 20 ms blocks, resulting in 50 block for a second. A second usually contains between 2 and 3 words.
    - Problem: Get rid of repeated characters and separate them into words (CTC = Connectionist Temporal Classification)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2089.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2089.png)

        Source: Hannun, "Sequence Modeling with CTC", Distill, 2017.

- How can the problem of removing repeated characters be solved?
    - Apply Beam Search and take the most likely output

- Explain a simple approach for Speech to Text

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2090.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2090.png)

- Explain Backpropagation through time (BPTT)
    1. Gather loss for all outputs and at every timestamp and sum them up.  (but for each state)

        Loss = Sum (y_t - y_t_hat)

    2. Apply "normal" backpropagation to each state, keep in mind the partial derivative’s 

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2091.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2091.png)

        Source: [http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)

- What is Truncated Backpropagation through Time?
    - The longer the sequence is the harder it is to calculate the gradients -> Solution: truncated backprop
    - BPTT is periodically on a fixed number of timesteps applied

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2092.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2092.png)

        Source: [https://srdas.github.io/DLBook/RNNs.html](https://srdas.github.io/DLBook/RNNs.html)

- What is a Language Model?
    - It gives you a probability of a following sequence of words/characters for a given word. P(Y) = p(y1 ,y2 ,y3 ,...,yn )
    - This effectively estimates the likelihood of different phrases (in a language).

    → Simpler models may look at a context of a short sequence of words, whereas larger models may work at the level of sentences or paragraphs. 

    Useful for:

    - Speech Recognition
    - Machine translation
    - Part-Of-Speech (POS) tagging
    - Optical Character Recognition (OCR)

- Explain the idea of N-Grams
    - Coalition of n words to "one"
    - 1-gram = Unigram
    - 2-gram = Bigram
    - Can be used to calculate the probability of a given word being followed by a particular one.
    - P(x1, x2, ..., xn) = P(x1)P(x2|x1)...P(xn|x1,...xn-1)
- Define One-Hot Encoding
    - Encoding categorical variable so that NN can handle it. We need a vocabulary V with all words in it.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2093.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2093.png)

        Source: [https://tensorflow.rstudio.com/guide/tfdatasets/feature_columns/](https://tensorflow.rstudio.com/guide/tfdatasets/feature_columns/)

- What are pros and cons of one hot encoding for text data?

    **Pros:**

    - Can be used on different stages for example words or letters
    - Can represent every text with a single vector

    **Cons:** There are a few problems with the one-hot approach for encoding:

    - The number of dimensions (columns in this case), increases linearly as we add words to the vocabulary. For a vocabulary of 50,000 words, each word is represented with 49,999 zeros, and a single “one” value in the correct location. As such, memory use is prohibitively large.
    - The embedding matrix is very sparse, mainly made up of zeros.
    - There is no shared information between words and no commonalities between similar words. All words are the same “distance” apart in the n-dimensional embedding space.
- Example of one hot encoding

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2094.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2094.png)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2095.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2095.png)

    Source: [https://programmersought.com/article/64902775751/](https://programmersought.com/article/64902775751/)

- What is the idea of word embeddings?
    - Encoding for meaning and similarity of words in form of a vector that exists of latent dimensions (latent feature space)
    - Main motivation: we want to capture some intrinsic “meaning” of a word and be able to draw comparisons between words
    - Example of latent feature space:

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2096.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2096.png)

        Source: [https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf](https://medium.com/@jayeshbahire/introduction-to-word-vectors-ea1d4e4b84bf)

- What are word embeddings used for?
    - Semantic Analysis
    - Named Entity Recognition
    - Weights initilization
- What is a problem with word embeddings?
    - Need huge corpora to train, that's why mostly pre-trained embeddings are used
- What is the basic approach of word embeddings?
    - Words that appear in similar contexts (close to each other) are likely to be semantically related.
    - If we want to “understand” the meaning of a word, we take information about the words it coappears with.
- How does the training process look like?

    We want to have p(context|wt), where wt is the word we are looking at and context the surrounding words.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2097.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2097.png)

    1. Used Loss function: -log[p(context|wt)] [//Somewhat](//somewhat) like cross entropy loss function
    2. Optimize NN based on Used Loss Function
    3. If we can predict context we have good embedding
    4. delete classifier keep embeddings

    Source Illustration: [https://laptrinhx.com/get-busy-with-word-embeddings-an-introduction-3637999906/](https://laptrinhx.com/get-busy-with-word-embeddings-an-introduction-3637999906/)

- What was an early attempt of word-embeddings?
    - Neural probabilistic Language Models
    - → Generate joint probability functions of word sequences
- What were shortcomings of Neural Probabilistic?
    - It was only looking at words before but not at words after the predicted one
- How does Skip-Gram work?

    blue is the given word to predict the surroundings → + + + C + + + 

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2098.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2098.png)

    Source: [http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

- How does word2vec work?

    Idea relies on estimating word co-occurences.

    For Skip-gram:

    1. Select Windows size
    2. Select a word wt
    3. For a selected word wt in our training corpus look a randomly selected word within m positions of wt.
    4. train to predict correct context words
    5. Predict by softmax classifier

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2099.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%2099.png)

- What is a problem with word2vec?
    - At each window the normalization factor over the whole vocabulary has to be trained (e.g. 10.000 dot products) → computationally too expensive
- How can the problems of word2vec be solved?
    - Applying negative sampling
- What is negative sampling

    Main idea: we want to avoid having to train the full softmax at each prediction. “A good model should be able to differentiate data from noise by means of logistic regression”.

    - Train two binary logistic regressions
        1. One for the true pair (center word, output)
        2. One for random pairs (center word, random word)

    Exaxmple:

    - *"I want a glass of orange juice to go along with my cereal."* 
    For every positive example, we have k-1 negative examples. Here k = 4.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20100.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20100.png)

- How does CBOW work?
    - Target is a single word → C C C + C C C
    - Context is now all words within the context window of size m.
    - Average embeddings of all context words
- Differentiate CBOW from Skip-gram

    According to the author, CBOW is faster but skip-gram but does a better job for infrequent words.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20101.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20101.png)

    Source: [https://www.knime.com/blog/word-embedding-word2vec-explained](https://www.knime.com/blog/word-embedding-word2vec-explained)

- What does GloVe?
    - Calculate co-occurrence directly for the whole corpus
    - We go through the entire corpus and update the co-occurrence counts.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20102.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20102.png)

    - Go through all pairs of words in the co-occurrence matrix.
    - Minimize distance between dot product of 2 word embeddings
    - Function *f*(Pij) - allows one to weight the co-occurrences. E.g. give a lower weight to frequent co-occurrences.
    - Trains much more quickly and with small corpora
- How can word embeddings be evaluated?
    1. Intrinsic Evaluation
        1. Evaluate on a specifically created subtask

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20103.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20103.png)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20104.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20104.png)

        Source Illustration: Pennington, Socher, Manning: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

        2. Another intrinsic evaluation approach is comparing with human judgments.

    2. Extrinsic Evaluation
        - How good are our embeddings for solving actual tasks?
        - Example of such task is named entity recognition.
- Information on dependence on Hyperparameters

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20105.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20105.png)

    Source Slides: Socher, Manning

- What are advantages of Character Level Models?
    - Vocabulary is just of the size of the alphabet + punctuation
    - No need for tokenization
    - No unknown word (out of vocabulary)
    - Can learn structures data encoded by characters and only plain text (For example gen informations)
- What are disadvantages of Character Level Models?
    - Require deeper RNN's because there are more characters in a sentence than words
    - Generally worse capturing long-distance dependencies
    - In order to model long term dependencies, hidden layers need to be larger
- What are different types of NNs?

    One to One: Image Classification

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20106.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20106.png)

    Illustration: Andrey Karpathy. The Unreasonable Effectiveness of Recurrent Neural Networks

    One to Many: Generating caption for an image

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20107.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20107.png)

    Source: [https://medium.com/swlh/automatic-image-captioning-using-deep-learning-5e899c127387](https://medium.com/swlh/automatic-image-captioning-using-deep-learning-5e899c127387)

    Many to One: Sentiment Classification

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20108.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20108.png)

    Source:

    Many to Many (Sequence2Sequence): Machine Translation where length of input and output can be different or Entity Recognition

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20109.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20109.png)

    Source: [https://confusedcoders.com/data-science/deep-learning/how-to-build-deep-neural-network-for-custom-ner-with-keras](https://confusedcoders.com/data-science/deep-learning/how-to-build-deep-neural-network-for-custom-ner-with-keras)

- Of what does the Sequence2Sequence architecture exist?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20110.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20110.png)

    Source: [https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20111.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20111.png)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20112.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20112.png)

- How can the process of predicting in the decoder part be improved?
    - **Further Improvements**
    Remember, we are creating our word vector based on a probabilistic model.
    Predicting the output vector using a greedy approach may not be the best solution.

- Describe Beam Search

    Beam search is an approximate search algorithm, so no guarantee to find a global maximum. Beam Search is a heuristic algorithm - it does not guarantee the optimal solution.

    Instead of selecting the first best word, we take the k best matching words and calculate the probabilities for all the next words. Out of these, choose again the top b matching samples and repeat. If we choose b=1, than beam seach is a greedy search.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20113.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20113.png)

    Source: [https://d2l.ai/chapter_recurrent-modern/beam-search.html](https://d2l.ai/chapter_recurrent-modern/beam-search.html)

- What is a painpoint of beam search?

    The probability of a sentence is defined as product of probabilities of words conditioned on previous words. Due to compounding probabilities, longer sentences
    have a lower probability by definition. Shorter sentences can have higher probability even if they make less sense.

- How can the painpoints of beam search be solved?

    We need to normalize sentences by their length multiply the probability of a sequence by its inverse length.

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20114.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20114.png)

- Describe the light normalization implementation
    1. run, select and save top 3 most likely sequences for every sequence length (e.g. Defined max sequence length = 5 → 1,2,3,4,5)
    - This results in a set of size 3 containing
        - 1 sequences (single words, e.g.: ("Oh","donald","trump"), ("Go","Barack","Obama"), (...) )
        - 2 sequences (two word sequences)
        - Up to N (here 5) sequences

    2. For every sequence length run a normalized probability score calculation and assign it to every sequence.

    3. Select the sequence with highest probability of all

- Which two errors can arise with Beam Search?
    - Model errors from our RNN
    - Search errors from our beam search hyperparameter

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20115.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20115.png)

## Reinforcement Learning

- Describe the idea of RL
    - The idea of RL is that an algorithm learns a task by itself without precise instructions from a human. A developer defines the goal and provides it with a reward. In addition, it must be defined which actions the model is allowed to perform (actions). The model explores several strategies over a period of time and acquires the best one. This is also known as trial-and-error search.
- Describe the terminology of RL
    - **Environment**: The abstract world that everything happens in.
    - **Agent**: The Algorithm which interacts with the environment
    - **State**: Part of the environment. Every action of the agent with the Env changes their state
    - **Reward**: Every interaction of an agent with the environment produces a reward. Reward is a signal of how good or bad the interaction was in terms of some goal. At some point, the interaction is terminate. The final goal of the agent is to maximize the cumultative reward
    - **Policy**: The rules by which the agent acts (π)
- When to use Reinforcement Learning
    - If the problem is an inherently sequential decision-making problem i.e. a board game like chess or games in general.
    - If the optimal behaviour is not known in advance (unsolved games without "perfect strategy")
    - When we can simulate the system behaviour as good as possible to imitate the real world
    - Solving problems with a non-differentiable loss function
- RL is separated into Model-Free RL and Model-Based RL. Name subcategories and associated algorithms

    Model-Free RL:

    - Policy Optimization (DDPG, Policy Gradient)
    - Q-Learning (DQN, DDPG)

    Model Based RL:

    - Learn the Model (World Models)
    - Given the Model (AlphaZero)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20116.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20116.png)

- Tasks of the Environment
    - Interface: An Env need to provide an interface with which the agent could interact
    - Calculating Effects: For an action performed by the agent, the Env needs to calculate the:
        - next state
        - reward for this action
- Describe Model Free approaches
    - The focus is on finding out the value functions directly by interactions with the environment. All Model Free algorithms learn the value function directly from the environment. That's not entirely true. Looking at Policy Gradient approaches we do not learn the value function.
- Define the Markov Decision Problem (MDP)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20117.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20117.png)

    - The MDP is inspired by the theorem "The future is independent of the past given the present". (Markov Property) That means, to calculate values for the future state $S_{t+1}$, only the state $S_t$ counts. It should keep all the information from already passed states.

- Describe the Markov Property
    - A state contains the Markov property. This gives the probabilities of all possible next states under the condition of the current state in a matrix P. (Markov Property: Capture all relevant information from the past)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20118.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20118.png)

- What is the discounted reward?
    - It is the discounted sum over all rewards that happened in the future. Intuitively speaking it is something like a measurement how efficient the agent solved an episode. E.g.: An agent decided to walk loops in a  GridWorld scenario and after a few loops he changed his route towards the goal and reached it. The discounted is still good but it would have been better if he had taken the direct path, because reward further in the future are discounted the most. Hence, the final is always discounted most.
    - Recall that return is the total discounted reward in an episode:
        - G_t = R_t+1 + γR_t+2 + ... + γT-1RT
- What is a policy?

    A policy is the agent’s behaviour. It is a map from state to action.

    - Deterministic policy: a=π(s).
    - Stochastic policy: π(a|s) =P[A=a|S=s].

- What is the State Value Function?
    - Value function for a state is the reward under the assumption of taking the best action/policy from there. → Kind of the best Q Value
    - The Value function takes the reward of all possible next states into account (and the next rewards of those states, which is G) and calculates the value by multiplying the action probability (policy) by the reward for the particular state.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20119.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20119.png)

    However the description above can be written differently (much smarter so that we can apply dynamic programming on it)!

    Explanation of the belman equation:

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20120.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20120.png)

    Discounted Reward: That's just the sum of all rewards (discounted with gamma, but this isn't important to understand the concept here). 

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20121.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20121.png)

    Defining G_t brings us to the Expectation_policy 

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20122.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20122.png)

    of 

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20123.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20123.png)

    The expectation is mathematically defined by the probability (here Pi/policy) times over all elements (here G_t). The policy/probability is the probability of a specific action under state s.

    DP, MC and TD Learning methods are value-based methods (Learnt Value Function, Implicit policy).

- What is the Action Value Function?
    - Equation almost the same as State Value Function

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20124.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20124.png)

- Why do we need the bellman equation and how to solve it?
    - Reason: Computing entire value function for some RL use case is too complex/time consuming, therefore, breaking it down to sub equations.
    - Usually solved by using backup methods, we transfer then the information back to a state from its successor state. Therefore we can calculate paths separately?

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20125.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20125.png)

        Source: [https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)

    Can also be applied to q-values!

- How is optimal policy defined?
    - Optimal policy is defined as that policy which has the greatest expected return for *all states*
    - We can explicitly write out this requirement for state-value functions and state-action value functions.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20126.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20126.png)

        Source: [https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)

- The optimal policy and the bellman equation leads to the Bellman optimality equation
    - By that we formulate the equation without referring to a specific policy (that's why a star is beneath v)
    - "Intuitively, this says that the value of a state under an optimal policy must equal the expected return for the best action from that state."
    - The next image shows this for q values. The order would be that we select the best action at the current state s, following the optimal policy which results in the best value function value.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20127.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20127.png)

        Source: [https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)

- What is the Markov Reward Process (MRP)?
    - MRP is a tuple of (S,P,R, gamma)
        - S is a state space
        - P is the state transaction function (probability function)
        - R is the reward function with E(R_t+1|S), i.e. how much immediate reward we would get for a state S.
        - Everything results in G_t, which is the total discounted reward for a time step t. The goal is to maximize this function:

            ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20128.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20128.png)

        - Gamma is the discount factor which is applied to future rewards. This should play in less than the "closer". This also leads us to avoid infinite returns in Markov processes.
    - Another important point is the value function V(S) which can be adapted to different use cases. The original MDP uses the Bellman Equation:

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20129.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20129.png)

        For gamma = 1 we aim at long term rewards and for gamma = 0 we tend to short term profit.

        In the following example, a discount of 0.9 is used from the final state reward(Peach) to the start state of the agent (Mario).

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20130.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20130.png)

        However, to solve the Bellman equation we need almost all the information of our environment: Markov property and calculate v* for all other states. Since this rarely works, many RL methods try to approximate the Bellman function with the last done state transition instead of having a complete knowledge of all transitions.

- What is the difference between State Value V(s) and Action Values Q(s,a)?
    - Basically two different ways to solve reinforcement learning problems.
    - V_Pi(s) describes the expected value of following the policy Pi when the agent starts at the state S. Basically, the action is greedy.
    - Q_Pi (s,a) describes the expected value when an action a is taken from state S under policy Pi.
- What is the relationship between Q-Values (Q_Pi) and the Value function (V_Pi)?

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20131.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20131.png)

    - You sum every action-value multiplied by the probability to take that action (the policy 𝜋(𝑎|𝑠)*π(a|s)).
    - If you think of the grid world example, you multiply the probability of (up/down/right/left) with the one step ahead state value of (up/down/right/left).

        Source: [https://datascience.stackexchange.com/questions/9832/what-is-the-q-function-and-what-is-the-v-function-in-reinforcement-learning](https://datascience.stackexchange.com/questions/9832/what-is-the-q-function-and-what-is-the-v-function-in-reinforcement-learning)

- Describe the idea of Q-Learning
    - The goal of Q-learning is to learn a policy, which tells an agent what action to take under what circumstances.
    - Approximation of the MDP with the approach of action states without solving the Bellman equation. It is a MDP with finite and small enough state space and actions
    - How can a grid world problem be solved with Q-learning?
        1. Initialize a table for each state-action pair
        2. Fill in the table step by step while interacting with the environment
        3. Calculate the Q-Values

            ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20132.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20132.png)

            Source: [http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf](http://users.isr.ist.utl.pt/~mtjspaan/readingGroup/ProofQlearning.pdf)

        4. Repeat step 2 + 3 for N epochs
        5. Check how well the learned table approximated the State Values (V(s)) for the Bellmann equation. This could look like the following:

            ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20133.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20133.png)

            Source: [https://towardsdatascience.com/qrash-course-deep-q-networks-from-the-ground-up-1bbda41d3677](https://towardsdatascience.com/qrash-course-deep-q-networks-from-the-ground-up-1bbda41d3677)

- What is Policy Iteration?

    **Policy iteration** includes: **policy evaluation** + **policy improvement**, (both are iteratively)

    - Uses also bellman equation but without maximum.
    - For each state S we check if the Q-value is larger than the value value: q(s,a) > v(s) under the same policy (test different policies). If the Q-Value returns a larger value, the strategy performs better and we switch to it.

    Sometimes it only finds local minima

    1. Choose a policy randomly (e.g. go in each direction with prob = 0.25)
    2. Do policy evaluation: Calculate the Value for each state
    3. Do policy improvement: Change an action a (that results in a new policy pi') and see if the values are better. If so, switch to the new policy,. (We could use the Value function to do this successively for each state
    - [https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/](https://www.analyticsvidhya.com/blog/2018/09/reinforcement-learning-model-based-planning-dynamic-programming/)

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20134.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20134.png)

- Example of Policy Iteration for Monte Carlo

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20135.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20135.png)

    Source: [https://www.programmersought.com/article/39135619445/](https://www.programmersought.com/article/39135619445/)

- What is Value iteration?

    **Value iteration** includes: **finding optimal value function** + one **policy extraction**

    **Usually finds global minima**

    It is similar to policy iteration, although it differs in the following:

    1. Value Iteration tries to find the optimal value function rather than first the optimal policy. Once the best value function was found, the policy can be derived.
    2. It shares the policy improvement function/approach and a truncated policy evaluation

    For value iteration it can be less iterations but for one iteration there can be so much of work, because it uses bellman maximum equation. For the policy iteration more iterations.

- Compare Policy Iteration with Value Iteration
    - *"In my experience, policy iteration is faster than value iteration, as a policy converges more quickly than a value function. I remember this is also described in the book."*

        Source: [https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration](https://stackoverflow.com/questions/37370015/what-is-the-difference-between-value-iteration-and-policy-iteration)

    - Policy Iteration: Sometimes it only finds local minima
    - Value Iteration: Finds global minima

- What is a disadvantage of policy iteration?
    - Can get stuck in a local minimum
- How can the disadvantages of policy iteration be compensated?
    - Early stopping

- What is the idea of the Monte Carlo method?

    In General: A statistical approach that averages the mean for a state over an episode.

    1. Initialize rewards randomly
    2. Run through a specific number of episodes (e.g. 1000)
    3. ~~Calculate the discounted reward for each state~~
    4. Apply either:
        1. First visit method: Average returns only for first time s is visited in an episode.
        2. Every visit method: Average returns for every time s is visited in an episode.

    (Check again if example is correct (might be wrong))

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20136.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20136.png)

    Source: [https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/](https://www.analyticsvidhya.com/blog/2018/11/reinforcement-learning-introduction-monte-carlo-learning-openai-gym/)

- Give a summary of Monte Carlo
    - MC methods learn directly from episodes of experience.
    - MC is model-free: no knowledge of MDP transitions / rewards.
    - MC uses the simplest possible idea: value = mean return.
    - Episode must terminate before calculating return.
    - Average return is calculated instead of using true return G.
    - First Visit MC: The first time-step t that state s is visited in an episode.
    - Every Visit MC: Every time-step t that state s is visited in an episode.
    - [https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/README.md#MonteCarlo](https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo/blob/master/README.md#MonteCarlo)
- What is the idea of Temporal Difference Learning?
    - Temporal Difference Learning (also TD learning) is a method of reinforcement learning. In reinforcement learning, an agent receives a reward after a series of actions and adjusts its strategy to maximize the reward. An agent with a TD-learning algorithm makes the adjustment not when it receives the reward, but after each action based on an estimated expected reward.

        Source: [https://de.wikipedia.org/wiki/Temporal_Difference_Learning](https://de.wikipedia.org/wiki/Temporal_Difference_Learning)

    - Is an on-policy learning approach

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20137.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20137.png)

        Source: [https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo](https://github.com/omerbsezer/Reinforcement_learning_tutorial_with_demo)

    - Compared to DP and MC, which are both offline because they have to finish the epoch first, TD is a mix of the two. Monte Carlo has to run a branch of States, DP even has to run all of them.
- Summary Temporal Difference Learning
    - TD methods learn directly from episodes of experience.
    - TD updates a guess towards a guess
    - TD learns from incomplete episodes, by bootstrapping.
    - TD uses bootstrapping like DP, TD learns experience like MC (combines MC and DP).
- MC-TD Difference
    - MC and TD learn from experience.
    - TD can learn before knowing the final outcome.
    - TD can learn online after every step. MC must wait until end of episode before return is known.
    - TD can learn without the final outcome.
    - TD can learn from incomplete sequences. MC can only learn from complete sequences.
    - TD works in continuing environments. MC only works for episodic environments.
    - MC has high variance, zero bias. TD has low variance, some bias.
- What is the idea of Dynamic Programming?
    - DP is a general framework concept. It is derived from the divide and conquer principle, where DP tries to break down a complex problem into sub-problems that are easier to solve. In the end, the solutions of the subproblems are merged to solve the main problem.

        In order to apply it, the following properties must be given:

        - The optimal solution can be divided into subproblems
        - Subproblems occur more often
        - Subproblems solutions can be cached
- Are the dynamic programming requirements met for a MDP?
    - Bellman equation (is a DP approach) gives us a recursive decomposition
    - The value function stores the action (serves as a cache) and can thus be used for DP
- What is meant by full backups vs sample backups and shallow vs deep?
    - The history of the learned. Full backup utilizes a lot of paths and states, whereas sample backup is a branch in the tree diagram.
    - Shallow means that it does not go very deep into the tree before making an update (fast), whereas deep waits for a complete episode.

    [](http://gki.informatik.uni-freiburg.de/teaching/ws0607/advanced/recordings/reinforcement.pdf)

- What is the difference between on-policy and off-policy?
    - Online: during the job: E.g TD Algorithm
    - On policy learning (Monte Carlo): Improve the same policy that is used to make decisions (generate experience).

    - Off policy learning (Q-Learning (TD-Algorithm)):  Improve policy different than the one used to generate experience. Here, improve target policy while following behaviour.
        - Target policy - what we want to learn. Therefore, we evaluate and improve upon.
        - Behaviour policy - what we use to generate data.

        Motivation for having 2 policies is the trade-off between exploration and exploitation. In order to know what actions are best, we need to explore as much as possible - i.e. take suboptimal actions as well.

        Target policy π(St) is greedy w.r.t. Q(s,a),  because we want our target policy to provide the best action available.

        Behavior policy μ(St) is often 𝜀-greedy w.r.t. Q(s,a).

        - 𝜀-greedy policy is an extension of greedy policy
            - With probability 𝜀 it will output a random action.
            - Otherwise, it output is the best action.
            - 𝜀 controls how much exploration we want to allow.

        This combination of two policies let’s us combine “the best of both worlds”. Our behavior policy gives enough randomness to explore all possible state-action pairs. Our target policy still learns to output the best action from each state.

- What is the difference between Episodic and Continuous Tasks?
    - Episodic task: A task which can last a finite amount of time is called Episodic task ( an episode )
    - Continuous task: A task which never ends is called Continuous task. For example trading in the cryptocurrency markets or learning Machine learning on internet.
- Differentiate Exploitation from Exploration
    - **Exploitation**: Following a strict policy by exploiting the known information from already known states to maximize the reward.
    - **Exploration**: Finding more information about the environment. Exploring a lot of states and actions in the environment by avoid a greedy policy every action.

- What is the sample efficiency problem?
    - After improving our policy by an experience we do not use this information anymore, but with future experiences, which is not sample efficient.
- Define Experience Replay
    - Another human behaviour imitation technique is called experience replay. The idea behind experience replay is, that the agent builds a huge memory with already experienced events.
    - For this, a memory M is needed which keeps experiences e. We store the agents experience et=(st,at,rt,st+1)e_{t}=\left(s_{t}, a_{t}, r_{t}, s_{t+1}\right)et=(st,at,rt,st+1) there.
    This means instead of running Q-learning on state/action pairs as they occur during simulation or actual experience, the system stores the data discovered for [state, action, reward, next_state] - typically in a large table. Note this does not store associated values - this is the raw data to feed into action-value calculations later.
    - The learning phase is then logically separated from gaining experience, and based on taking random samples from this table which does not stand in a directional relationship. You still want to interleave the two processes - acting and learning - because improving the policy will lead to different behaviour that should explore actions closer to optimal ones, and you want to learn from those.
- What are the advantages of experience replay?
    - More efficient use of previous experience, by learning with it multiple times. This is key when gaining real-world experience is costly, you can get full use of it. The Q-learning updates are incremental and do not converge quickly, so multiple passes with the same data is beneficial, especially when there is low variance in immediate outcomes (reward, next state) given the same state, action pair.
    - Better convergence behaviour when training a function approximator. Partly this is because the data is more like i.i.d. data assumed in most supervised learning convergence proofs.
- What are advantages of Experience Replay?
    - This turns our RL problem into a supervised learning problem!
    - Now we reuse our data, which is far more sample efficient.
    - Before, we worked with non i.i.d., strongly correlated data - we always saw states

        one after the other in the trajectory (now we randomly sample) 

        - We break the correlation by presenting (state,value) pairs in a random order.
        - This greatly increases the convergence rate (or even makes possible).

- What are the disadvantages of experience replay?
    - It is harder to use multi-step learning algorithms, such as $Q(\lambda)$ , which can be tuned to give better learning curves by balancing between bias (due to bootstrapping) and variance (due to delays and randomness in long-term outcomes). Multi-step DQN with experience-replay DQN is one of the extensions explored in the paper Rainbow: Combining Improvements in Deep Reinforcement Learning.
- How can we re-use the data?
    1. Saving experiences (Gather data)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20138.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20138.png)

    1. Apply Experience Replay
        1. Define cost function, such as Least Square
        2. Random sample (state, value) pair from experience
        3. Apply SGD Update
- What is the idea of Deep Q-Learning (DQN)?

    DQN is quite young and was introduced only in 2013. It is an end-to-end learning for the Q(s,a) values for pixels with which well-known Atari games can be solved. Two main ideas are followed:

    - Experience Replay (kind of generator for the data generation of the training process).
    - Fixed Q-Targets:
        - Form of off-policy method
        - Direct manipulation of a used Q Value leads to an unstable NN.
        - To solve this two parameters are needed:
            1. current Q values
            2. target Q values
            - During training, we update the target Q values with Stochastic Gradient Descent. After the epoch, we replace our current weights with the target weights and start a new epoch.

- So far we have seen Value Based RL which relies on improving the model by the state value function, but there is also the possibility to learn a policy directly. What are the advantages?
    - Better convergence properties
    - When the space is large, the usage of memory and computation consumption grows rapidly. The policy based RL avoids this because the objective is to learn a set of parameters that is far less than the space count.
    - Can learn stochastic policies. Stochastic policies are better than deterministic policies, especially in 2 players game where if one player acts deterministically the other player will develop counter measures in order to win.
- What are disadvantages of Policy Based RL?
    - Typically converge to a local rather than global optimum
    - Evaluating a policy is typically inefficient and high variance policy based RL has high variance, but there are techniques to reduce this variance.
- How can we fix a RL case where the problem is too large (keeping track of every single state or state-action pair becomes infeasible)
    - Value Function Approximation. Therefore we extend the original Value function of a state and the q- values with a weight parameter.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20139.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20139.png)

- What kind of function can be used for Value Function approximation? Elaborate.
    - **Linear combinations of features:**
        - In this case we represent the value function by a linear combination of the features and weights.

            ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20140.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20140.png)

            - This lets us generalize from seen states to unseen states, by optimizing the function to already seen value function for a state and apply the parameters for unseen states.

        Big advantage of this approach - it converges on a global optimum.

    - **Neural networks:**
        - Most general goal: find optimal parameter w by finding its (local) minimum. Cost function: mean squared error.
    - ~~Monte Carlo Method:~~
- In which step during the value function approximation is the approximation checked?
    - Policy evaluation

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20141.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20141.png)

    Source: [https://dnddnjs.gitbook.io/rl/chaper-8-value-function-approximatiobn/learning-with-function-approximator](https://dnddnjs.gitbook.io/rl/chaper-8-value-function-approximatiobn/learning-with-function-approximator)

    Slide: D. Silver 2015.

- What are Policy Gradient Methods?
    - Return a matrix of probabilities for each action via softmax
    - Exploration is handled automatically
    - We don't care about calculating accurate state values
    - Adjust the action probabilities depending on rewards
- What is a parametrized policy (belongs to Policy Gradient Methods)?
    - Assign a probability density to each action
    - Such policies can select actions without consulting a value function
    - As we will see, the most sophisticated methods still utilize value functions to speed up learning, *but not for action selection (but this is considered as Actor-Critic)*
- A stochastic Policy is a Policy Based RL. Why would we choose that?

    At first it is important to note that stochastic does not mean randomness in all states, but it can be stochastic in some states where it makes sense. Usually maximizing reward leads to deterministic policy. But in some cases deterministic policies are not a good fit for the problem, for example in any two player game, playing deterministically means the other player will be able to come with counter measures in order to win all the time. For example in Rock-Cissors-Paper game, if we play deterministically meaning the same shape every time, then the other player can easily counter our policy and wins every game.

    - I.e. at every point select one of the three actions with equal probability (0.33).
    - Therefore, a stochastic policy is desirable!

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20142.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20142.png)

    Source: [https://en.wikipedia.org/wiki/Rock_paper_scissors](https://en.wikipedia.org/wiki/Rock_paper_scissors)

- Another Policy Based Algorithm is REINFORCE with Monte Carlo Policy Gradient, explain it. (Also called Monte-Carlo Policy Gradient)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20143.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20143.png)

- REINFORCE, also called Vanilla Policy Gradient (VPG) can also work with Baseline, explain.
    - With the baseline, the variance is mitigated

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20144.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20144.png)

    Source Illustration: [https://papoudakis.github.io/announcements/policy_gradient_a3c/](https://papoudakis.github.io/announcements/policy_gradient_a3c/)

- What are disadvantages of REINFORCE?
    - REINFORCE is unbiased and will converge to a local minimum.
    - However, it is a Monte Carlo method which has its disadvantages.
    - Slow learning because of high variance (partially solved by baseline).
    - Impossible to implement online because of return.
    - we need to generate a full episode to perform any learning.
    - Therefore, can only apply to episodic tasks (needs to terminate).
- Provide a intersection diagram to show what kind of learning a RL Algo can have

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20145.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20145.png)

    Source: Karlsruhe University of Applied Science - Yurii Tolochko, 2019 

- Explain the Actor-Critic Method

    Actor-Critic methods are a buildup on REINFORCE-with-baseline.

    Two parts of the algorithm:

    - Actor - updates the action selection probabilities
    - Critic - provides a baseline for actions’ value

    Actor Critic Method has the same goal as the Baseline for REINFORCE: It should remove the high variance in predicting an action. Instead of applying a Monte Carlo method for value function approximation, we perform an one-step Temporal Difference algorithm. 

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20146.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20146.png)

    Source Illustration: Reinforcement Learning: An Introduction. A. Barto, R. Sutton. Page 325

- What is the motivation for Monte Carlo Tree search rather than use Depth-First search?
    - DFS fails with more complicated games (too many states and branches to go through)
- Explain Monte Carlo Tree search
    - TLDR: Random moves is the main idea behind MCTS.
    - Monte Carlo Tree Search is an efficient reinforcement learning approach to lookup for best path in a tree without performing a depth search over each path. Imagine an easy game like tik tak toe, where we have only 9 field so a game definitively ends after 9 actions. A depth-first search can easy find the best path through all actions. But for games like go or chess, where wie have a huge amount of states (1012010^12010120), it is impossible to perform a normal depth-first search. This is where the MCTS comes in.

    ## Minimax and Game Trees

    - A game tree is a directed graph whose nodes are positioned in a game. The following picture is a game tree for the game tic tac toe.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20147.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20147.png)

        Source: [https://upload.wikimedia.org/wikipedia/commons/1/1f/Tic-tac-toe-full-game-tree-x-rational.jpg](https://upload.wikimedia.org/wikipedia/commons/1/1f/Tic-tac-toe-full-game-tree-x-rational.jpg)

    - The main idea behind MCTS is to make random moves in the graph. For any given state we are no longer looking through every future combinations of states and actions. Instead, we use random exploration to estimate a state’s value. If we simulate 1000 games from a state s with random move selection, and see that on average P1 wins 75% of the time, we can be fairly certain that this state is better for P1 than P2.

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20148.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20148.png)

- Vanilla MCTS has a problem with the mere number of possible next states. How could it be improved?
    - The Vanilla MCTS suffers in the point, that if there are a lot of possible actions in a state, only a few would lead to a good result. The chance to hit one of these good states is really low. An improvement for this could be to change these with something more intelligent. This is where **MCTS Upper-Confidence Bound** comes in. We can use the information after our backpropagation. The states currently keep information about their winning chance. We do actual move selection by visit count from root (max or as proportional probability). Idea is that by UCT "good" nodes were visited more often.

    - But of course, a balance between exploration and exploitation is quite important. We need to make sure that we explore as many states as possible, especially in the first few rounds. The following formula can be used to calculate the selection of the next node:

        ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20149.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20149.png)

    - This approach can be leveraged by using a neural net which does value function approximation. So we do not need to explore a lot of states because the NN tells us, how a good a state is.

- How does AlphaGo Zero work?

    [How AlphaGo Zero works - Google DeepMind](https://www.youtube.com/watch?v=MgowR4pq3e8)

    ![5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20150.png](5%20EN%20AI%20-%20Computer%20Vision,%20NLP,%20Reinforcement%20Lear%2079441a9ad0954925a2b006d0ae801fd2/Untitled%20150.png)

    Source Illustrations: David Silver

- Shorten the algorithm of AlphaGo Zero
    1. Use current network and play some games to create training data. use MCTS (Monte Carlo Tree Search) for that. Completly focused on self-training.
    2. Use this data to retrain the network (Fix and target values). We train a policy and a value network.
    3. Now we have a slightly better network, repeat 1 + 2.

    Notes: 

    - AlphaGo Zero uses ResNet
    - Takes the last 8 images of the board into account

    Output of the feature vector:

    - Value vector:
    - Policy vector: Probability distribution for all actions
- Define Actor-Critic
    - Learnt value function
    - Learnt policy
    - Value function “helps” learn the policy