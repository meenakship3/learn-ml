# Study Guide: Unsupervised Learning

## Overview
This video series provides a comprehensive introduction to unsupervised learning, a crucial area in machine learning. It covers the core concepts, common techniques, and practical applications across different data types like images, text, and graphs. The first part introduces the fundamentals of unsupervised learning, contrasting it with supervised learning, and delves into dimensionality reduction, clustering, and generative modeling. The second part focuses on representation learning, explaining how machines can learn meaningful numerical representations of complex data such as words, entire sentences, graphs, and images, even without explicit labels.

## Key Concepts

*   **Unsupervised Learning:** This is a type of machine learning where the algorithm tries to find patterns and structures in data without being given any pre-existing labels or categories. Imagine having a big box of unsorted toys and trying to group them into different categories (cars, dolls, blocks) all by yourself, without anyone telling you what each toy is. The goal is to discover hidden insights or organization within the data.

*   **Supervised Learning:** In contrast to unsupervised learning, supervised learning involves training a model using data that already has "labels." For example, if you want to teach a computer to identify cats and dogs, you'd feed it many pictures, each clearly marked "cat" or "dog."

*   **Reinforcement Learning:** A type of machine learning where an "agent" learns to make decisions by performing actions in an environment to maximize a reward. It's like training a pet by giving it treats for good behavior.

*   **Dimensionality Reduction:** This technique aims to simplify complex data by reducing the number of "dimensions" or features while keeping the most important information. Imagine squishing a super-detailed 3D drawing onto a 2D piece of paper without losing the main features. This helps visualize data, build smaller models, and compress information.

*   **Clustering:** The process of grouping similar data points together into "clusters," while ensuring that data points in different clusters are as dissimilar as possible. Think of sorting your LEGO bricks by color or shape without being told which colors or shapes exist – you just find the natural groupings.

*   **Generative Modeling:** This involves creating models that learn the underlying "distribution" of the input data. In simpler terms, it learns how the data was generated. Once a generative model understands this, it can then generate new, realistic samples that resemble the original data. For instance, a model could learn from many cat pictures and then create entirely new, believable cat images.

*   **Representation Learning:** The process of automatically discovering good "representations" or numerical codes for raw data. This is especially important for non-numeric data like text, images, or graphs. The goal is to turn complex information into a simpler, more compact form (often a vector of numbers) that computers can easily process and use for various tasks.

*   **Principal Component Analysis (PCA):** A popular technique for dimensionality reduction. It works by finding the directions (called "principal components") in your data where it spreads out the most (has the most "variance"). By projecting the data onto these main directions, you can reduce its dimensions while retaining as much important information as possible.

*   **K-Means Clustering:** A widely used centroid-based clustering algorithm. It groups data points into a pre-defined number of clusters (K). Each cluster is represented by its "centroid" (the average position of all data points in that cluster). The algorithm iteratively assigns points to the nearest centroid and then updates the centroids until the clusters are stable.

*   **Generative Adversarial Networks (GANs):** A powerful type of generative model consisting of two neural networks: a "Generator" and a "Discriminator," which compete against each other in an "adversarial game." The Generator tries to create realistic fake data (e.g., images), while the Discriminator tries to distinguish between real and fake data. Both networks improve over time, with the Generator becoming better at producing fakes that fool the Discriminator, and the Discriminator becoming better at spotting fakes.

*   **Word2Vec:** A technique in natural language processing (NLP) that learns to represent words as dense numerical vectors (embeddings). It's based on the "distributional hypothesis," which suggests that words appearing in similar contexts tend to have similar meanings. It has two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.

*   **BERT (Bidirectional Encoder Representations from Transformers):** A highly influential language model that generates word representations by considering the full context of a word within a sentence, both from the left and the right. Unlike older models, BERT can understand the deep meaning of words based on how they relate to all other words around them. It's pre-trained on large amounts of text and can then be fine-tuned for specific tasks.

*   **DeepWalk:** An early neural network approach for learning "node embeddings" (numerical representations for nodes) in graphs. It uses "random walks" (random paths through the graph) to generate sequences of nodes, which are then treated like sentences. These sequences are fed into a Skip-gram model (similar to Word2Vec) to learn embeddings that capture the graph's structure.

*   **Node2Vec:** An improvement over DeepWalk, Node2Vec uses "biased random walks" that can be controlled to explore either the immediate neighborhood of a node (like a Breadth-First Search, BFS) or venture further away (like a Depth-First Search, DFS). This allows the learned node embeddings to capture different aspects of the graph's structure, either local or global.

*   **Self-supervised Learning:** A paradigm within unsupervised learning where the model creates its own "supervision" or "labels" from the unlabeled data itself. It designs a "pretext task" that allows the model to learn useful features without human annotations.

*   **Contrastive Learning:** A self-supervised learning approach that learns meaningful representations by contrasting similar and dissimilar data points. It tries to pull representations of "positive pairs" (similar items) closer together in an embedding space while pushing "negative pairs" (dissimilar items) farther apart.

*   **SimCLR (A Simple Framework for Contrastive Learning of Visual Representations):** A powerful framework for self-supervised contrastive learning that learns high-quality visual features from unlabeled images. It works by creating different augmented versions of the same image (positive pairs) and then maximizes the similarity between their learned representations, while minimizing similarity with other images in the batch (negative pairs).

## Technical Terms

*   **Input Data (X):** The raw information given to a machine learning model, like images, text, or numbers.
*   **Label (Y):** The correct answer or category associated with a piece of input data in supervised learning.
*   **Interesting Pattern:** A subjective term in unsupervised learning, referring to the hidden structures, groups, or relationships that the algorithm aims to discover in data.
*   **Unlabeled Data:** Data that does not have pre-assigned categories or classifications. This is very common and a key target for unsupervised learning.
*   **Image Super-Resolution:** The task of taking a low-resolution image and creating a higher-resolution version of it.
*   **Anomaly Detection:** Identifying unusual patterns or outliers in data that do not conform to expected behavior. Used for fraud detection or spotting unusual system activity.
*   **Density (of Normal Data Points):** In anomaly detection, this refers to how concentrated or spread out normal (non-anomalous) data points are in a given space. Anomalies often fall into low-density regions.
*   **High-Dimensional Vector Space:** A mathematical space with many "dimensions" or features. For example, an image with thousands of pixels can be considered high-dimensional.
*   **Low-Dimensional Vector Space:** A mathematical space with fewer "dimensions" or features, simplifying data for easier analysis or processing.
*   **Model Parameters:** The internal settings or values that a machine learning model learns during training to make predictions or find patterns.
*   **Compression:** Reducing the size of data while trying to maintain its quality and important information.
*   **Centroid-Based Clustering:** A type of clustering where each group is represented by a central point (the centroid) that is the mean of all data points in that group.
*   **Connectivity-Based Clustering:** A type of clustering that groups objects into a single cluster if they are connected to each other, forming a chain or network.
*   **Probabilistic-Based Clustering:** Clustering methods where each data point can belong to multiple clusters with a certain probability, rather than being assigned exclusively to one.
*   **Hard Assignment:** When a data point is assigned strictly to only one cluster.
*   **Distance Metric:** A rule or formula used to calculate how "far apart" or "similar" two data points are in a given space (e.g., Euclidean distance).
*   **Iterative Process:** A process that involves repeating a set of steps multiple times, usually to refine a solution or reach a desired outcome.
*   **Convergence:** In iterative algorithms, convergence occurs when the changes between iterations become very small, indicating that the algorithm has found a stable solution and is no longer significantly improving.
*   **Pixel:** The smallest unit of an image, typically represented by its color values (e.g., RGB).
*   **RGB Value:** A way to represent colors using a combination of red, green, and blue light intensities. Each pixel in a color image has an RGB value.
*   **Distribution (of Underlying Data):** The way data points are spread out or arranged in a dataset. Generative models aim to learn this pattern.
*   **New Samples:** Data points that are created by a generative model, resembling the original data it was trained on.
*   **Data Augmentation:** Creating new, modified versions of existing data (e.g., rotating images, adding noise to text) to increase the size and diversity of a training dataset.
*   **Classification:** The task of assigning an input data point to one of several predefined categories or classes.
*   **Regression:** The task of predicting a continuous numerical value (e.g., predicting house prices).
*   **Missing Data:** Gaps or absent values in a dataset that generative models can sometimes fill in.
*   **Density Estimation Models:** Models that explicitly try to estimate the probability distribution from which data points were generated.
*   **Multi-Gaussian Model:** A statistical model that assumes data is distributed as a combination of multiple Gaussian (bell-curve) distributions.
*   **Maximum Likelihood Estimation (MLE):** A method for estimating the parameters of a statistical model by finding the parameter values that make the observed data most probable.
*   **Generator Network:** In a GAN, the neural network responsible for creating synthetic data (e.g., fake images).
*   **Random Noise:** Random, unstructured data (like static on a TV screen) that is used as input to the Generator in GANs to create diverse outputs.
*   **Discriminator Network:** In a GAN, the neural network responsible for distinguishing between real data and fake data produced by the Generator.
*   **Adversarial Game Network:** The competitive training setup in GANs where the Generator and Discriminator try to outperform each other.
*   **Loss Function:** A mathematical equation that measures how "wrong" a model's predictions are. The goal during training is to minimize this loss.
*   **Conditional GANs (CGANs):** A variant of GANs where the generator and discriminator are given additional information (a "condition"), allowing the generator to create specific types of data (e.g., generating an image of a dog, not just any animal).
*   **Compact Representation:** A smaller, more efficient numerical representation of data that still contains its essential information.
*   **NLP (Natural Language Processing):** A field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language.
*   **Sentiment Detection:** The task of determining the emotional tone (positive, negative, neutral) of a piece of text.
*   **Vector Representation:** Representing data (like words, images, or nodes) as a list of numbers (a vector) that captures its characteristics and relationships.
*   **Eigenvector:** A special vector that, when transformed by a matrix, only changes by a scalar factor (its direction remains the same). In PCA, they represent the principal components.
*   **Covariance Matrix:** A square table that shows how different features in a dataset vary together. It's used in PCA to find the directions of maximum variance.
*   **Principal Component:** A new direction in the data that captures the most variance. The first principal component captures the most, the second the next most, and so on.
*   **Variance:** A measure of how spread out a set of numbers is from its average value.
*   **Cluster Membership:** The assignment of a data point to a specific cluster.
*   **Stochastic Process:** A collection of random variables indexed by time, often used to model sequences of events, like random walks.
*   **Neural Node Embeddings:** Numerical representations (vectors) for nodes in a graph, learned using neural networks.
*   **Hierarchical Tree Structure:** A tree-like organization of data, used in some models (like hierarchical softmax) to speed up calculations by breaking down a large prediction problem into a series of smaller binary choices.
*   **Huffman Coding:** A method for compressing data by assigning shorter codes to more frequent items and longer codes to less frequent ones, often used to build hierarchical trees.
*   **Random Walk:** A path through a graph where each step is chosen randomly from the available connections.
*   **Node Features:** Characteristics or attributes associated with individual nodes in a graph.
*   **Neighborhood:** The set of nodes directly connected to a given node in a graph, or nodes within a certain distance.
*   **Breadth-First Search (BFS):** A graph traversal algorithm that explores all the neighbor nodes at the current depth level before moving on to nodes at the next depth level. It explores locally.
*   **Depth-First Search (DFS):** A graph traversal algorithm that explores as far as possible along each branch before backtracking. It explores globally.
*   **Hyperparameters (P, Q):** Settings or configurations of an algorithm that are set *before* the training process begins (e.g., the number of clusters K in K-Means, or the return and in-out parameters in Node2Vec).
*   **Embedding Space:** A multi-dimensional numerical space where data points (like words, images, or nodes) are represented as vectors. Similar items are placed closer together in this space.
*   **Loss Functions (Contrastive Loss, Triplet Loss):** Specific types of loss functions used in contrastive learning to push dissimilar items apart and pull similar items closer in the embedding space.
*   **Anchor (Sample):** In contrastive learning, the reference data point against which other samples are compared.
*   **Positive (Sample):** A data point that is similar to the anchor and should be pulled closer in the embedding space.
*   **Negative (Sample):** A data point that is dissimilar to the anchor and should be pushed farther away in the embedding space.
*   **Margin (M):** A hyperparameter in contrastive and triplet loss functions that defines a minimum desired separation distance between positive and negative samples.
*   **Hard Negatives:** Negative samples that are particularly difficult for the model to distinguish from positive samples because their representations are very similar. Mining these helps the model learn better.
*   **Data Augmentation (SimCLR specific):** Techniques like cropping, resizing, color distortion, rotation, adding noise, and cutout (masking parts of the image) used to create varied versions of an input image.
*   **Minibatch:** A small subset of the total training data that is processed at one time during a training iteration.
*   **Non-linear Transformation (G Network):** In SimCLR, an additional neural network (G) applied on top of the main feature extractor (F) to project representations into a space where the contrastive loss is applied. It helps preserve more useful information in the F network.
*   **Cosine Similarity:** A measure of similarity between two non-zero vectors that indicates whether vectors are pointing in roughly the same direction. Values range from -1 (opposite) to 1 (identical).
*   **Softmax (for Loss Function/Probabilities):** A mathematical function that converts a vector of numbers into a probability distribution, where the sum of probabilities is 1. Used in various models to get probabilities.
*   **Top-1 Accuracy:** A common metric for classification tasks, indicating the percentage of times the model's top predicted class matches the true class.
*   **ResNet:** A type of convolutional neural network architecture known for its "residual connections" that allow training of very deep networks.
*   **Transfer Learning:** A machine learning technique where a model trained on one task (e.g., image recognition on a large dataset) is repurposed or "fine-tuned" for a different but related task.
*   **One-Hot Representation:** A simple numerical representation where each item is represented by a vector of all zeros except for a single "1" at the index corresponding to that item. Limited for capturing meaning.
*   **Distributed Representation:** Representing items (like words) using dense numerical vectors where the meaning is spread across multiple dimensions.
*   **Dense Vectors:** Numerical vectors where most or all of the elements have non-zero values, unlike sparse vectors.
*   **Low-Dimensional Space:** A simplified numerical representation with fewer features compared to the original data, often used to capture key characteristics.
*   **Distributional Semantics (Distributional Hypothesis):** The idea that words that appear in similar contexts tend to have similar meanings.
*   **Target Word:** In Word2Vec, the word whose embedding is currently being learned.
*   **Context Words:** The words that appear around the target word within a defined window.
*   **Corpus (C):** A large collection of text or data used for training language models.
*   **Vocabulary (V):** The complete set of unique words or tokens in a given corpus.
*   **Word Vector:** The numerical representation (embedding) of a word when it is used as a target word.
*   **Context Vector:** The numerical representation (embedding) of a word when it is used as a context word.
*   **Continuous Bag of Words (CBOW):** A Word2Vec architecture that predicts a target word given its surrounding context words.
*   **Skip-gram (Model for Word2Vec):** A Word2Vec architecture that predicts the surrounding context words given a target word.
*   **Dot Product:** A mathematical operation between two vectors that results in a single number, indicating their similarity or alignment.
*   **Stochastic Gradient Descent (SGD):** An optimization algorithm used to train machine learning models by iteratively updating model parameters based on the gradient of the loss function calculated on small batches of data.
*   **Adam (Optimizer):** A popular optimization algorithm that adapts the learning rate for each parameter, often leading to faster convergence.
*   **T-SNE (t-Distributed Stochastic Neighbor Embedding):** A dimensionality reduction technique specifically used for visualizing high-dimensional data in 2D or 3D, preserving local relationships.
*   **Negative Sampling:** A technique used to speed up the training of Word2Vec and similar models by simplifying the softmax calculation. Instead of considering all words in the vocabulary, it only compares the correct context word with a few randomly chosen "negative" (incorrect) words.
*   **Noise Contrastive Estimation (NCE):** A more general statistical technique that negative sampling is based on, which approximates the full softmax.
*   **Logistic Regression:** A statistical model used for binary classification, predicting the probability of an event occurring (e.g., whether two words occur in context).
*   **Binary Classification Task:** A classification problem where there are only two possible output classes (e.g., real/fake, yes/no).
*   **Unigram Distribution:** A probability distribution where each word's probability is based solely on its individual frequency in the corpus, without considering context.
*   **Downweighting:** Reducing the importance or influence of certain elements, often frequent but less informative words in NLP.
*   **Elmo (Embeddings from Language Models):** A context-aware word embedding model that uses a bidirectional LSTM to create word representations that change based on the surrounding words in a sentence.
*   **Bidirectional LSTM (Long Short-Term Memory):** A type of recurrent neural network that processes sequences in both forward and backward directions to capture context from both sides.
*   **Language Model (Forward/Backward):** A model that predicts the next word in a sequence (forward) or the previous word in a sequence (backward), based on the words already seen.
*   **Token (in NLP):** A basic unit of text, usually a word or a punctuation mark.
*   **Transformer (Architecture):** A neural network architecture that relies on "self-attention" mechanisms to weigh the importance of different parts of the input sequence. It's highly effective for language tasks.
*   **Encoder (of Transformer):** The part of the Transformer architecture that processes the input sequence and generates contextualized representations for each token.
*   **Self-Attention:** A mechanism in Transformers that allows each word in a sequence to "pay attention" to other words in the same sequence, determining their relevance to its own meaning.
*   **Key, Query, Value Vectors:** Components used in the self-attention mechanism within Transformers to calculate attention scores and derive contextualized representations.
*   **Pre-training:** The initial phase of training a large language model on a massive amount of unlabeled data to learn general language understanding.
*   **Fine-tuning:** The subsequent phase after pre-training, where a pre-trained model is further trained on a smaller, labeled dataset for a specific task.
*   **Masked Language Modeling (MLM):** A pre-training task used by BERT where some words in the input sentence are randomly "masked," and the model learns to predict those masked words based on their context.
*   **Next Sentence Prediction (NSP):** A pre-training task used by BERT where the model predicts whether a given sentence B logically follows sentence A.
*   **Special Tokens ([CLS], [SEP]):** Unique markers added to input sentences for BERT and similar models to indicate the beginning of a sequence ([CLS]) or the separation between sentences ([SEP]).
*   **Knowledge Graphs:** Graphs that represent information as a network of interconnected entities and their relationships (e.g., "Eiffel Tower located in Paris").
*   **Social Graphs:** Graphs that represent social connections between people (e.g., friends on Facebook).
*   **Product Graphs:** Graphs that represent relationships between products (e.g., "Product A is an accessory for Product B").
*   **Nodes (V):** The individual entities or points in a graph (also called vertices).
*   **Edges (E):** The connections or relationships between nodes in a graph.
*   **Directed/Undirected (Edges):** Directed edges have a specific flow (e.g., "A follows B"), while undirected edges simply show a connection (e.g., "A is friends with B").
*   **Weighted/Unweighted (Edges):** Weighted edges have a numerical value indicating the strength or cost of the connection, while unweighted edges do not.
*   **Modality (Predictions about Nodes, Edges, Subgraphs, Whole Graph):** Refers to what specific part of the graph the model is making predictions about.
*   **Unnormalized Probability:** A value that indicates a likelihood but has not been scaled to sum to 1 (like a true probability).
*   **Return Parameter (P) / In-Out Parameter (Q):** Hyperparameters in Node2Vec that control the bias of the random walk, influencing whether it explores locally (BFS-like) or globally (DFS-like).
*   **Loss of Information Induced by the Contrastive Loss:** A potential side effect where the contrastive loss, by making representations invariant to certain transformations (like color), might remove information that could be useful for other downstream tasks.

## Important Points

*   **Why Unsupervised Learning is Important:**
    *   Vast amounts of unlabeled data are available, which can provide meaningful insights if leveraged properly.
    *   Some problems are inherently unsupervised (e.g., improving image resolution where there's no single "correct" high-resolution image).
    *   It can efficiently tackle problems where labeled data is scarce or changes over time (e.g., anomaly detection, where normal behavior evolves).
*   **Types of Unsupervised Learning Problems:** Dimensionality Reduction, Clustering, Generative Modeling, and Representation Learning are the four main types covered.
*   **Benefits of Dimensionality Reduction:** Easier visualization of high-dimensional data, creation of smaller and simpler models, and data compression.
*   **PCA's Core Idea:** Projects data onto axes that capture the maximum variance, which mathematically correspond to the eigenvectors of the data's covariance matrix.
*   **K-Means Algorithm Steps:** Initialize K cluster centers randomly, assign each data point to the closest center, update centers by taking the mean of assigned points, and repeat until convergence.
*   **K-Means for Image Compression:** Pixels (RGB values) are treated as data points, clustered, and then the original pixel is replaced by its cluster's centroid color. This reduces the number of unique colors to store, achieving compression.
*   **GANs' Adversarial Training:** The generator and discriminator networks learn by playing a continuous game, leading to increasingly realistic generated data and improved fake detection.
*   **Challenges of Word Representations:** One-hot encoding is simple but doesn't capture meaning and can be very large. Distributed representations (embeddings) overcome these issues by using dense, low-dimensional vectors.
*   **Word2Vec's Distributional Hypothesis:** The idea that a word's meaning is derived from the words it frequently appears with.
*   **Speeding up Word2Vec:** Negative sampling addresses the computational cost of softmax by turning a multi-class classification problem into several binary classification problems.
*   **Elmo vs. BERT:** Elmo uses shallow combinations of unidirectional LSTMs, while BERT uses deep bidirectional Transformers, providing a richer contextual understanding.
*   **BERT's Training Tasks:** Masked Language Modeling (predicting masked words) and Next Sentence Prediction (predicting if sentences follow each other) enable BERT to understand both word-level and sentence-level context.
*   **Graph Applications:** Graphs are versatile for modeling relationships and have applications in network classification, recommendation systems, anomaly detection, and missing link prediction.
*   **Graph to Sequence Conversion (DeepWalk/Node2Vec):** Random walks are used to transform the graph structure into sequences of nodes, which can then be processed like sentences to learn node embeddings.
*   **Node2Vec's Biased Random Walks:** The P (return) and Q (in-out) hyperparameters allow control over the random walk, enabling the model to prioritize local (BFS-like) or global (DFS-like) graph structure, capturing different types of node relationships.
*   **Motivation for Self-supervised Learning in Vision:** Overcomes the expense of annotating large image datasets, allowing the use of vast amounts of unlabeled image data for pre-training feature extractors.
*   **SimCLR's Core Idea:** Learns image features by maximizing the agreement between different augmented views of the *same* image (positive pairs) and pushing them away from augmented views of *other* images (negative pairs) within a batch.
*   **SimCLR's Architecture (F & G):** The F network extracts initial representations, and the G network (a small nonlinear head) transforms these for the contrastive loss calculation. After training, G is discarded, and F provides features for downstream tasks.

## Summary

Unsupervised learning is a powerful branch of AI that extracts patterns and insights from unlabeled data. It addresses challenges like vast amounts of unannotated information, inherently unsupervised problems (e.g., image enhancement), and evolving data behaviors (e.g., anomaly detection). Key techniques include **Dimensionality Reduction** (like PCA, for simplifying data and visualization), **Clustering** (like K-Means, for grouping similar items), and **Generative Modeling** (like GANs, for creating new, realistic data by learning its underlying distribution through an adversarial game between a generator and a discriminator).

Furthermore, **Representation Learning** is crucial for converting complex data (text, graphs, images) into numerical vectors that machines can understand. For text, models evolved from simple one-hot encoding to dense **Word2Vec** embeddings (capturing word similarity from context), and then to advanced, context-aware **BERT** models (using bidirectional Transformers and tasks like masked language modeling). For graphs, techniques like **DeepWalk** and **Node2Vec** use random walks to learn node embeddings that capture structural similarities. In computer vision, **Self-supervised Contrastive Learning**, exemplified by **SimCLR**, has achieved remarkable success by learning robust image features from unlabeled data through clever data augmentations and a loss function that brings similar views closer and dissimilar views farther apart in an embedding space. These unsupervised methods are essential for leveraging the enormous amount of unlabeled data available today.

## Additional Resources

*   **Further Explore Specific Algorithms:**
    *   **PCA:** Look up how eigenvalues and eigenvectors are calculated and how they relate to variance.
    *   **K-Means:** Understand different distance metrics (e.g., Euclidean, Manhattan) and methods for choosing the optimal 'K' (e.g., elbow method, silhouette score).
    *   **GANs:** Dive deeper into the mathematical loss functions, different GAN architectures (DCGAN, CycleGAN, StyleGAN), and their creative applications.
    *   **Word Embeddings:** Research GloVe and FastText, other popular word embedding models, and how they compare to Word2Vec.
    *   **Transformers:** Learn about the Transformer architecture in detail, including multi-head attention and positional encodings, which are fundamental to models like BERT.
    *   **Graph Neural Networks (GNNs):** Explore more advanced techniques for learning on graphs beyond just node embeddings, such as Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs).
    *   **Self-supervised Learning:** Look into other self-supervised methods for vision like MoCo, BYOL, and DINO.

*   **Practical Applications:**
    *   Explore real-world use cases of unsupervised learning in fields like finance (fraud detection), healthcare (patient phenotyping), marketing (customer segmentation), and content recommendation.

*   **Ethical Considerations:**
    *   Discuss the potential biases in data that can be perpetuated or amplified by unsupervised learning models and how to mitigate them.## Study Guide: Unsupervised Learning

## Overview
This video series provides a comprehensive introduction to unsupervised learning, a crucial area in machine learning. It covers the core concepts, common techniques, and practical applications across different data types like images, text, and graphs. The first part introduces the fundamentals of unsupervised learning, contrasting it with supervised learning, and delves into dimensionality reduction, clustering, and generative modeling. The second part focuses on representation learning, explaining how machines can learn meaningful numerical representations of complex data such as words, entire sentences, graphs, and images, even without explicit labels.

## Key Concepts

*   **Unsupervised Learning:** This is a type of machine learning where the algorithm tries to find patterns and structures in data without being given any pre-existing labels or categories. Imagine having a big box of unsorted toys and trying to group them into different categories (cars, dolls, blocks) all by yourself, without anyone telling you what each toy is. The goal is to discover hidden insights or organization within the data.

*   **Supervised Learning:** In contrast to unsupervised learning, supervised learning involves training a model using data that already has "labels." For example, if you want to teach a computer to identify cats and dogs, you'd feed it many pictures, each clearly marked "cat" or "dog."

*   **Reinforcement Learning:** A type of machine learning where an "agent" learns to make decisions by performing actions in an environment to maximize a reward. It's like training a pet by giving it treats for good behavior.

*   **Dimensionality Reduction:** This technique aims to simplify complex data by reducing the number of "dimensions" or features while keeping the most important information. Imagine squishing a super-detailed 3D drawing onto a 2D piece of paper without losing the main features. This helps visualize data, build smaller models, and compress information.

*   **Clustering:** The process of grouping similar data points together into "clusters," while ensuring that data points in different clusters are as dissimilar as possible. Think of sorting your LEGO bricks by color or shape without being told which colors or shapes exist – you just find the natural groupings.

*   **Generative Modeling:** This involves creating models that learn the underlying "distribution" of the input data. In simpler terms, it learns how the data was generated. Once a generative model understands this, it can then generate new, realistic samples that resemble the original data. For instance, a model could learn from many cat pictures and then create entirely new, believable cat images.

*   **Representation Learning:** The process of automatically discovering good "representations" or numerical codes for raw data. This is especially important for non-numeric data like text, images, or graphs. The goal is to turn complex information into a simpler, more compact form (often a vector of numbers) that computers can easily process and use for various tasks.

*   **Principal Component Analysis (PCA):** A popular technique for dimensionality reduction. It works by finding the directions (called "principal components") in your data where it spreads out the most (has the most "variance"). By projecting the data onto these main directions, you can reduce its dimensions while retaining as much important information as possible.

*   **K-Means Clustering:** A widely used centroid-based clustering algorithm. It groups data points into a pre-defined number of clusters (K). Each cluster is represented by its "centroid" (the average position of all data points in that cluster). The algorithm iteratively assigns points to the nearest centroid and then updates the centroids until the clusters are stable.

*   **Generative Adversarial Networks (GANs):** A powerful type of generative model consisting of two neural networks: a "Generator" and a "Discriminator," which compete against each other in an "adversarial game." The Generator tries to create realistic fake data (e.g., images), while the Discriminator tries to distinguish between real and fake data. Both networks improve over time, with the Generator becoming better at producing fakes that fool the Discriminator, and the Discriminator becoming better at spotting fakes.

*   **Word2Vec:** A technique in natural language processing (NLP) that learns to represent words as dense numerical vectors (embeddings). It's based on the "distributional hypothesis," which suggests that words appearing in similar contexts tend to have similar meanings. It has two main architectures: Continuous Bag of Words (CBOW) and Skip-gram.

*   **BERT (Bidirectional Encoder Representations from Transformers):** A highly influential language model that generates word representations by considering the full context of a word within a sentence, both from the left and the right. Unlike older models, BERT can understand the deep meaning of words based on how they relate to all other words around them. It's pre-trained on large amounts of text and can then be fine-tuned for specific tasks.

*   **DeepWalk:** An early neural network approach for learning "node embeddings" (numerical representations for nodes) in graphs. It uses "random walks" (random paths through the graph) to generate sequences of nodes, which are then treated like sentences. These sequences are fed into a Skip-gram model (similar to Word2Vec) to learn embeddings that capture the graph's structure.

*   **Node2Vec:** An improvement over DeepWalk, Node2Vec uses "biased random walks" that can be controlled to explore either the immediate neighborhood of a node (like a Breadth-First Search, BFS) or venture further away (like a Depth-First Search, DFS). This allows the learned node embeddings to capture different aspects of the graph's structure, either local or global.

*   **Self-supervised Learning:** A paradigm within unsupervised learning where the model creates its own "supervision" or "labels" from the unlabeled data itself. It designs a "pretext task" that allows the model to learn useful features without human annotations.

*   **Contrastive Learning:** A self-supervised learning approach that learns meaningful representations by contrasting similar and dissimilar data points. It tries to pull representations of "positive pairs" (similar items) closer together in an embedding space while pushing "negative pairs" (dissimilar items) farther apart.

*   **SimCLR (A Simple Framework for Contrastive Learning of Visual Representations):** A powerful framework for self-supervised contrastive learning that learns high-quality visual features from unlabeled images. It works by creating different augmented versions of the same image (positive pairs) and then maximizes the similarity between their learned representations, while minimizing similarity with other images in the batch (negative pairs).

## Technical Terms

*   **Input Data (X):** The raw information given to a machine learning model, like images, text, or numbers.
*   **Label (Y):** The correct answer or category associated with a piece of input data in supervised learning.
*   **Interesting Pattern:** A subjective term in unsupervised learning, referring to the hidden structures, groups, or relationships that the algorithm aims to discover in data.
*   **Unlabeled Data:** Data that does not have pre-assigned categories or classifications. This is very common and a key target for unsupervised learning.
*   **Image Super-Resolution:** The task of taking a low-resolution image and creating a higher-resolution version of it.
*   **Anomaly Detection:** Identifying unusual patterns or outliers in data that do not conform to expected behavior. Used for fraud detection or spotting unusual system activity.
*   **Density (of Normal Data Points):** In anomaly detection, this refers to how concentrated or spread out normal (non-anomalous) data points are in a given space. Anomalies often fall into low-density regions.
*   **High-Dimensional Vector Space:** A mathematical space with many "dimensions" or features. For example, an image with thousands of pixels can be considered high-dimensional.
*   **Low-Dimensional Vector Space:** A mathematical space with fewer "dimensions" or features, simplifying data for easier analysis or processing.
*   **Model Parameters:** The internal settings or values that a machine learning model learns during training to make predictions or find patterns.
*   **Compression:** Reducing the size of data while trying to maintain its quality and important information.
*   **Centroid-Based Clustering:** A type of clustering where each group is represented by a central point (the centroid) that is the mean of all data points in that group.
*   **Connectivity-Based Clustering:** A type of clustering that groups objects into a single cluster if they are connected to each other, forming a chain or network.
*   **Probabilistic-Based Clustering:** Clustering methods where each data point can belong to multiple clusters with a certain probability, rather than being assigned exclusively to one.
*   **Hard Assignment:** When a data point is assigned strictly to only one cluster.
*   **Distance Metric:** A rule or formula used to calculate how "far apart" or "similar" two data points are in a given space (e.g., Euclidean distance).
*   **Iterative Process:** A process that involves repeating a set of steps multiple times, usually to refine a solution or reach a desired outcome.
*   **Convergence:** In iterative algorithms, convergence occurs when the changes between iterations become very small, indicating that the algorithm has found a stable solution and is no longer significantly improving.
*   **Pixel:** The smallest unit of an image, typically represented by its color values (e.g., RGB).
*   **RGB Value:** A way to represent colors using a combination of red, green, and blue light intensities. Each pixel in a color image has an RGB value.
*   **Distribution (of Underlying Data):** The way data points are spread out or arranged in a dataset. Generative models aim to learn this pattern.
*   **New Samples:** Data points that are created by a generative model, resembling the original data it was trained on.
*   **Data Augmentation:** Creating new, modified versions of existing data (e.g., rotating images, adding noise to text) to increase the size and diversity of a training dataset.
*   **Classification:** The task of assigning an input data point to one of several predefined categories or classes.
*   **Regression:** The task of predicting a continuous numerical value (e.g., predicting house prices).
*   **Missing Data:** Gaps or absent values in a dataset that generative models can sometimes fill in.
*   **Density Estimation Models:** Models that explicitly try to estimate the probability distribution from which data points were generated.
*   **Multi-Gaussian Model:** A statistical model that assumes data is distributed as a combination of multiple Gaussian (bell-curve) distributions.
*   **Maximum Likelihood Estimation (MLE):** A method for estimating the parameters of a statistical model by finding the parameter values that make the observed data most probable.
*   **Generator Network:** In a GAN, the neural network responsible for creating synthetic data (e.g., fake images).
*   **Random Noise:** Random, unstructured data (like static on a TV screen) that is used as input to the Generator in GANs to create diverse outputs.
*   **Discriminator Network:** In a GAN, the neural network responsible for distinguishing between real data and fake data produced by the Generator.
*   **Adversarial Game Network:** The competitive training setup in GANs where the Generator and Discriminator try to outperform each other.
*   **Loss Function:** A mathematical equation that measures how "wrong" a model's predictions are. The goal during training is to minimize this loss.
*   **Conditional GANs (CGANs):** A variant of GANs where the generator and discriminator are given additional information (a "condition"), allowing the generator to create specific types of data (e.g., generating an image of a dog, not just any animal).
*   **Compact Representation:** A smaller, more efficient numerical representation of data that still contains its essential information.
*   **NLP (Natural Language Processing):** A field of artificial intelligence that focuses on enabling computers to understand, interpret, and generate human language.
*   **Sentiment Detection:** The task of determining the emotional tone (positive, negative, neutral) of a piece of text.
*   **Vector Representation:** Representing data (like words, images, or nodes) as a list of numbers (a vector) that captures its characteristics and relationships.
*   **Eigenvector:** A special vector that, when transformed by a matrix, only changes by a scalar factor (its direction remains the same). In PCA, they represent the principal components.
*   **Covariance Matrix:** A square table that shows how different features in a dataset vary together. It's used in PCA to find the directions of maximum variance.
*   **Principal Component:** A new direction in the data that captures the most variance. The first principal component captures the most, the second the next most, and so on.
*   **Variance:** A measure of how spread out a set of numbers is from its average value.
*   **Cluster Membership:** The assignment of a data point to a specific cluster.
*   **Stochastic Process:** A collection of random variables indexed by time, often used to model sequences of events, like random walks.
*   **Neural Node Embeddings:** Numerical representations (vectors) for nodes in a graph, learned using neural networks.
*   **Hierarchical Tree Structure:** A tree-like organization of data, used in some models (like hierarchical softmax) to speed up calculations by breaking down a large prediction problem into a series of smaller binary choices.
*   **Huffman Coding:** A method for compressing data by assigning shorter codes to more frequent items and longer codes to less frequent ones, often used to build hierarchical trees.
*   **Random Walk:** A path through a graph where each step is chosen randomly from the available connections.
*   **Node Features:** Characteristics or attributes associated with individual nodes in a graph.
*   **Neighborhood:** The set of nodes directly connected to a given node in a graph, or nodes within a certain distance.
*   **Breadth-First Search (BFS):** A graph traversal algorithm that explores all the neighbor nodes at the current depth level before moving on to nodes at the next depth level. It explores locally.
*   **Depth-First Search (DFS):** A graph traversal algorithm that explores as far as possible along each branch before backtracking. It explores globally.
*   **Hyperparameters (P, Q):** Settings or configurations of an algorithm that are set *before* the training process begins (e.g., the number of clusters K in K-Means, or the return and in-out parameters in Node2Vec).
*   **Embedding Space:** A multi-dimensional numerical space where data points (like words, images, or nodes) are represented as vectors. Similar items are placed closer together in this space.
*   **Loss Functions (Contrastive Loss, Triplet Loss):** Specific types of loss functions used in contrastive learning to push dissimilar items apart and pull similar items closer in the embedding space.
*   **Anchor (Sample):** In contrastive learning, the reference data point against which other samples are compared.
*   **Positive (Sample):** A data point that is similar to the anchor and should be pulled closer in the embedding space.
*   **Negative (Sample):** A data point that is dissimilar to the anchor and should be pushed farther away in the embedding space.
*   **Margin (M):** A hyperparameter in contrastive and triplet loss functions that defines a minimum desired separation distance between positive and negative samples.
*   **Hard Negatives:** Negative samples that are particularly difficult for the model to distinguish from positive samples because their representations are very similar. Mining these helps the model learn better.
*   **Data Augmentation (SimCLR specific):** Techniques like cropping, resizing, color distortion, rotation, adding noise, and cutout (masking parts of the image) used to create varied versions of an input image.
*   **Minibatch:** A small subset of the total training data that is processed at one time during a training iteration.
*   **Non-linear Transformation (G Network):** In SimCLR, an additional neural network (G) applied on top of the main feature extractor (F) to project representations into a space where the contrastive loss is applied. It helps preserve more useful information in the F network.
*   **Cosine Similarity:** A measure of similarity between two non-zero vectors that indicates whether vectors are pointing in roughly the same direction. Values range from -1 (opposite) to 1 (identical).
*   **Softmax (for Loss Function/Probabilities):** A mathematical function that converts a vector of numbers into a probability distribution, where the sum of probabilities is 1. Used in various models to get probabilities.
*   **Top-1 Accuracy:** A common metric for classification tasks, indicating the percentage of times the model's top predicted class matches the true class.
*   **ResNet:** A type of convolutional neural network architecture known for its "residual connections" that allow training of very deep networks.
*   **Transfer Learning:** A machine learning technique where a model trained on one task (e.g., image recognition on a large dataset) is repurposed or "fine-tuned" for a different but related task.
*   **One-Hot Representation:** A simple numerical representation where each item is represented by a vector of all zeros except for a single "1" at the index corresponding to that item. Limited for capturing meaning.
*   **Distributed Representation:** Representing items (like words) using dense numerical vectors where the meaning is spread across multiple dimensions.
*   **Dense Vectors:** Numerical vectors where most or all of the elements have non-zero values, unlike sparse vectors.
*   **Low-Dimensional Space:** A simplified numerical representation with fewer features compared to the original data, often used to capture key characteristics.
*   **Distributional Semantics (Distributional Hypothesis):** The idea that words that appear in similar contexts tend to have similar meanings.
*   **Target Word:** In Word2Vec, the word whose embedding is currently being learned.
*   **Context Words:** The words that appear around the target word within a defined window.
*   **Corpus (C):** A large collection of text or data used for training language models.
*   **Vocabulary (V):** The complete set of unique words or tokens in a given corpus.
*   **Word Vector:** The numerical representation (embedding) of a word when it is used as a target word.
*   **Context Vector:** The numerical representation (embedding) of a word when it is used as a context word.
*   **Continuous Bag of Words (CBOW):** A Word2Vec architecture that predicts a target word given its surrounding context words.
*   **Skip-gram (Model for Word2Vec):** A Word2Vec architecture that predicts the surrounding context words given a target word.
*   **Dot Product:** A mathematical operation between two vectors that results in a single number, indicating their similarity or alignment.
*   **Stochastic Gradient Descent (SGD):** An optimization algorithm used to train machine learning models by iteratively updating model parameters based on the gradient of the loss function calculated on small batches of data.
*   **Adam (Optimizer):** A popular optimization algorithm that adapts the learning rate for each parameter, often leading to faster convergence.
*   **T-SNE (t-Distributed Stochastic Neighbor Embedding):** A dimensionality reduction technique specifically used for visualizing high-dimensional data in 2D or 3D, preserving local relationships.
*   **Negative Sampling:** A technique used to speed up the training of Word2Vec and similar models by simplifying the softmax calculation. Instead of considering all words in the vocabulary, it only compares the correct context word with a few randomly chosen "negative" (incorrect) words.
*   **Noise Contrastive Estimation (NCE):** A more general statistical technique that negative sampling is based on, which approximates the full softmax.
*   **Logistic Regression:** A statistical model used for binary classification, predicting the probability of an event occurring (e.g., whether two words occur in context).
*   **Binary Classification Task:** A classification problem where there are only two possible output classes (e.g., real/fake, yes/no).
*   **Unigram Distribution:** A probability distribution where each word's probability is based solely on its individual frequency in the corpus, without considering context.
*   **Downweighting:** Reducing the importance or influence of certain elements, often frequent but less informative words in NLP.
*   **Elmo (Embeddings from Language Models):** A context-aware word embedding model that uses a bidirectional LSTM to create word representations that change based on the surrounding words in a sentence.
*   **Bidirectional LSTM (Long Short-Term Memory):** A type of recurrent neural network that processes sequences in both forward and backward directions to capture context from both sides.
*   **Language Model (Forward/Backward):** A model that predicts the next word in a sequence (forward) or the previous word in a sequence (backward), based on the words already seen.
*   **Token (in NLP):** A basic unit of text, usually a word or a punctuation mark.
*   **Transformer (Architecture):** A neural network architecture that relies on "self-attention" mechanisms to weigh the importance of different parts of the input sequence. It's highly effective for language tasks.
*   **Encoder (of Transformer):** The part of the Transformer architecture that processes the input sequence and generates contextualized representations for each token.
*   **Self-Attention:** A mechanism in Transformers that allows each word in a sequence to "pay attention" to other words in the same sequence, determining their relevance to its own meaning.
*   **Key, Query, Value Vectors:** Components used in the self-attention mechanism within Transformers to calculate attention scores and derive contextualized representations.
*   **Pre-training:** The initial phase of training a large language model on a massive amount of unlabeled data to learn general language understanding.
*   **Fine-tuning:** The subsequent phase after pre-training, where a pre-trained model is further trained on a smaller, labeled dataset for a specific task.
*   **Masked Language Modeling (MLM):** A pre-training task used by BERT where some words in the input sentence are randomly "masked," and the model learns to predict those masked words based on their context.
*   **Next Sentence Prediction (NSP):** A pre-training task used by BERT where the model predicts whether a given sentence B logically follows sentence A.
*   **Special Tokens ([CLS], [SEP]):** Unique markers added to input sentences for BERT and similar models to indicate the beginning of a sequence ([CLS]) or the separation between sentences ([SEP]).
*   **Knowledge Graphs:** Graphs that represent information as a network of interconnected entities and their relationships (e.g., "Eiffel Tower located in Paris").
*   **Social Graphs:** Graphs that represent social connections between people (e.g., friends on Facebook).
*   **Product Graphs:** Graphs that represent relationships between products (e.g., "Product A is an accessory for Product B").
*   **Nodes (V):** The individual entities or points in a graph (also called vertices).
*   **Edges (E):** The connections or relationships between nodes in a graph.
*   **Directed/Undirected (Edges):** Directed edges have a specific flow (e.g., "A follows B"), while undirected edges simply show a connection (e.g., "A is friends with B").
*   **Weighted/Unweighted (Edges):** Weighted edges have a numerical value indicating the strength or cost of the connection, while unweighted edges do not.
*   **Modality (Predictions about Nodes, Edges, Subgraphs, Whole Graph):** Refers to what specific part of the graph the model is making predictions about.
*   **Unnormalized Probability:** A value that indicates a likelihood but has not been scaled to sum to 1 (like a true probability).
*   **Return Parameter (P) / In-Out Parameter (Q):** Hyperparameters in Node2Vec that control the bias of the random walk, influencing whether it explores locally (BFS-like) or globally (DFS-like).
*   **Loss of Information Induced by the Contrastive Loss:** A potential side effect where the contrastive loss, by making representations invariant to certain transformations (like color), might remove information that could be useful for other downstream tasks.

## Important Points

*   **Why Unsupervised Learning is Important:**
    *   Vast amounts of unlabeled data are available, which can provide meaningful insights if leveraged properly.
    *   Some problems are inherently unsupervised (e.g., improving image resolution where there's no single "correct" high-resolution image).
    *   It can efficiently tackle problems where labeled data is scarce or changes over time (e.g., anomaly detection, where normal behavior evolves).
*   **Types of Unsupervised Learning Problems:** Dimensionality Reduction, Clustering, Generative Modeling, and Representation Learning are the four main types covered.
*   **Benefits of Dimensionality Reduction:** Easier visualization of high-dimensional data, creation of smaller and simpler models, and data compression.
*   **PCA's Core Idea:** Projects data onto axes that capture the maximum variance, which mathematically correspond to the eigenvectors of the data's covariance matrix.
*   **K-Means Algorithm Steps:** Initialize K cluster centers randomly, assign each data point to the closest center, update centers by taking the mean of assigned points, and repeat until convergence.
*   **K-Means for Image Compression:** Pixels (RGB values) are treated as data points, clustered, and then the original pixel is replaced by its cluster's centroid color. This reduces the number of unique colors to store, achieving compression.
*   **GANs' Adversarial Training:** The generator and discriminator networks learn by playing a continuous game, leading to increasingly realistic generated data and improved fake detection.
*   **Challenges of Word Representations:** One-hot encoding is simple but doesn't capture meaning and can be very large. Distributed representations (embeddings) overcome these issues by using dense, low-dimensional vectors.
*   **Speeding up Word2Vec:** Negative sampling addresses the computational cost of softmax by turning a multi-class classification problem into several binary classification problems.
*   **Elmo vs. BERT:** Elmo uses shallow combinations of unidirectional LSTMs, while BERT uses deep bidirectional Transformers, providing a richer contextual understanding.
*   **BERT's Training Tasks:** Masked Language Modeling (predicting masked words) and Next Sentence Prediction (predicting if sentences follow each other) enable BERT to understand both word-level and sentence-level context.
*   **Graph Applications:** Graphs are versatile for modeling relationships and have applications in network classification, recommendation systems, anomaly detection, and missing link prediction.
*   **Graph to Sequence Conversion (DeepWalk/Node2Vec):** Random walks are used to transform the graph structure into sequences of nodes, which can then be processed like sentences to learn node embeddings.
*   **Node2Vec's Biased Random Walks:** The P (return) and Q (in-out) hyperparameters allow control over the random walk, enabling the model to prioritize local (BFS-like) or global (DFS-like) graph structure, capturing different types of node relationships.
*   **Motivation for Self-supervised Learning in Vision:** Overcomes the expense of annotating large image datasets, allowing the use of vast amounts of unlabeled image data for pre-training feature extractors.
*   **SimCLR's Core Idea:** Learns image features by maximizing the agreement between different augmented views of the *same* image (positive pairs) and pushing them away from augmented views of *other* images (negative pairs) within a batch.
*   **SimCLR's Architecture (F & G):** The F network extracts initial representations, and the G network (a small nonlinear head) transforms these for the contrastive loss calculation. After training, G is discarded, and F provides features for downstream tasks.

## Summary

Unsupervised learning is a powerful branch of AI that extracts patterns and insights from unlabeled data. It addresses challenges like vast amounts of unannotated information, inherently unsupervised problems (e.g., image enhancement), and evolving data behaviors (e.g., anomaly detection). Key techniques include **Dimensionality Reduction** (like PCA, for simplifying data and visualization), **Clustering** (like K-Means, for grouping similar items), and **Generative Modeling** (like GANs, for creating new, realistic data by learning its underlying distribution through an adversarial game between a generator and a discriminator).

Furthermore, **Representation Learning** is crucial for converting complex data (text, graphs, images) into numerical vectors that machines can understand. For text, models evolved from simple one-hot encoding to dense **Word2Vec** embeddings (capturing word similarity from context), and then to advanced, context-aware **BERT** models (using bidirectional Transformers and tasks like masked language modeling). For graphs, techniques like **DeepWalk** and **Node2Vec** use random walks to learn node embeddings that capture structural similarities. In computer vision, **Self-supervised Contrastive Learning**, exemplified by **SimCLR**, has achieved remarkable success by learning robust image features from unlabeled data through clever data augmentations and a loss function that brings similar views closer and dissimilar views farther apart in an embedding space. These unsupervised methods are essential for leveraging the enormous amount of unlabeled data available today.

## Additional Resources

*   **Further Explore Specific Algorithms:**
    *   **PCA:** Look up how eigenvalues and eigenvectors are calculated and how they relate to variance.
    *   **K-Means:** Understand different distance metrics (e.g., Euclidean, Manhattan) and methods for choosing the optimal 'K' (e.g., elbow method, silhouette score).
    *   **GANs:** Dive deeper into the mathematical loss functions, different GAN architectures (DCGAN, CycleGAN, StyleGAN), and their creative applications.
    *   **Word Embeddings:** Research GloVe and FastText, other popular word embedding models, and how they compare to Word2Vec.
    *   **Transformers:** Learn about the Transformer architecture in detail, including multi-head attention and positional encodings, which are fundamental to models like BERT.
    *   **Graph Neural Networks (GNNs):** Explore more advanced techniques for learning on graphs beyond just node embeddings, such as Graph Convolutional Networks (GCNs) and Graph Attention Networks (GATs).
    *   **Self-supervised Learning:** Look into other self-supervised methods for vision like MoCo, BYOL, and DINO.

*   **Practical Applications:**
    *   Explore real-world use cases of unsupervised learning in fields like finance (fraud detection), healthcare (patient phenotyping), marketing (customer segmentation), and content recommendation.

*   **Ethical Considerations:**
    *   Discuss the potential biases in data that can be perpetuated or amplified by unsupervised learning models and how to mitigate them.