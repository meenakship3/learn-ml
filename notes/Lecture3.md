# Study Guide: Dimensionality Reduction in Machine Learning

## Overview
This session from the Amazon ML Summer School 2022, led by Atu, a Senior Manager for Applied Science at Amazon, provides a comprehensive introduction to **dimensionality reduction** techniques in machine learning. The core problem addressed is how to handle datasets with a very large number of "features" (characteristics or inputs), which can overwhelm machine learning algorithms and make models hard to understand or slow to run. The session explores two main categories: **feature selection** (choosing a subset of existing features) and **feature extraction/transformation** (creating new, smaller features from combinations of old ones). It dives into specific methods like Singular Value Decomposition (SVD), Principal Component Analysis (PCA), Matrix Factorization (including Non-Negative Matrix Factorization - NMF), and t-Distributed Stochastic Neighbor Embedding (t-SNE), explaining their underlying mathematics, applications, and benefits.

## Key Concepts

*   **The Problem of High Dimensions**: Imagine you're trying to describe something, and you have thousands of tiny details. If you have too many details, it becomes hard to focus on what's truly important, and it takes a long time to sort through everything. In machine learning, having "too many features" (inputs or characteristics of your data) can make models slow, less accurate, and hard to understand.
*   **Dimensionality Reduction**: This is like summarizing a really long book into a much shorter one, but still keeping all the most important ideas. It's about reducing the number of "features" or "dimensions" in your data while trying to keep as much important information as possible.
*   **Two Main Approaches**:
    *   **Feature Selection**: This is like picking out only the most important ingredients from a long recipe and throwing away the ones that don't add much flavor or are just confusing. You select a *subset* of your original features.
    *   **Feature Extraction/Transformation**: This is like blending several ingredients to create a new, simpler, but still flavorful mix. You create *new* features that are combinations of the original ones, but in a smaller number.
*   **Why Reduce Dimensions?**
    *   **Improved Accuracy**: Less "noise" or irrelevant information can make your model learn better.
    *   **Reduced Search Space**: The model has fewer things to consider, making it faster to train and predict.
    *   **Reduced Prediction Time**: With fewer features, the model makes decisions quicker, which is important for real-time applications.
    *   **Interpretability**: It's easier to understand what your model is doing if it relies on fewer, more meaningful features.
*   **Feature Selection Techniques**:
    *   **Wrapper Methods**: These are like trying out different combinations of ingredients in your recipe, baking a cake with each combination, and seeing which cake tastes best. They use a machine learning model to evaluate different subsets of features. They can be slow because they try many combinations.
        *   *Greedy Algorithms*: These are like making the "best" choice at each small step, hoping it leads to the best overall result. It doesn't always guarantee the *absolute* best outcome, but it's often practical.
        *   *Sequential Forward Selection*: Start with no features, then add one feature at a time, always picking the one that improves the model the most. Stop when adding more features doesn't help.
        *   *Recursive Backward Elimination*: Start with all features, then remove one feature at a time, always removing the one that hurts the model the least (or improves it by being removed).
    *   **Filter Methods**: These are like ranking ingredients based on how good they smell or how healthy they are, *before* you even start cooking. They evaluate features based on their individual relationship with the outcome, without involving a specific machine learning model. They are fast but might miss how features work *together*.
        *   *Mutual Information*: Measures how much information one feature gives you about another, like how knowing if it's raining tells you something about whether people are carrying umbrellas.
        *   *Pearson Correlation Coefficient*: Measures how strongly two numerical things are linearly related (e.g., as one goes up, does the other tend to go up too?).
        *   *Chi-Square Statistic*: Used for categorical data to see if two categories are related.
    *   **Embedded Methods**: These are like a chef who knows exactly which ingredients to focus on *while* they are cooking, because their cooking process naturally highlights the important ones. They perform feature selection *during* the model training process, combining benefits of both wrapper and filter methods.
        *   *Lasso (L1 Norm)*: A technique that tries to make the "weights" (importance) of less useful features exactly zero, effectively getting rid of them. It promotes "sparsity."
        *   *Ridge Regression (L2 Norm)*: A technique that makes the "weights" of features very small, but usually not exactly zero. It helps prevent overfitting.
*   **Feature Extraction Techniques**:
    *   **Singular Value Decomposition (SVD)**: Imagine a big spreadsheet of numbers. SVD is a powerful mathematical tool that can break down this big spreadsheet into three smaller, simpler ones. These smaller spreadsheets represent the main patterns and variations in the original data, often in a much lower dimension. It's like finding the core components that make up your data.
    *   **Principal Component Analysis (PCA)**: Think of a cloud of dots in 3D space. PCA finds the best 2D flat surface (or a line) to project these dots onto, so that they spread out as much as possible and you lose the least amount of information. It essentially finds the "directions" in your data where there's the most variation. It's closely related to SVD.
    *   **Matrix Factorization**: This is like trying to guess movie ratings. If you know some people's ratings for some movies, Matrix Factorization tries to break down that incomplete rating table into two smaller, "hidden" tables: one describing user preferences (e.g., how much they like action vs. comedy) and another describing movie characteristics (e.g., how "action-packed" a movie is). By multiplying these hidden tables, you can fill in the missing ratings. It's especially useful for incomplete data, like in recommendation systems.
        *   *Non-Negative Matrix Factorization (NMF)*: A special type of Matrix Factorization where all the numbers in the hidden tables must be positive (non-negative). This often makes the "hidden" patterns (like topics in text or features in faces) more understandable and easy for humans to interpret.
    *   **t-Distributed Stochastic Neighbor Embedding (t-SNE)**: This is a visualization tool. Imagine you have data points spread out in many dimensions, and you want to squish them down to 2D or 3D so you can see them on a screen. t-SNE tries really hard to keep points that were close together in the high-dimensional space still close together in the lower-dimensional space. It's particularly good at showing "clusters" or groups of similar data points.
    *   **Autoencoders**: These are like special neural networks that learn to compress data into a smaller "code" (the "encoder") and then uncompress it back to its original form (the "decoder"). By trying to reconstruct the original data, they learn a very efficient, low-dimensional "representation" or "embedding" of the data. While PCA is a linear version, autoencoders can learn much more complex, non-linear compressions.

## Technical Terms

*   **Accuracy**: How correct a machine learning model's predictions are.
*   **Alternating Least Squares (ALS)**: A method used to solve Matrix Factorization problems by taking turns optimizing one of the smaller matrices while holding the other one fixed.
*   **Asymmetric Distance**: A way to measure how different two things are, where the "distance" from A to B is not necessarily the same as the distance from B to A (like KL Divergence).
*   **Autoencoders**: Special neural networks designed to learn efficient data representations by trying to compress input data and then reconstruct it.
*   **Bag of Words Representation**: A way to represent text data for machine learning where each document is treated as a "bag" (collection) of its words, ignoring grammar and word order, and just counting how many times each word appears.
*   **Basis Vectors**: A set of fundamental vectors that can be combined (like building blocks) to create any other vector in a space.
*   **Bias Term**: An extra value added to a model's prediction that helps it fit the data better, often representing a general tendency (e.g., a user's general tendency to give high ratings, or a movie's general popularity).
*   **Bigram**: A sequence of two words (e.g., "machine learning").
*   **Block Diagonal Structures**: A type of matrix where non-zero values are only found in square blocks along its main diagonal, and zeros everywhere else.
*   **Centered Data Matrix**: A data matrix where the average (mean) of each feature (column) has been subtracted from all values in that column, so that each feature now has an average of zero.
*   **Chi-Square Statistic (K Square statistic)**: A statistical test used to determine if there's a significant relationship between two categorical variables.
*   **Dimensionality Reduction**: The process of reducing the number of features (or variables) in a dataset while trying to keep as much important information as possible.
*   **Document Cross Topics**: In NMF for text, one of the smaller matrices that represents how much each document relates to various "topics" (the latent factors).
*   **Eigenvalue (Igen value)**: A special number associated with a matrix that tells you how much a vector is stretched or compressed when transformed by that matrix.
*   **Eigenvalue Problem**: A mathematical problem of finding special vectors (eigenvectors) that, when transformed by a matrix, only change in length, not direction.
*   **Eigenvector (Igen vector)**: A special vector that, when transformed by a matrix, only gets scaled (stretched or compressed) but doesn't change its direction.
*   **Embedded Methods**: A category of feature selection techniques where the feature selection process is built right into the machine learning model's training.
*   **Embeddings**: Low-dimensional numerical representations of data (like words, images, or users) that capture their meaning and relationships.
*   **Encoder-Decoder View**: An architecture common in deep learning where an "encoder" compresses data into a smaller representation, and a "decoder" reconstructs the original data from that representation.
*   **Feature Extraction**: A type of dimensionality reduction where new, lower-dimensional features are created from combinations of the original features.
*   **Feature Interactions**: When the effect of one feature on the outcome depends on the value of another feature (e.g., "sweetness" alone might not predict "good taste," but "sweetness" combined with "sourness" in a specific ratio might).
*   **Feature Set**: The collection of all input characteristics or variables used to describe something in a dataset.
*   **Feature Selection**: A type of dimensionality reduction where you choose a subset of the most relevant existing features from your original dataset.
*   **Filter Methods**: Feature selection techniques that rank or score features based on statistical measures (like correlation) *before* training a machine learning model.
*   **Gaussian Distribution**: A common type of probability distribution, often called the "bell curve," which describes how many natural phenomena are distributed around an average value.
*   **Gaussian Noise**: Randomness added following a Gaussian distribution, often used to help optimization algorithms avoid getting stuck.
*   **Global Minimum**: The lowest possible point (best solution) in an optimization problem's "landscape."
*   **Goodness Scores**: Metrics used in filter methods to evaluate how useful or relevant a feature is, often by how well it correlates with the target.
*   **Gradient Descent**: A common optimization algorithm used in machine learning to find the best parameters for a model by iteratively moving "downhill" along the slope (gradient) of the error function.
*   **Greedy Algorithms**: Algorithms that make the best possible local choice at each step with the hope that this will lead to a globally optimal or near-optimal solution.
*   **High Dimensional Space**: A conceptual space where data points are described by many features (dimensions), making it hard to visualize or work with directly.
*   **Image Compression**: Reducing the size of image data while trying to maintain its visual quality, often achieved using dimensionality reduction techniques.
*   **Imputation**: The process of filling in missing data points with estimated values.
*   **Incomplete Matrix**: A matrix (like a spreadsheet) that has some missing values, often represented as empty cells.
*   **Information Theory**: A mathematical field that studies how information is quantified, stored, and communicated, particularly in the presence of noise.
*   **Interpretability**: How easy it is for humans to understand how a machine learning model makes its decisions.
*   **Isolo Term**: In the context of Lasso and Ridge regression visualization, "isoloss" lines or contours represent points where the loss function (the error your model makes) has the same value.
*   **Kullback-Leibler Divergence (KL Divergence)**: A measure from information theory that quantifies how one probability distribution is different from a second, reference probability distribution. It's not a true "distance" because it's asymmetric.
*   **Lasso (L1 Norm)**: A regularization technique in machine learning that adds a penalty based on the sum of the absolute values of the model's "weights." It has the effect of forcing some weights to become exactly zero, thus performing feature selection.
*   **Latent Factors**: "Hidden" or underlying patterns or characteristics that are discovered by dimensionality reduction techniques, especially in Matrix Factorization.
*   **Latent Semantic Indexing Analysis (LSI)**: An application of SVD to text data (like a term-document matrix) to discover hidden "topics" or concepts within the documents and terms.
*   **Latency**: The time delay between giving an input to a system (like a machine learning model) and getting an output. Lower latency means faster response.
*   **Learning Algorithm**: The set of rules and procedures a machine learning model uses to learn patterns from data.
*   **Left Singular Vectors**: Special vectors that, along with right singular vectors and singular values, make up the Singular Value Decomposition of a matrix. They form an orthonormal basis.
*   **Linear Algebra**: A branch of mathematics that deals with vectors, matrices, and linear transformations; it's fundamental to many machine learning techniques.
*   **Linear Models**: Machine learning models where the relationship between inputs and outputs can be represented by a straight line or a flat plane.
*   **Local Distances**: How close data points are to each other in a specific, small neighborhood.
*   **Local Minima**: A point in an optimization problem's "landscape" that is lower than all nearby points, but not necessarily the overall lowest point (global minimum).
*   **Low Dimensional Space**: A conceptual space where data points are described by a smaller number of features (dimensions), making it easier to visualize and analyze.
*   **Machine Learning**: A field of artificial intelligence where computers learn from data without being explicitly programmed.
*   **Matrix Factorization**: A technique that decomposes a large matrix (like a user-item rating matrix) into two or more smaller matrices, often to uncover hidden "latent factors."
*   **Multi-dimensional Scaling (MDS)**: A technique that visualizes similarities or dissimilarities in data by representing them as distances in a lower-dimensional space.
*   **Multitask Learning**: Training a single machine learning model to perform multiple related tasks at the same time, allowing it to learn shared patterns.
*   **Mutual Information**: A measure of the shared information between two variables, indicating how much knowing one variable reduces uncertainty about the other.
*   **MNIST Data Set**: A very famous dataset in machine learning consisting of thousands of handwritten digits (0-9), often used for testing image classification algorithms.
*   **Noisy Features**: Features in a dataset that contain random errors or irrelevant information, which can confuse a machine learning model.
*   **Non-Negative Matrix Factorization (NMF)**: A type of Matrix Factorization where all values in the decomposed matrices are forced to be non-negative, often leading to more interpretable "parts" or "topics."
*   **Nonlinear Models**: Machine learning models where the relationship between inputs and outputs is not a straight line or flat plane but a more complex, curved relationship.
*   **Nonlinear Neural Networks**: Neural networks that use activation functions that are not linear, allowing them to learn complex, non-linear relationships in data.
*   **Numerical Features**: Inputs to a machine learning model that are represented by numbers (e.g., age, price, temperature).
*   **OCR (Optical Character Recognition)**: The technology that allows computers to "read" text from images (like scanned documents or handwritten notes).
*   **Online Setting**: In machine learning, where models are updated continuously as new data arrives, often in real-time.
*   **Optimization**: The process of finding the best set of parameters for a machine learning model to minimize its errors or maximize its performance.
*   **Orthogonal**: In mathematics, two vectors are orthogonal if they are perpendicular to each other (form a 90-degree angle). Their dot product is zero.
*   **Pearson Correlation Coefficient**: A statistical measure that indicates the strength and direction of a linear relationship between two numerical variables.
*   **Power Iteration Technique**: An iterative algorithm used to find the largest eigenvalue (and its corresponding eigenvector) of a matrix.
*   **Pre-trained Models**: Machine learning models that have already been trained on very large datasets for a general task (like understanding language) and can then be adapted for specific tasks. Examples include BERT, XLM-RoBERTa.
*   **Principal Component Analysis (PCA)**: A popular dimensionality reduction technique that finds new "principal components" (directions) in the data that capture the most variation. It's like finding the most important angles to look at your data from.
*   **Recommender Systems**: Systems that suggest items (like movies, products, or music) to users based on their past behavior or preferences.
*   **Reconstruction Error**: The difference between the original data and the data that has been compressed (reduced in dimension) and then uncompressed (reconstructed). A smaller error means the compression worked well.
*   **Recursive Backward Elimination**: A wrapper method for feature selection that starts with all features and iteratively removes the least important one until a stopping criterion is met.
*   **Redundant Features**: Features in a dataset that provide similar or overlapping information, often making other features unnecessary.
*   **Representation Learning**: The process by which a machine learning model learns to automatically discover good features or representations of the input data, often in a lower-dimensional space.
*   **Reduced Prediction Time**: When a machine learning model makes predictions faster due to fewer features or a simpler structure.
*   **Ridge Regression (L2 Norm)**: A regularization technique in machine learning that adds a penalty based on the sum of the squared values of the model's "weights." It discourages large weights, making the model simpler but usually doesn't force weights to zero.
*   **Right Singular Vectors**: Special vectors that, along with left singular vectors and singular values, make up the Singular Value Decomposition of a matrix. They form an orthonormal basis.
*   **Sammon Mapping**: Another non-linear dimensionality reduction technique, similar to t-SNE, used for visualization.
*   **Search Space**: The range of possible solutions or combinations that an algorithm explores when trying to solve a problem (e.g., all possible subsets of features).
*   **Sequential Forward Selection**: A greedy wrapper method for feature selection that starts with an empty set of features and iteratively adds the feature that best improves the model's performance.
*   **Side Information**: Additional data about users or items (beyond just their interactions) that can be incorporated into models like Matrix Factorization to improve recommendations.
*   **Signal Processing**: A field concerned with analyzing, modifying, and synthesizing signals (like audio, images, or sensor data).
*   **Singular Value Decomposition (SVD)**: A powerful mathematical technique that breaks down any matrix into three simpler matrices, revealing its core structure and allowing for dimensionality reduction.
*   **Singular Values**: Non-negative scalar values in SVD that represent the "strength" or importance of the underlying patterns (singular vectors) in the data. Larger singular values correspond to more significant patterns.
*   **Sparsity**: In machine learning, refers to a situation where many of the "weights" or coefficients in a model are zero, meaning those features are not used.
*   **Squared Loss**: A common way to measure the error of a machine learning model, calculated as the square of the difference between the predicted value and the actual target value.
*   **Stochastic Gradient Descent (SGD)**: A popular optimization algorithm that updates model parameters using the gradient calculated from only a small, randomly selected subset of the data at each step, making it fast for large datasets.
*   **Stochastic Neighbor Embedding (SNE)**: The precursor to t-SNE, which aimed to preserve local neighborhood structures when mapping high-dimensional data to a lower dimension.
*   **Student t-Distribution (T distribution)**: A type of probability distribution with "heavier tails" than a Gaussian distribution, meaning it's better at modeling data with more outliers or extreme values. Used in t-SNE.
*   **Supervised Machine Learning**: A type of machine learning where the model learns from labeled data (inputs with their correct outputs) to make predictions on new, unlabeled data.
*   **Symmetric Matrices**: Square matrices that are equal to their transpose (meaning the elements are mirrored across the main diagonal).
*   **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear dimensionality reduction technique especially good for visualizing high-dimensional data by placing similar data points close together in a low-dimensional map.
*   **Tensor Decomposition**: A generalization of matrix factorization to higher-order arrays (tensors), used when data has more than two dimensions.
*   **Term Cross Topics**: In NMF for text, one of the smaller matrices that represents how much each "term" (word) contributes to various "topics" (the latent factors).
*   **Term Document Matrix**: A table where rows represent words (terms) and columns represent documents, with each cell showing how often a word appears in a document.
*   **Text Classification**: The task of assigning categories or labels to text documents (e.g., categorizing emails as spam or not spam).
*   **Textual Data**: Information that is in the form of written language.
*   **Tomography**: A technique for creating 2D images of slices or sections through a 3D object, like medical scans (e.g., CT scans).
*   **Transformers (Neural Network architecture)**: A powerful and popular type of neural network architecture, especially for language tasks, known for its ability to handle long-range dependencies in data.
*   **Trigram**: A sequence of three words (e.g., "bag of words").
*   **Unitary Matrices**: Square matrices whose inverse is equal to their conjugate transpose. They preserve lengths and angles when transforming vectors.
*   **Variance**: A measure of how spread out a set of numbers are from their average value. In data, it indicates how much a feature changes.
*   **Weights**: Numbers in a machine learning model that determine the importance of each input feature in making a prediction. The model "learns" these weights during training.
*   **Wrapper Methods**: See "Feature Selection Techniques".
*   **Zero-Shot Learning**: A machine learning task where a model learns to recognize or categorize things it has never seen before during training, often by using shared representations or descriptions.

## Important Points

*   **Curse of Dimensionality**: Having too many features can make it harder for machine learning models to learn effectively, requiring much more data and computational power.
*   **Feature Selection vs. Feature Extraction**: Feature selection chooses *existing* features, while feature extraction *creates new* features from combinations.
*   **Trade-offs in Feature Selection**:
    *   **Wrapper Methods** can find the best feature subsets for a specific model and capture how features interact, but they are computationally very expensive.
    *   **Filter Methods** are fast and can be used as a pre-processing step, but they don't consider how features interact and might lead to sub-optimal models.
    *   **Embedded Methods** offer a good balance, performing selection during model training and often handling feature interactions efficiently.
*   **Lasso's Role**: The L1 Norm (Lasso) is unique among regularization techniques for its ability to force less important feature weights to exactly zero, effectively performing automatic feature selection and leading to sparse, interpretable models.
*   **t-SNE's Strength**: t-SNE is excellent for visualizing high-dimensional data because it prioritizes keeping *local* distances (how close nearby points are) accurate in the lower-dimensional space, and uses the Student's t-distribution to model "heavier tails," which helps avoid crowding distant points too closely.
*   **PCA and SVD Connection**: PCA is essentially the Singular Value Decomposition (SVD) of a *centered* data matrix (where the mean of each feature has been removed).
*   **Matrix Factorization for Incomplete Data**: It's particularly powerful for problems like recommender systems where the input data (e.g., user-item ratings) is often very incomplete (most users haven't rated most items). It learns hidden "latent factors" to fill in the gaps.
*   **Interpretable Latent Factors with NMF**: Non-Negative Matrix Factorization (NMF) can yield very human-readable "latent factors," like specific topics in text data or facial features in image data, because it constrains all values to be positive.
*   **Linear Algebra is Key**: Many dimensionality reduction techniques are rooted deeply in linear algebra concepts like vectors, matrices, eigenvalues, and eigenvectors. Re-visiting these fundamentals is highly beneficial.

## Summary
Dimensionality reduction is a crucial set of techniques in machine learning that helps manage datasets with a huge number of input features. It addresses challenges like computational slowness, poor model accuracy, and difficulty in understanding complex models. The field broadly divides into **feature selection** (choosing important existing features) and **feature extraction/transformation** (creating new, condensed features). Techniques like **Wrapper, Filter, and Embedded Methods** offer different ways to select features, each with its own pros and cons regarding computational cost and ability to capture feature interactions. For feature extraction, powerful methods include **Singular Value Decomposition (SVD)** and **Principal Component Analysis (PCA)**, which find fundamental patterns in data, and **Matrix Factorization**, especially useful for incomplete data like in recommender systems. **Non-Negative Matrix Factorization (NMF)** further enhances interpretability by ensuring positive components. Additionally, **t-SNE** provides an excellent way to visualize high-dimensional data in 2D or 3D while preserving local relationships. These techniques are vital for building more efficient, accurate, and understandable machine learning models.

## Additional Resources
*   **Deep Learning**: Explore how advanced neural networks (like Autoencoders and Transformers) learn representations and perform dimensionality reduction.
*   **Autoencoders**: Dive deeper into how these networks learn efficient data encodings for various tasks.
*   **Transformers**: Learn about the architecture that revolutionized natural language processing and how it handles high-dimensional sequential data.
*   **Tensor Decomposition**: Understand how dimensionality reduction extends to data with more than two dimensions (tensors).
*   **Optimization Algorithms**: Study Stochastic Gradient Descent (SGD) and Alternating Least Squares (ALS) in more detail, as they are fundamental to training many machine learning models.
*   **Linear Algebra for Machine Learning**: Revisit concepts like vectors, matrices, eigenvalues, eigenvectors, and matrix operations to strengthen your understanding of underlying principles.