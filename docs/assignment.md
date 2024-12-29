# Assignment - Pattern Recognition & Machine Learning

**Instructor:** Assist. Prof. Panagiotis Petrantonakis (ppetrant@ece.auth.gr)  
**Teaching Assistant:** Ph.D. Candidate Stefanos Papadopoulos (stefpapad@iti.gr)  
**2024**

---

## Part A (2 points)

You work for a company that produces video games, specifically in a department that was recently established to study stress levels in users based on the frequency and pressure patterns of button presses on the console. 
A colleague in your department, analyzing these patterns, derived an index-number $ x $ and claims that this index can be used in a classification system to determine whether a user feels stressed or not. 
Additionally, from studies conducted by your colleague, it was observed that the probability density distribution followed by this index for both classes (no stress = class $ \omega_1 $, stress = class $ \omega_2 $) is:

$$
p(x|\theta) = \frac{1}{\pi} \cdot \frac{1}{1 + (x - \theta)^2}
$$

with the parameter $\theta$ being unknown.

To determine if this index is indeed a reliable measure of user stress levels, you asked 12 other colleagues from the same department to play a specific game on the console produced by your company and calculated the index for each user. 
You then asked these colleagues whether they felt stressed during the game or not. Out of the 12, 7 reported no stress, while 5 stated that the game caused significant stress. 
You are tasked with implementing a maximum likelihood classifier. Specifically:

1. Estimate the parameters $ \hat{\theta}_1 $ and $ \hat{\theta}_2 $ using the maximum likelihood method for both classes, given that for class $ \omega_1 $, the indices are $ D_1 = [2.8, -0.4, -0.8, 2.3, -0.3, 3.6, 4.1] $, while for class $ \omega_2 $, the indices are $ D_2 = [-4.5, -3.4, -3.1, -3.0, -2.3] $. Plot $ \log p(D_1|\theta) $ and $ \log p(D_2|\theta) $ as a function of $ \theta $.
2. Use the discriminant function

   $$
   g(x) = \log P(x|\hat{\theta}_1) - \log P(x|\hat{\theta}_2) + \log P(\omega_1) - \log P(\omega_2)
   $$

   and classify the two sets of values. What do you observe about the sign of $ g(x) $ in relation to your data (plot it)? Describe the decision rule. What do you observe regarding the classification of your data using this rule?

**Hint:** You can create a class `Classifier` with:  
- A function `fit` that takes as input a dataset $ D $ and a vector of candidate $ \theta $ values and computes the maximum likelihood estimates for $ \theta $.  
- A function `predict` that takes as input a dataset $ D $ and the a priori probabilities of the classes and returns the values of $ g(x) $.

---

## Part B (2 points)

In this part, you are tasked with implementing a new classifier, estimating the unknown parameter $ \theta $ using the Bayesian estimation method.

After extensive experimentation, you determined that the values of the parameter $ \theta $ can be modeled with the prior probability density function:

$$
p(\theta) = \frac{1}{10\pi} \cdot \frac{1}{1 + (\theta/10)^2}
$$

With this model and based on the theory, you are now able to compute the posterior probability $ p(\theta|D) $ and the probability density $ p(x|D_j), j = 1, 2 $.

1. Plot the posterior probability densities $ p(\theta|D_1) $ and $ p(\theta|D_2) $. What do you observe in relation to the prior $ p(\theta) $?  
   **Hint:** Use the trapezoidal rule for integral computations.
2. Implement a function `predict` that computes the values of a discriminant function:

   $$
   h(x) = \log P(x|D_1) - \log P(x|D_2) + \log P(\omega_1) - \log P(\omega_2)
   $$

   What do you observe now about the values of $ h(x) $ in relation to your datasets (plot it)? How do you evaluate the Bayesian parameter estimation method compared to the maximum likelihood method for this example? Why do you think the difference between the two approaches exists in this specific example?

   **Hint:** You can adopt a similar implementation to Part A.

---

## Part C (2 points)

### Section 1

You work as a research assistant at the Floriculture Laboratory of the Department of Agriculture, Aristotle University of Thessaloniki, specializing in data analysis. 
A research department in the lab focuses on the automated recognition of different species of a specific plant, Iris. 
The three specific species Iris setosa, Iris versicolor, and Iris virginica exhibit differences in the length and width of their sepals and petals. 
Using the sklearn library, you can download a dataset of 150 measurements (50 for each species) of the sepal and petal length and width. 
By isolating only the first two features of the dataset, use the built-in `DecisionTreeClassifier` algorithm from sklearn and classify 50% of the dataset randomly after training the algorithm with the remaining 50%.

1. What percentage of correct classifications do you achieve? What tree depth gives you the best percentage?  
2. Plot the decision boundaries of the classifier for the best result.  
   **Hint:** Use the `contourf` function.

### Section 2

Now create a Random Forest classifier with 100 trees using the Bootstrap technique. Specifically, the 50% of the samples you used for training in the previous section (let’s call it dataset \( A \)) should now be used to create 100 new training sets, one for each tree, where each time you use \( \gamma = 50\% \) of dataset \( A \). Use the dataset classified in the previous part for evaluating the algorithm. All trees should have the same maximum depth.

1. What percentage of correct classifications do you achieve? What tree depth gives you the best percentage?  
2. Plot the decision boundaries of the classifier for the best result. What do you observe compared to the simple classifier in the previous section?  
3. How do you think the percentage $ \gamma $ affects the algorithm’s performance? Provide examples.

---

## Part D (4 points)

In this part, you will work with the file `datasetTV.csv`, which you will use as a training set. The training data consists of 8743 samples and 224 features per sample, accompanied by a label (1, ..., 5) in the last column. Develop a classification algorithm using any method of your choice. You may preprocess your features as you see fit.

Next, use the data from the file `datasetTest.csv` (6955 samples) as a test set (labels are not provided in this file). Apply your final trained model to this dataset and generate a vector named `labelsX` (see the explanation for \( X \) below), which you will submit in numpy format.

Additional bonus points will be awarded to teams with the best results (minimum classification error) for this part.

---

## Instructions

- The implementation of the assignment should be done in Python. Choose a notebook (e.g., Jupyter, Colab) and write both your code and comments.  
- For submission, upload a SINGLE file named: `TeamX.zip` containing all the necessary files (if you are a team of two, ONLY one member should submit the assignment). The `.zip` file must include:
  1. The file `TeamX-AC.ipynb` with the code for Parts A-C.
  2. The file `TeamX-D.ipynb` with the code for Part D.
  3. The file `labelsX.npy`, which will contain the label vector derived from Part D.  
     - **Important:** Make sure that the saved `labelsX.npy` file can be loaded using `numpy.load()` and that it has dimension $ N $ (where $ N $ is the number of samples in the test set).  
  4. A file `TeamX.pdf` in slide format, where all parts of the assignment (Parts A to D) are described.  
     - The presentation file must be strictly limited to **50 slides**:
       - 10 slides for each of Parts A-C
       - 20 slides for Part D  
     - In each of the `.ipynb` and `.pdf` files, include your personal details (name, student ID).

---

## Notebook Structure and Comments

- Each question in Parts A-C should be answered in a **separate cell** with the corresponding code.
- The code in each cell should be accompanied by **brief comments** (**important!**).
- For Part D, you can structure your code as you wish, but relevant comments are also required.

---

## Evaluation Criteria

The evaluation will be based on the following criteria:
1. The quality of the code and comments.
2. The quality of the corresponding presentation for each part.
3. The correctness of the approaches and results.

The best assignments from Part D will present their classifier in person. (**In-person presentation is mandatory for bonus grading.**)

---

## Final Submission Date

**Wednesday, January 8, 2025, 23:59**
