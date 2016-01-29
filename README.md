A Naive Bayes Model
===

This is an implementation of a Naive-Bayes model in Python, a supervised learning model used to probabilistically classify datasets. For this project, I used  "Applied Predictive Models" by Max Kuhn, which is an excellent introduction to Naive Bayes estimation. I took on this project as a personal learning experience on a machine-learning model which I love to use, but haven't had to chance to understand fully. With this implementation, I think I've obtained a much better understand of how this model works, which should help me with pre-built Naive Bayes models, such as that of Scikit-Learn.

Currently, this SVM only supports gaussian denisty estimation, but I eventually plan to update the implementation to support Bernoulli, kernel and multinomial density estimation. It may also be valuable to update the model to support nominal attributes and combine log probabilities to deal with possible floating point underflow.

Note: Data for this particular implementation comes from the National Institute of Diabetes and Digestive and Kidney Diseases. More information can be found in the description file (pima-indians-diabetes.names.txt).