
Introduction

Coronavirus vaccines and the beginning of the vaccination program caused some controversy about the importance of the vaccine. Some people were greatly influenced by conspiracy theories and refused the vaccine due to the widespread of fake news about it, while others have personal exceptions for not accepting the vaccine.

The outbreak of the Corona virus and the large number of infections required the use of machine learning techniques which help to struggle the virus rapidly. As the capacity for immediate clinical decisions and effective usage of healthcare resources is decisive, a lot of work is related to machine learning techniques that rely on machine learning emerged. This work has tried to help health centers, doctors and health workers in their decisive effort to eliminate the pandemic. One of the methods of work that emerged focused on predicting infection with the Corona virus using eight features like sex, age, known contact with an infected individual and the appearance of five initial clinical symptoms. Other work focused on deep learning such as trying to detect the infection through images taken by CT and X-Ray scans of infected patients.

The idea in this project is to predict the percentage of the acceptance of coronavirus. This will be done using dataset which was taken from “Mendeley Data” website. The dataset includes information about US citizens like age, race, education level and financial statues.

Predicting the likelihood of receiving the vaccine in the community is an effective option to struggle the virus. The higher rate of acceptance of the vaccine in the community means increasing the ability to get rid of the virus. On the other hand, decreasing this percentage requires taking measures such as raising awareness and struggling the misleading information about the vaccine that widely spread on social media.

In this project, many machine learning algorithms have been used for the prediction task like logistic regression, Random Forest, KNN and Support Vector Machine. The results showed acceptable accuracy achieved by most of these algorithms.

MATERIALS AND METHODS

Dataset Description The dataset which was used in this work was taken from “Mendeley Data” website and includes information from approximately 2978 respondents. This information includes 94 features like age, gender, sources of Covid-19 updates, financial status, race and education level all of which was collected through a survey. The survey was conducted on the assumption that socio-demographic factors can affect the decision about accepting a vaccine or not. The survey showed that 81.1% of the participants indicated their willingness to accept the vaccine.
The features in the dataset were used as input while “covid_vaccine” feature was used as output. “covid_vaccine” feature represents the answer of the question: Would you like to get COVID-19 vaccine, If available? The dataset was divided to two parts, first part was used as train data while second part was used as test data with ratio of 70% and 30% respectively.

Data Processing The database that was used in this project was not ideal as it had many flaws such as misinformation. In addition, there are many features which have object type and must be converted into numbers to allow computer interpretation.
Initially, the data was checked to get idea about the null cells for each feature. The features which have high number of null cells were dropped. Most of the dropped features are not important and don’t affect the prediction task (i.e. US_State, child and nasal_spray). For example, the state where the respondent lives, doesn't affect his decision to accept or refuse the vaccine. “Your_race” feature wasn’t dropped because the number of null cells is not high and it is considered as an important factor which can affect the result. The number of remaining features is 76.

Many of these 76 features also have null cells but they were treated in a different way; however, the null cells in these features were filled with the most frequent integer, float or object. For example, the most frequent integer in “healthcare_worker” feature was 0. This feature describes if the respondent is healthcare worker (1) or not (0).

The same procedure was applied to the features which have object components where the null cells were filled with the most frequent object. After filling the null cells, object values in the features were replaced by integers to present it which is an important step for prediction task. For example, the components of the feature “Gender_string” were replaced as follows: Male=0 and Female=1. In addition, some features like “your_race” were encoded as a one-hot numeric array. Data selection technique was used to eliminate the features that don’t have correlation with the output.

The heat map shows the correlation between the features. The features which have positive and negative correlation with the output were kept. The number of remaining features after applying this technique are 44.

Experiments

Machine learning algorithms were used in this work for the classification task. Considering that there are a large number of algorithms in machine learning, the algorithms that were used in this work were chosen on the basis that the project focuses on classification task, as the aim is to identify the relationship between many variables or features and the output (will accept the vaccine or not). In addition, supervised machine learning category was performed as the model will be trained with the given dataset.

With regard to the previous two criteria, the models which were chosen are Random Forest, Logistic Regression, Linear SVC, Support Vector Machines, Naive Bayes, Stochastic Gradient Decent, KNN and Perceptron.

Random forest is classification and regression algorithm which is based on decision trees algorithm. The model consists of large number of individual decision trees that operate as a set. The difference between random forest and decision tree is that a decision tree depends on all dataset for classification or regression, while random forest choose the features randomly to make its decision trees and take the average of the all results. Logistic Regression is a machine learning algorithm which depends on the probability. Logistic Regression is generally used when the value of the target variable is categorical.

The Linear SVC algorithm depends on a linear kernel function for classification. This algorithm works well with big datasets which include high number of samples. Support vector machine is a supervised learning method which can be used for classification, regression and outlier detection. SVM is effective in high dimensional datasets.

A Naive Bayes is a probabilistic machine learning algorithm, it is used for classification task. Naive Bayes classifier is based on the Bayes theorem. Bayes theorem can be summarized as: we can find the probability of A happening, given that B has occurred as the following equation shows with respecting that the predictors are independent.

P(A∖B)=(P(A∖B)P(A))/(P(B))

KNN is a classification model that classifies the data according to the data that is most similar. This algorithm is widely used specially in simple recommendation systems. Perceptron is a linear and binary classifier. it is used in supervised learning and its output can only be either a 0 or 1.
