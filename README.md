# Bankruptcy prediction using k-nearest-neighbor

This project provides a supervised machine learning model that predicts whether or not a company will go bankrupt or not based on a collection of data of that company using k-nearest neighbors algorithm.





## Getting Started

## Prerequisites
* python 3 or above
* This is a small project that doens't need much preparation to run, so youcan just test it on google colab.



## Our data
We will be using the "Bankruptcy data from the Taiwan Economic Journal for the years 1999–2009" which can be found here : https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction.

 In my code I imported the dataset from my local machine which is provided in the project's repository as well :
```
#Locally importing the dataset
from google.colab import files
uploaded = files.upload()
```
```
df = pd.read_csv("data.csv")
```
We can check the structure of our data using : 
```
df.head()
```
The most important attribute is the "Bankrupt?" attribute which can be eitehr 1 or 0:
* if it's 1 then the company is most likely to go bankrupt.
* else if it is a 0 then it least likely to go bankrupt.



### K-nearest-neighbor


We will be using the K-nearest-neighbor learning algorithm which is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other. The KNN algorithm hinges on this assumption being true enough for the algorithm to be useful. KNN captures the idea of similarity with some mathematics calculating the distance between points on a graph.

The steps of KNN can be summarized in 6 simple steps:
Step-1: Select the number K of the neighbors
Step-2: Calculate the Euclidean distance of K number of neighbors
Step-3: Take the K nearest neighbors as per the calculated Euclidean distance.
Step-4: Among these k neighbors, count the number of the data points in each category.
Step-5: Assign the new data points to that category for which the number of the neighbor is maximum.
Step-6: Our model is ready.

Determining the best value for K is one of the limitations of KNN, but we can get around it by calculting the errot rate of each value of K in an intervall that we want and chose the values with the smallest error rate, which can be be done by running this short code 
```
error = []

for i in range(1, 40):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
```
Then we visualise a plot showing the error rate at each value using:
```
plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
```

#### Result 
With a K value of 20 (obtained by calculating the error rate of K from 1 to 40) we get an accuracy that is equal to 0.9695692025664528 which is means our model was trained correctly.





