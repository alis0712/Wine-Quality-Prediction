# Wine Quality Prediction from Scratch

## Introduction 
The goal of this project is to determine the quality of the wine data. We were asked to create a decision tree based on Hunts Algorithm for three different measures of node impurity which are Gini Index, Entropy, and Misclassification Error. The input variables were fixed acidity, volatile acidity, citric acid, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, and percentage of alcohol content. All these input variables determined the quality of wine which is scored between 0 and 10.After implementing hunt’s decision tree based on entropy we obtained a predicted quality score of 6.0 and had an accuracy of 65%, similarly implementing the hunt’s decision tree on Gini Index gave us a predicted quality score of 6.0 and an accuracy of 75%, and lastly implementing the decision tree for misclassification error gave us a predicted score of 6.0 and accuracy of 65% which means that based on the three node impurity measures the quality of the wine is 6.0 based on the wine data provided, which would be considered an “normal” quality wine. 

## Implementation
(i)	Decision Tree Analysis 
When implementing the decision tree, it was decided that id column would be dropped since it didn’t affect the quality of wine. Initial dataset of 5 data points was taken to see what the tree would like after dropping off the id column, the handwritten decision tree is shown below: 

![image](https://user-images.githubusercontent.com/62857780/209243679-b917f9e0-7812-4131-b4a9-eec5038a5b3e.png)


As we can see from the flow chart, the depth of the tree was around 5 and these values were used to determine the quality of the wine. When determining the split point, the node impurity Gini Index was used, and the classification was based on high and low values of alcohol, volatility, sulphate, free sulphate, residual, citric acid, and density. To simplify the hand calculations, it was decided to drop fixed acidity, total sulfur dioxide, chlorides, pH in the hand calculations to simplify the model, however, it was decided to add these back when implementing the decision tree in our actual model. These input parameters determined if the quality of wine was considered good, bad, or average. 

(ii)	Decision Tree Implementation in Python
In the programming model the decision tree was implemented the following way:
![image](https://user-images.githubusercontent.com/62857780/209243757-e8586ddf-7306-4ce4-ae60-6301144fc3cb.png)

After implementation of the programming model, the dataset was loaded. The data field id was dropped as it wasn’t an input variable and didn’t contribute to the wine’s quality. The loaded dataset is given below: 

![image](https://user-images.githubusercontent.com/62857780/209243825-9204e3b9-3a4d-4a1e-bc41-3f5fce839839.png)

After the data was loaded, the function trainTestSplit was used which split the data between a training set and a test set. The test set contained a small fraction of observations (data validations), and the rest of the data set was used for training. I then checked if the data was pure which measured the extent to which a group of records share the same class, and then moved on to classify the data points as categorical or continuous training datasets. To do this, I created a function called determineTypeofFeature which determines whether there are categorical datasets or continuous datasets. In our case, the parameters “volatile_acidity”, “total_sulfur_dioxide”, “chlorides”, and “density” had the highest number of unique values and were therefore considered categorical. A table below summarizes which data sets were considered categorical and which data sets were considered continuous: 

![image](https://user-images.githubusercontent.com/62857780/209243888-a382a37d-e0b8-4904-8f65-0d68939db355.png)

Once we split the data between categorical and continuous, I then created a function called obtainPotentialSplits, which splitted the unique column values exactly at the middle of two columns. A two-way split was used when determining the quality of the wine. A figure below shoes how the data points in each column were split:

![image](https://user-images.githubusercontent.com/62857780/209243925-82b49259-225b-49af-97c4-cf5ed681df75.png)

After obtaining the potential splits, I then implemented a function called splitData which carried out the splitting process. To make the splitting process simpler I decided to split the data horizontally into two parts which is the data points below and the data points above (I could’ve done it vertically as well but decided to split the data horizontally). Categorical datasets had their column values equal to the split value.   A figure below shows how the data was split: 

![image](https://user-images.githubusercontent.com/62857780/209243949-0ee34ecb-d20f-45a2-b855-024f736cd69c.png)

After the data was split horizontally, I then calculated the entropy, the entropy was divided into two functions which were the lowest overall entropy and the entropy function itself. The calculateEntropy function calculated the probabilities of each class and the overallEntropy function calculated the total probabilities below and above the split data points. Once the overallEntropy function was completed I then created a function called determineBestSplit which determined which best split point values below and above had the lowest overall entropy values. Once the lowest overall entropy values have been calculated, I then began the process of creating the decision tree. We were asked to create the decision tree using Hunt’s algorithm. Hunt’s algorithm was implemented the following way: 

huntsAlgorithm(Node t, trainingdatabase D, split selection method S)
(1) Apply S to D to find the splitting criterion¶
(2) if (Node t is not a pure node)
(3) Create Children nodes of t
(4) Partition D into children partitions
(5) Recurse on each partition
(6) endif
The implementation of the algorithm in Python is given below:

![image](https://user-images.githubusercontent.com/62857780/209244102-8e79a843-a2b3-4620-a6d0-31c938c327e5.png)

To begin the algorithm process I created a dictionary which determines the quality of the wine based on the string values “good” and “bad” and the whole tree is based on this sub tree. My hunts algorithm takes in raw data and the count function since hunts algorithm is a recursive function, once the data is prepared, I then begin the process of classifying the data in terms of its purity. The purity is base case and determines when to stop the algorithm. If the base case is not pure then we can begin the recursion process and call the functions obtainPotentialSplits, determineBestSplit, and splitData. We also need to make sure that there are data points below and above have no empty data sets, if there are then we should stop recursion process. We also need to accommodate the fact that the data can either be categorical or continuous. Once we have identified whether the data is categorical or continuous, we can then instantiate the process of subtree which determines whether the quality of wine is either “good” and “bad” and use the huntsAlgorithm function for the data points below and above the tree. The whole process is then repeated for the node impurities Gini Index and Misclassification. The three decision trees with a max depth of 5 for Entropy, Gini Index and Misclassifiication are plotted below. 

## Decision Trees
(i)	Entropy Based: 

{'alcohol <= 10.525': [{'sulphates <= 0.625': [{'citrix_acid <= 0.055': [{'sulphates <= 0.525': [{'citrix_acid <= 0.045': [5.0,
                                                                                                                           3.0]},
                                                                                                 {'volatile_acidity = 0.58': [7.0,
                                                                                                                              5.0]}]},
                                                                         5.0]},
                                               {'alcohol <= 9.850000000000001': [{'fixed_acidity <= 12.05': [5.0,
                                                                                                             {'sulphates <= 0.815': [6.0,
                                                                                                                                     7.0]}]},
                                                                                 {'density = 0.996': [{'alcohol <= 9.975': [3.0,
                                                                                                                            8.0]},
                                                                                                      6.0]}]}]},
                       {'citrix_acid <= 0.295': [{'sulphates <= 0.585': [{'free_sulfur_dioxide <= 31.5': [{'free_sulfur_dioxide <= 8.5': [5.0,
                                                                                                                                          6.0]},
                                                                                                          {'alcohol <= 11.45': [5.0,
                                                                                                                                7.0]}]},
                                                                         {'fixed_acidity <= 5.699999999999999': [{'citrix_acid <= 0.03': [6.0,
                                                                                                                                          7.0]},
                                                                                                                 {'volatile_acidity = 0.64': [5.0,
                                                                                                                                              6.0]}]}]},
                                                 {'alcohol <= 11.55': [{'total_sulfur_dioxide = 16.0': [8.0,
                                                                                                        {'citrix_acid <= 0.315': [7.0,
                                                                                                                                  6.0]}]},
                                                                       {'residual_sugar <= 4.65': [{'sulphates <= 0.745': [6.0,
                                                                                                                           7.0]},
                                                                                                   {'alcohol <= 12.8': [7.0,
                               
                                                                                        5.0]}]}]}]}]}


(ii)	Gini Based

{'alcohol <= 10.525': [{'sulphates <= 0.585': [{'alcohol <= 9.75': [{'pH <= 3.545': [{'density = 0.99888': [6.0,
                                                                                                            5.0]},
                                                                                     6.0]},
                                                                    {'sulphates <= 0.545': [{'density = 0.99538': [6.0,
                                                                                                                   5.0]},
                                                                                            {'density = 0.99568': [7.0,
                                                                                                                   6.0]}]}]},
                                               {'fixed_acidity <= 10.850000000000001': [{'alcohol <= 9.850000000000001': [5.0,
                                                                                                                          {'residual_sugar <= 3.3': [6.0,
                                                                                                                                                     5.0]}]},
                                                                                        {'total_sulfur_dioxide = 42.0': [{'sulphates <= 0.645': [5.0,
                                                                                                                                                 7.0]},
                                                                                                                         {'free_sulfur_dioxide <= 29.0': [6.0,
                                                                                                                                                          5.0]}]}]}]},
                       {'citrix_acid <= 0.295': [{'free_sulfur_dioxide <= 8.5': [{'pH <= 3.34': [{'sulphates <= 0.685': [6.0,
                                                                                                                         5.0]},
                                                                                                 {'sulphates <= 0.62': [5.0,
                                                                                                                        6.0]}]},
                                                                                 {'volatile_acidity = 0.64': [{'total_sulfur_dioxide = 33.0': [6.0,
                                                                                                                                               5.0]},
                                                                                                              6.0]}]},
                                                 {'alcohol <= 11.45': [{'pH <= 3.085': [{'citrix_acid <= 0.51': [5.0,
                                                                                                                 7.0]},
                                                                                        6.0]},
                                                                       {'sulphates <= 0.745': [{'residual_sugar <= 4.65': [6.0,
                                                                                                                           7.0]},
                                                                                               {'fixed_acidity <= 10.7': [7.0,
                                         
                                                                                 6.0]}]}]}]}]}

(iii)	Misclassification Error

{'alcohol <= 10.25': [{'fixed_acidity <= 10.850000000000001': [{'volatile_acidity = 0.51': [{'alcohol <= 9.35': [5.0,
                                                                                                                 6.0]},
                                                                                            {'density = 0.9984': [{'free_sulfur_dioxide <= 17.5': [6.0,
                                                                                                                                                   5.0]},
                                                                                                                  {'density = 0.9972': [6.0,
                                                                                                                                        5.0]}]}]},
                                                               {'sulphates <= 0.565': [5.0,
                                                                                       {'free_sulfur_dioxide <= 19.5': [{'density = 1.0022': [7.0,
                                                                                                                                              6.0]},
                                                                                                                        {'pH <= 3.035': [7.0,
                                                                                                                                         4.0]}]}]}]},
                      {'residual_sugar <= 5.55': [{'volatile_acidity = 0.28': [{'pH <= 3.3099999999999996': [7.0,
                                                                                                             {'sulphates <= 0.8600000000000001': [5.0,
                                                                                                                                                  6.0]}]},
                                                                               {'total_sulfur_dioxide = 23.0': [{'free_sulfur_dioxide <= 12.0': [5.0,
                                                                                                                                                 6.0]},
                                                                                                                {'citrix_acid <= 0.735': [6.0,
                                                                                                                                          7.0]}]}]},
                                                  {'fixed_acidity <= 7.6': [{'alcohol <= 11.0': [{'alcohol <= 10.65': [5.0,
                                                                                                                       7.0]},
                                                                                                 {'alcohol <= 11.25': [4.0,
                                                                                                                       6.0]}]},
                                                                            {'density = 0.9988': [6.0,
                                                                                                  {'density = 0.9976': [5.0,
                             
                                                                                           7.0]}]}]}]}]}

## Classification and Accuracy
Python Implementation: 
After the tree was created, we then needed to classify our tree in terms of quality of the wine  and calculate an accuracy of the tree. To do this we first determined if the quality of the wine is good or bad and tested it on the sample random dataset given below:
![image](https://user-images.githubusercontent.com/62857780/209244332-9c4754ba-2ec9-44fb-b11f-e7bef9e4de5c.png)

I then created a function called classify which outputs the quality of the wine, to do this I first determined the “alcohol” content in the wine, once “alcohol” content was determined, the dataset was were splitted into other dataset points which looked at parameters such as “fixed acidity’, “sulphates”, and so forth eventually going to the bottom of the tree and finding the quality of the wine which in this case was 6.0.
 Once we have classified the quality of the wine, we can then determine how accurate our tree is which I did by creating a function called accuracy which takes in data and our tree. The data parameter contains the classification and classification that were correct and compares the classification parameter with the quality of the wine. I then take the mean of the correct classification to obtain the accuracy of the decision tree. This gave us a decision tree based on entropy with a predicted quality score of 6.0 and had an accuracy of 65%, similarly implementing the hunt’s decision tree on Gini Index gave us a predicted quality score of 6.0 and an accuracy of 75%, and lastly implementing the decision tree for misclassification error gave us a predicted score of 6.0 and accuracy of 65% which means that based on the three node impurity measures the quality of the wine is 6.0 based on the wine data provided, which would be considered an “normal” quality wine. The table below shows which index values were misclassified based on Entropy, Gini, and Misclassification: 

Entropy Misclassified Data Table
![image](https://user-images.githubusercontent.com/62857780/209244370-9101d1e2-f0da-4c66-9cd4-eda1b6d24042.png)

Gini Misclassified Data Table
![image](https://user-images.githubusercontent.com/62857780/209244444-c9d39371-7e0f-40c3-b4e3-32fe1d7d6392.png)

Misclassification Error Misclassified Data Table
![image](https://user-images.githubusercontent.com/62857780/209244547-524e4c0c-a2f4-4e6b-90c7-e91558bc9360.png)

## Post Pruning
The results obtained through the decision tree were pre-pruned therefore the accuracy of the decision tree was not very high and therefore I decided to prune the decision tree which gave it a higher accuracy of the model. To do this, the checkPurity function was renamed to createLeaf as we are lower transitioning to lower branches of the decision tree. The classify function was renamed to predict as we are predicting the quality of the wine as we are transitioning to the lower branches of the tree. The calculateoverallEntropy function was renamed to calculateoverallMetric since we are adding the metric regression which compares the difference between actual values and the predicted values. 

Several new functions were also added which were calculateMSE which calculates the mean squared accuracy along with the functions filterData, determineLeaf, determineErrors, pruningResult, and postPrune. determineErrors function determines if there are any errors in the validation data or test data as we go down the decision tree. The determineBestSplitMetric included a regression task other than the entropy task. The filterData is like splitData as described in the decision tree implementation and basically is a function that stores and filters all the good and bad data points. The pruningResult function considers training dataset and validation dataset. It also removes all the bad(misclassified) data points obtained in the determineErrors function. The determineLeaf function takes the mean of the most common quality found in the wine dataset. We then pass all these newly created functions to the postPrune function which filters out all the good datapoints along with the bad data points and prints a pruned tree as shown below:

![image](https://user-images.githubusercontent.com/62857780/209244635-e0757f88-1e44-4864-b85f-097df8ad7c8c.png)

If we compare the accuracy obtained from pre-pruning the decision tree based on entropy as the node impurity vs the accuracy obtained after pruning the tree(based on entropy), we notice a slightly higher accuracy value of 75% vs 65% which shows that after post pruning, we get better accuracy since the wine data decision tree has a very large depth and shows the overfitting of the model. A graph below illustrates the difference in accuracies as we go deeper into the tree: 

![image](https://user-images.githubusercontent.com/62857780/209244669-63d0afb2-d417-475f-8f34-771baf19449c.png)

## Conclusion
The goal of this project was to analyze the wine dataset and predict the quality of the wine using the measures of node impurities which are Entropy, Misclassification and Gini. The quality of wine was scored between 0 and 10 and after implementing the decision tree based on these three nodes impurities, we obtained a quality value of 6.0 for Entropy with accuracy of 65%. Similarly for Gini we obtained a quality value of 6.0 with accuracy of 75% and for misclassification node impurity we obtained a quality value of 6.0 with accuracy of 65% which means the quality of the wine based on the dataset provided was a 6.0. 

To further improve the accuracy of the model I then went ahead and did post pruning of the decision tree which improved the accuracy of the entropy-based decision tree to 75%. I could have improved the accuracy of the model by implementing a cross validation method since the data points would have been divided into n-folds and that would have performed training on 50% of the dataset and the rest of the 50% would have been used on the testing set. Another way the model could have been improved is by implementing a random forest tree algorithm since hunt’s algorithm only combines some decisions (for example fixed acidity vs volatile acidity) whereas the random forest algorithm combines several decision trees thereby improving the accuracy of the model, even though the algorithm is slower it still does a rigorous job on the training dataset thereby improving the accuracy of the model, and also averts the issue of overfitting by utilizing multiple decision trees. 

## References

https://www.youtube.com/watch?v=sgQAhG5Q7iY

https://towardsdatascience.com/decision-tree-algorithm-in-python-from-scratch-8c43f0e40173

https://www.w3schools.com/python/default.asp

https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial/notebook

https://allysonf.medium.com/predict-red-wine-quality-with-svc-decision-tree-and-random-forest-24f83b5f3408

https://scikit-learn.org/stable/modules/cross_validation.html










