CS349 Milestone 3

Brenna Carver and Cece Tsui

A. Project Changes

After looking at comments and speaking with Sravana, we have made some changes to our initial proposal. 

(1) We have decided that instead of creating one vector of the text in a tfidf model and a review's "useful", "funny", and "cool" votes, we would separate this into two different vectors in which we would use to predict a reveiw's label. One vector would simply be the text. The other vector would be composed of a review's "useful", "funny", and "cool" votes. Thus, we would run our algorithm on both the vectors and see which attributes (the review's text or the reveiw's votes) are better in determining a review's label.

(2) We plan on binarizing the review labels such that instead of predicting {1,2,3,4,5} for reviews, we will predict 0 or 1. 0 will mean a "bad" review, or a star prediction between 1-3 and 1 will mean a "good" review, or a star prediction from 4-5. Because we are binarizing the labels, we have thereby settled on using the Logistic Regression Model. The reason why we are binarizing the review lables is because the "useful","funny", and "cool" votes matrix do not have as many features, and thereby would be much more difficult to predict a {1,2,3,4,5} label.

(3) Our project's focus thereby shifts away from just predicting a review, but rather, looking at which attributes of a review can give a better sense of whether a review is "good" or "bad" (in a sense, "positive" or "negative"). 

B. Updates in Milestone 3 Goal

Due to the changes in our project, we had to focus on making the necessary changes in our parsing_functions.py to separate the two attribtues -- text and the review's votes. This helped with lowering the run-time, as we no longer needed to concatenate the review's votes to the end of each vector of text, meaning we also could leave the text vector returned by the TfidfVectorizer sparse. 

Our goal by Milestone 3, however, was to finalize the alogorithm we are using and get it ready to be tested. We were able to finalize our algorithm, the function that allows us to create the classifier given the testing data, and the testing portion . However, there are quite a few bugs we have to work through, especially when working with a sparse array, in order to get our algorithm working to give us a classifier. However, we have made significant progress, and feel that we've almost reached our goal! The only set back was the small changes we have made to our project.

C. Pending Work

We must still make changes to our algorithm to work within the bounds of a sparse matrix. In addition, we need to hyperparameter tune (the min-df for the TfidfVectorizer for the text matrix) and find the accuracies of running data on the two separate classifiers -- a classifier built on the text matrix and another built on the vote matrix (thus, we must also write a means to test the classifers on our test data). Once we are able to compare the accuracies, we will be able to make a clear distinction as to which classifier is better and what features of a review are better indications of a review's label. 

If we are to have more time, we plan on including more information into our votes matrix. However, we are simply looking to get our algorithm working. 