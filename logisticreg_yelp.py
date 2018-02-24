"""
Brenna Carver & Cece Tsui
CS349 Final Project
Spring 2017
"""

import numpy as np 
from sklearn import linear_model
from numpy.linalg import norm
import math
from parsing_functions import getVocabulary


def show_significant_features(w, featurelist):
	"""Prints the top twenty words predictive of positive or negative reviews"""
	wsorted = np.argsort(w)
	print 'Features predicting negative review:', ', '.join(map(lambda i: featurelist[i], wsorted[:20]))
	print 'Features predicting positive review:', ', '.join(map(lambda i: featurelist[i], wsorted[-20:][::-1]))
    
def get_accuracy(testy, predictions):
    """return proportion of correct predictions"""
    # convert probabilities to 0 or 1 predictions first
    predictions = np.array(predictions)
    predictions = (predictions>0.5).astype(int)  # 1 if over 3, 0 if not
    return 1-norm(predictions-testy, 0)/float(len(testy))
    
def predict_confidence(w, b, testX):
    """return a vector of probabilities of the class y=1 for each data point x"""
    probs = []
    for x in testX:
        probs.append(sigmoid(x, w, b))
    return np.asarray(probs)

def log_reg_grid_param(textX, voteX, devText, devVote, labels, devLabels, maxiter, eta, alpha):
    '''Grid Searches the Hyperparameters for Logistic Regression Model for both
    the text and vote vectors to find the best hyperparameters for the models.
    Returns the best parameters and their associated development accuracy and model
    for both the text and vote.'''
    text_log_accuracy = []
    vote_log_accuracy = []
    #Initialize best parameters; tuple of (1) dev accuracy, (2) hyperparameters associated
    #   with accuracy, (3) model
    text_best_param, vote_best_param = (0,0,0), (0,0,0)
    model = 0

    for m in maxiter:
        for a in alpha:
            for e in eta:
                #-----Text Vector-------
                #Make prediction
                textClass_log, textPred_log = logreg_model(textX, labels[0], devText, m, a, e)
                textAccuracy = get_accuracy(devLabels[0], textPred_log)
                if text_best_param[0] < textAccuracy:
                    #Have new best hyperparameter if current accuracy is bigger
                    text_best_param = (textAccuracy, (m,a,e), textClass_log)
                    model = textClass_log
                text_log_accuracy.append((textAccuracy, (m,a,e), textClass_log))
                print "Text", "Logistic", m, a, e, textAccuracy

                #-----Vote Vector-------
                #Make prediction
                voteClass_log, votePred_log = logreg_model(voteX, labels[0], devVote, m, a, e)
                voteAccuracy = get_accuracy(devLabels[0], votePred_log)
                if vote_best_param[0] < voteAccuracy:
                    #Have new best hyperparameter if current accuracy is bigger
                    vote_best_param = (voteAccuracy, (m,a,e), voteClass_log)
                vote_log_accuracy.append((voteAccuracy, (m,a,e), voteClass_log))
                print "Vote", "Logistic", m, a, e, voteAccuracy

    print "Text Best", text_best_param
    print "Vote Best", vote_best_param

    #Return best hyperparameters and associated accuracies for development and model
    return (text_best_param, vote_best_param, model)

def testClassifiers(trainFile, devFile, testFile, mindf, maxiter, eta, alpha):
    '''Given the training file, development file, and test file, create a logistic
    regression and linear regression model from the train file. 

    For Logsitic Regression -With the given development file, find
    the best hyperparameters and their associated accuracies for the logistic regression
    model. Using the hyperparameters, test the model on the test file and find accuracy.

    For Linear Regression - With the given development file, find the accuracy of the
    data. Then, test the model on the test file and find accuracy.'''

    #Vectorizing the data from train, dev, and test files
    print "Vectorizing Train"
    trainList, vect, voteX, labels = getVocabulary(trainFile, mindf)
    textX = vect.transform(trainList)
    vocab = vect.vocabulary_

    print "Vectorizing Dev"
    devList, devVect, devVote, devLabels = getVocabulary(devFile, mindf)
    # use same vectorizer as train
    devText = vect.transform(devList)

    print "Vectorizing Test"
    testList, testVect, testVote, testLabels = getVocabulary(testFile, mindf)
    #use same vectorizer as train
    testText = vect.transform(testList)

    #Logistic Regression
    print "-----Logistic Regression-----"
    #Find best hyperparameters with development file
    text_best_param, vote_best_param, model = log_reg_grid_param(textX, voteX, devText, devVote, labels, devLabels, maxiter, eta, alpha)
    
    #Show development accuracy of best hyperparameters for text vector
    print "Best Parameters (Text):", text_best_param[1]
    print "Development Accuracy with Best Hyperparameters (Text):", text_best_param[0]
    #Test the model with best hyperparameters on the test
    text_best_logModel = text_best_param[2]
    textTest_pred = text_best_logModel.predict(testText)
    text_test_accuracy = get_accuracy(testLabels[0], textTest_pred)
    print "Test Accuracy with Best Hyperparameters (Text):", text_test_accuracy

    #Show development accuracy of best hyperparameters for vote vector
    print "Best Parameters (Vote):", vote_best_param[1]
    print "Development Accuracy with Best Hyperparameters (Vote):", vote_best_param[0]
    #Test the model with the best hyperparameters on the test
    vote_best_logModel = vote_best_param[2]
    voteTest_pred = vote_best_logModel.predict(testVote)
    vote_test_accuracy = get_accuracy(testLabels[0], voteTest_pred)
    print "Test Accuracy with Best Hyperparameters (Vote):", vote_test_accuracy

    #Linear Regression
    print "-----Linear Regression-----"
    #Show development accuracy of model for text
    textClass_lin, textPred_lin, r2 = linreg_model(textX, labels[1], devText, devLabels[1])
    textAccuracy = get_accuracy_lin(np.rint(textPred_lin), devLabels[1])
    print "Development Accuracy (Text):", textAccuracy

    #Show accuracy of model on test data for text
    textTest_pred = textClass_lin.predict(testText)
    text_test_accuracy = get_accuracy_lin(np.rint(textTest_pred), testLabels[1])
    print "Test Accuracy (Text):", text_test_accuracy

    #Show development accuracy of model for vote
    voteClass_lin, votePred_lin, r2 = linreg_model(voteX, labels[1], devVote, devLabels[1])
    voteAccuracy = get_accuracy_lin(np.rint(votePred_lin), devLabels[1])
    print "Development Accuracy (Vote):", voteAccuracy

    #Show accuracy of model on test data for vote
    voteTest_pred= voteClass_lin.predict(testVote)
    vote_test_accuracy = get_accuracy_lin(np.rint(voteTest_pred), testLabels[1])
    print "Test Accuracy (Vote):", vote_test_accuracy
    
    featurelist = map(lambda x:x[0], sorted(vocab.items(), key=lambda x:x[1]))
    show_significant_features(model.coef_[0], featurelist)
    

def get_accuracy_lin(testy, predictions):
    '''Given the real predictions (testy) and predicted values (predictions),
    return the accuracy of the linear regression model.'''
    accurate = 0
    total = 0
    for i in range(len(predictions)):
        if int(testy[i]) == int(predictions[i]):
            accurate += 1
        total += 1
    print len(predictions)
    return float(accurate)/float(total)

def logreg_model(trainX, trainy, testX, maxiter, alpha, eta):
    '''Create the logistic regression model with the given data and hyperparameters.
    Test and predict on the development data (testX).'''
    model = linear_model.SGDClassifier(loss='hinge', penalty='l2', alpha=alpha, l1_ratio=0.15, fit_intercept=True, n_iter=maxiter, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', eta0=eta, power_t=0.5, class_weight=None, warm_start=False, average=False)
    model.fit(trainX, trainy)
    predictions = model.predict(testX)
    return (model, predictions)

def linreg_model(trainX, trainy, testX, testy):
    '''Create the linear regression model with the given data and hyperparameters.
    Test and predict on the development data (testX)'''
    model = linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
    model.fit(trainX, trainy)
    predictions = model.predict(testX)
    score = model.score(testX, testy)
    return (model, predictions, score)

testClassifiers("data/train.json", "data/development.json", "data/test.json", 50, [10, 50, 100], [1e-1, 1e-2, 1e-3, 1e-4], [1e-1, 1e-2, 1e-3, 1e-4])


    


