"""
Brenna Carver & Cece Tsui
CS349 Final Project
Spring 2017
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import json
import numpy as np

def getAverageRatings(filename, type):
	''' Returns a dictionary consisting of user or business ids as keys and average rating
	as values '''
	
	ratingDict = {}
	#According to type, refer to the corresponding keys
	stars = "stars"
	obj_id = "business_id"
	if type=="user":
		stars = "average_stars"
		obj_id = "user_id"

	with open(filename, "r") as f:
		jsonobj = json.load(f)
		for line in f:
			obj = json.loads(line)
			#Associate the id of the business/user to their average rating
			ratingDict[obj[obj_id]] = obj[stars]
	return ratingDict

def splitData(filename):
	'''Takes in a filename where the file contains all data in the form of json. The function
	splits all the data in the given file where 80% of the data is training data, 10% is development
	and 10% is testing. Once split, the function writes the data into new files - "training.json", 
	"development.json", and "test.json" (Note we only looked at half of the data because
	there was too much data and it was crashing out computers).'''
	with open(filename) as f:
		data = []
		for obj in f:
			data.append(json.loads(obj)) #Load each json object

		#Calculate the number that shows 80/10 percent of 50% of the data
		halfData = int(len(data)*.5)
		eightper = int(halfData*.8)
		tenper = int(halfData*.1)

		#Split the data into train, development test
		train = []
		dev = []
		test = []
		counter = 0
		for i in range(halfData):
			if counter < eightper: #Take first 80% of the data for train
				train.append(data[i])
			elif counter < eightper+tenper: #Take the next 10% for development
				dev.append(data[i])
			else:
				test.append(data[i]) #Take the final 10% for test
			counter += 1

	#Write the split data into their correlating files
	with open("train.json", "w") as trainFile:
		json.dump(train, trainFile)
	with open("development.json", "w") as devFile:
		json.dump(dev, devFile)
	with open("test.json", "w") as testFile:
		json.dump(test, testFile)

		
def getVocabulary(filename, mindf):
	''' Returns a tuple consisting of the (1) list of text, (2) the text vector, 
	(3) the vote vector, (4) a tuple of the labels - one for logistic regression
	and the other for linear regression.'''
	reviewIndexDict = {} #{indexInMatrix: [useful, funny, cool]}
	textList = []
	labelList_log = []
	labelList_lin = []
	#counter = 0 #place of data in matrix
	with open(filename) as f:
		data = []
		for line in f: 
			data += json.loads(line) #Load each JSON object
		votes = np.zeros((len(data),3))
		#Look through each JSON obj
		for i in range(len(data)):
			obj = data[i]
			textList.append(obj.get("text", "")) #Create text List
			#reviewIndexDict[counter] = [obj["useful"], obj["funny"], obj["cool"]]
			#counter += 1 #next data
			#Add to the vote vector
			votes[i][0] = obj["useful"]
			votes[i][1] = obj["funny"]
			votes[i][2] = obj["cool"]
			#Label of the current text list; linear regression will label based on 1-5 num scale
			labelList_lin.append(int(obj["stars"]))
			#Label for logisitc regressipn - binarize
			label = 0 #0 for stars 1-3; "bad" review 
			if int(obj["stars"]) > 3: #1 for stars 4-5; "good" review
				label = 1
			labelList_log.append(label) #labels
		print("Finished text list")
		#Vectorize the text list to show tfidf values
		vectorizer = TfidfVectorizer(strip_accents="unicode", min_df = mindf, lowercase = True, use_idf = True)
		textX = vectorizer.fit(textList)
		print("Finished vectorizing")
	return (textList, textX, votes, (labelList_log, labelList_lin))

def getVotes(reviewIndexDict):
	votes = np.zeros((len(reviewIndexDict),3))
	for i in range(len(reviewIndexDict)):
		for j in range(3):
			votes[i][j] = reviewIndexDict[i][j]
	return votes
