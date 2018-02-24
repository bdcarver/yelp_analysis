
The three functions we created to parse our functions are:

splitText(filename)

  This function takes in the file containing all review texts and splits it into three files: development, training, and testing. The development file contains 80% of the review texts; the training, 10%; the testing, 10%. The way we split the review texts was by loading the JSON data from the input file and loading each JSON object into a list. We then took the length of the list in order to calculate the 0-80%, 80-90%, and 90-100% portions. Then, we created new files and wrote the corresponding portion of the list into the files. 

getAverageRatings(filename, type)

  This function takes in a file containing information about businesses or users and a type (either user or business) which represents the type of entities whose averages we are calculating. It starts by creating an empty dictionary and then fills the dictionary so that the keys are user or business ids and the values are the user's or business's average rating. This was rather simple to do because Yelp thankfully provides average rating in the JSON object. All we had to do was access the rating and then add it to our dictionary. Then, the populated dictionary is returned.

getVocabulary(filename, mindf)

  This function takes in a file containing review texts and a min df parameter which indicates the minimum value of a word that should be considered in the TFIDF vectorization. It starts by creating a review-index dictionary, which has an index as the key (the index represents where in TFIDF matrix the data resides) and has as a value a tuple consisting of review ID, the ID of the user who wrote the review, the ID of the business that is being reviewed, and a list of integers where each integer represents the number of "useful," "funny," and "cool" votes that the review received, respectively. We made this dictionary by iterating through the review JSON objects. As we are iterating, we also append all of the review texts into a list and all of the review ratings into a separate list. Once the dictionary has been populated, we created a TFIDF vectorizer using the min_df parameter passed in. We use the vectorizer to transform the list of reviews into a matrix where each row is the review and each column represent's the word's TFIDF value. Because we also want to include the votes as features, we then concatenated those values for each review onto the end of each row. Finally, we return a tuple whose 0th element is the matrix, whose 1st element is a matrix of labels corresponding to the reviews, and whose 2nd element is the review-index dictionary.
  
  
Challenges:

  The challenges we faced were trying to split the data into training, development, and testing. This was challenging because we couldn't find a built-in way to split apart the file, so we had to loop through and do it manually. In addition, figuring out how to add on our additional voting features onto the TFIDF vectors was challenging because we have to loop through each row to add it -- we are looking into a more efficient way to do that. 
  
  Another challenge we are facing is that when we run our code, our computers run out of memory. This is a challenge we are still seeking a solution to. We are thinking about reducing the size of the data set or editing the code to use fewer data structures.

