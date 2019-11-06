from sklearn import tree

#Using DecisionTree classifier for prediction
clf = tree.DecisionTreeClassifier()

#Here the array contains three values which are height,weight and shoe size
X = [[181, 80, 91], [182, 90, 92], [183, 100, 92], [184, 200, 93], [185, 300, 94], [186, 400, 95], [187, 500, 96], [189, 600, 97],
[190, 700, 98], [191, 800, 99], [192, 900, 100], [193, 1000, 101], [194, 2000, 102]]


Y = ['male', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'transgender']

clf = clf.fit(X, Y)

#Predicting on basis of given random values for each given feature
prediction = clf.predict([[188, 200, 182]])

#Printing final prediction
print(prediction)
