import numpy
import pandas
import random

class MoviesRecommender(object):
    
    
    def __init__(self):
        self.Users_Matrix = None
        self.Items_Matrix = None
        self.User_To_Index = dict()
        self.Movie_To_Index = dict()
    
    
    def Train(self, observed_matrix, user_to_index, movie_to_index):
        
        self.User_To_Index = user_to_index
        self.Movie_To_Index = movie_to_index
        
        rows = len(observed_matrix)
        cols = len(observed_matrix[0])
        
        features_number = 3

        self.Users_Matrix = numpy.random.rand(rows, features_number)
        self.Items_Matrix = numpy.random.rand(cols, features_number)

        self.Gradient_Descent_Matrix_Factorization(observed_matrix, features_number)
        return
        
     
    def Predictions(self, users_movies_vector):
        predictions = list()
        
        for i, row in users_movies_vector.iterrows():
            if row.userID in self.User_To_Index and row.movieID in self.Movie_To_Index:
                user_index = self.User_To_Index[row.userID]
                movie_index = self.Movie_To_Index[row.movieID]
                user_row = self.Users_Matrix[user_index]
                movie_row = self.Items_Matrix[:,movie_index]
                prediction = self.Predict(user_row, movie_row.T)
                predictions.append(str(row.testID) + ", " + str(prediction) + "\n")
            
        file = open("results.csv", "w")
        for prediction in predictions:
            file.write(prediction)
        
        
    def Predict(self, user_row, item_col):
        prediction = numpy.dot(user_row,item_col)
        return prediction
        
        
    def Error_Function(self, target, prediction):
        return pow(target - prediction, 2)


    def Error_Partial_Derivative_Users(self, target, prediction, item):
        return 2 * (target - prediction) * item


    def Error_Partial_Derivative_Items(self, target, prediction, user):
        return 2 * (target - prediction) * user


    def Is_Observed(self, entry):
        return entry > 0


    def Gradient_Descent_Matrix_Factorization(self, Observed_Matrix, Features_Number, steps=5000, learning_rate=0.002, beta=0.02):
        
        counter = 0
        
        Items_Matrix = self.Items_Matrix.T
        Users_Matrix = self.Users_Matrix

        for step in range(steps):
            total_error = 0.0

            for row in range(len(Observed_Matrix)):
                for col in range(len(Observed_Matrix[row])):
                    if self.Is_Observed(Observed_Matrix[row][col]) == True:
                        
                        target = Observed_Matrix[row][col]

                        user_row = Users_Matrix[row,:]
                        item_col = Items_Matrix[:,col]
                        prediction = self.Predict(user_row,item_col)

                        for feat in range(Features_Number):
                            Users_Matrix[row][feat] = Users_Matrix[row][feat] + learning_rate * self.Error_Partial_Derivative_Users(target, prediction, Items_Matrix[feat][col])
                            Items_Matrix[feat][col] = Items_Matrix[feat][col] + learning_rate * self.Error_Partial_Derivative_Items(target, prediction, Users_Matrix[row][feat])

                        user_row = Users_Matrix[row,:]
                        item_col = Items_Matrix[:,col]
                        prediction = self.Predict(user_row,item_col)

                        total_error = total_error + self.Error_Function(target, prediction)

            if total_error < 0.01:
                break

        self.Users_Matrix = self.Users_Matrix
        self.Items_Matrix = self.Items_Matrix.T
        return
        
        
        
################################# MAIN #############################################

train_set = pandas.read_csv('user_ratedmovies_train.csv',',')
train_set = train_set.reindex(numpy.random.permutation(train_set.ID))[0:100]
train_set_list = train_set.iloc()

user_to_index = dict()
movie_to_index = dict()

user_counter = 0
movie_counter = 0
for i, row in train_set.iterrows():
    if (row.userID in user_to_index) == False:
        user_to_index[row.userID] = user_counter
        user_counter = user_counter + 1
    if (row.movieID in movie_to_index) == False:
        movie_to_index[row.movieID] = movie_counter
        movie_counter = movie_counter + 1
        
rows = len(user_to_index)
cols = len(movie_to_index)

observed_matrix = numpy.zeros((rows, cols))

for userID, userIndex in user_to_index.items():
    user_ratings = train_set[train_set.userID == userID]
    for movieID, movieIndex in movie_to_index.items():
        movie_rating = user_ratings[user_ratings.movieID == movieID].rating
        if len(movie_rating) == 1:
            rating = movie_rating.iloc()[0]
            observed_matrix[userIndex, movieIndex] = rating





movies_classifier = MoviesRecommender()

movies_classifier.Train(observed_matrix, user_to_index, movie_to_index)

unknown_set = pandas.read_csv('predictions.dat','\t')

movies_classifier.Predictions(unknown_set)

