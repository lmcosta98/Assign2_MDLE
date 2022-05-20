from pyspark import SparkContext
import re
import sys
import math
import random
import numpy as np
import hashlib
import time
from datetime import datetime


'''
    Create a ratings matrix 
    keys -> moviesID
    values -> list of ratings => indexes: usersID
'''
def get_matrix(by_movie, num_users):
    matrix = {}
    for movieID,value in by_movie.collect():
        users_vector = [0 for j in range(num_users)]
        for userID,rating in value:
            users_vector[int(userID)-1] = float(rating)

        matrix[movieID] = users_vector   
        
    return matrix


'''
    Calculate similarity between two vectors
'''
def calc_similarity(vector1, vector2):
    mean_vector1 = np.mean([value for value in vector1 if value!=0])
    v1 = [value-mean_vector1 for value in vector1]
    
    mean_vector2 = np.mean([value for value in vector2 if value!=0])
    v2 = [value-mean_vector2 for value in vector2]

    sim = np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1) * np.dot(v2, v2))) #Cosine    
    return sim


'''
      5 Most similar items (neighbors) to a given movie
'''
def get_similar_movies(ratings_matrix, by_users, movie, selected_user):
    similar_movs = []
    # Movies user has seen
    users_dict = by_users.collect()
    for user, ratings in users_dict:
        user_idx = int(user)-1
        if user_idx == selected_user:
            # Iterate movies
            for movie_2 in ratings_matrix:
                # Only watched movies
                if movie != movie_2 and ratings_matrix[movie_2][selected_user] != 0:
                    sim = calc_similarity(ratings_matrix[movie], ratings_matrix[movie_2])
                    similar_movs.append((movie_2, ratings_matrix[movie_2][selected_user], sim))
                        
    similar_movs.sort(key=lambda y: y[2], reverse=True)
    # Only 5 more similar
    return similar_movs[:5] 


'''
    Estimate a rating based on similar_movies
'''
def estimate_rating(similar_movs):
    rating_predicted = 0
    similarity_sum = 0
    for m1, rating, sim in similar_movs:
        similarity_sum += sim
        rating_predicted += rating*sim
    
    return round(rating_predicted / similarity_sum, 1)


'''
    Get ratings predictions
'''
def model_evaluation(data_test, total_matrix):
    real_ratings = []
    predicted_ratings = []
    print(len(data_test))
    i = 0
    for line in data_test:
        
        user, movie, rating, _ = line
        user = int(user)-1
        real_ratings.append(float(rating))
        sm = get_similar_movies(total_matrix, by_users, movie, user) # Neighbor selection
        predicted_ratings.append(estimate_rating(sm)) # Estimated rating
        print(i)
        i+=1
    return real_ratings, predicted_ratings


'''
    Calculate RMSE
'''
def calc_rmse(real_ratings, predicted_ratings):
    quo = sum([ (predicted_ratings[i]-real_ratings[i])**2 for i in range(len(real_ratings))])
    den = len(predicted_ratings)
    return math.sqrt( quo / den )


'''
    Calculate Presision at top 10
'''
def calc_precision(real_ratings, predicted_ratings):
    quo = sum([1 for i in range(len(real_ratings)) if (round(real_ratings[i])==round(predicted_ratings[i]))])
    den = len(predicted_ratings)
    precision = quo / den
    return precision


if __name__ == "__main__":
    file = "assign2/ratings_small.csv"
    
    try:
        file = sys.argv[1]
        
    except:
        print("Usage: spark-submit ex2.py <file>")
        exit(1)
    
    sc = SparkContext(master='local', appName="Assignment2_E2")
    sc.setLogLevel("WARN")
    begin = time.time()
    print("TIME START!!!")
    ratings_data = sc.textFile(file)
    header = ratings_data.first() #extract header
    ratings_data = ratings_data.filter(lambda line: line != header)   #filter out header
    
    data = ratings_data.map(lambda line: re.split(',', line.lower()))
    
    # Get ratings by user
    by_users = data.map(lambda elem: (elem[0],(elem[1], elem[2]))).groupByKey().mapValues(list)
    num_users = by_users.count()
    
    # Get ratings by movies
    by_movie = data.map(lambda elem: (elem[1],(elem[0], elem[2]))).groupByKey().mapValues(list)
    num_movies = by_movie.count()
    
    # Get matrix
    ratings_matrix = get_matrix(by_movie, num_users)
    
    # Get 10% of the data to evaluate
    evaluation_size = round(data.count() * 0.1)
    
    # Get rating predictions
    print("BEFORE PREDICTIONS!!!")
    real_ratings, predicted_ratings = model_evaluation(data.take(evaluation_size), ratings_matrix)
    
    print("Top 10 real ratings: ", real_ratings[:10])
    print("Top 10 predicted ratings: ", predicted_ratings[:10])
    
    ''' Evaluation Metrics'''
    rmse = calc_rmse(real_ratings, predicted_ratings)
    print("RMSE value: {}".format(rmse))
    
    precision_10 = calc_precision(real_ratings[:10], predicted_ratings[:10])
    print("Precision at top 10 value: {}".format(precision_10))
    
    elapsed_time = time.time() - begin
    print("Elapsed time: ",elapsed_time)
    
    result = [("Precision_10", precision_10), ("RMSE", rmse), ("Elapsed Time", elapsed_time)]
    result_rdd = sc.parallelize(result)
    
    # Save results
    format_time = str(datetime.now().strftime("%Y-%m-%dT%H_%M_%S"))
    result_rdd.saveAsTextFile("{0}_{1}".format("assign2/results/ex2_"+str(evaluation_size), format_time))
    
    sc.stop()