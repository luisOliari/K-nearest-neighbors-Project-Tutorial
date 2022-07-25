from utils import db_connect
engine = db_connect()

# your code here

# importar liberiras:
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# cargar los dataset:
movies = pd.read_csv('../data/tmdb_5000_movies.csv')
credits = pd.read_csv('../data/tmdb_5000_credits.csv')

# ver los datos de movies:
movies.head(5)

#Merge both dataframes on the 'title' column.
movies = movies.merge(credits, on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]

# As there are only 3 missing values in the 'overview' column, drop them.

movies.isnull().sum()
movies.dropna(inplace = True)

#As you can see there are some columns with json format. With the following code, you can view what genres are included in the first row.
movies.iloc[0].genres

#We will start converting these columns using a function to obtain only the genres, without a json format. We are only interested in the values of the 'name' keys.
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies.dropna(inplace = True)

movies['genres'] = movies['genres'].apply(convert)
movies.head()

# Repeat the process for the 'keywords' column.
movies['keywords'] = movies['keywords'].apply(convert)

# For the 'cast' column we will create a new but similar function. This time we will limit the number of items to three.
def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L


movies['cast'] = movies['cast'].apply(convert3)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)

# Finally, let's look at the first row of the 'overview' column:
movies.overview[0]

movies['overview'] = movies['overview'].apply(lambda x: x.split())

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

#Now let's apply our function to the 'genres', 'cast', 'crew' and 'keywords' columns.
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['tags'] = movies['overview']+movies['genres'] + \
    movies['keywords']+movies['cast']+movies['crew']

new_df = movies[['movie_id','title','tags']]
new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))

# We will use KNN algorithm to build the recommender system. Before entering the model let's proceed with the text vectorization which you already learned in the NLP lesson.
cv = CountVectorizer(max_features=5000 ,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
vectors.shape

def recommend(movie):
    movie_index = new_df[new_df['title'] ==
                         movie].index[0]  # fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),
                        reverse=True, key=lambda x: x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)

recommend('The Dark Knight Rises')



