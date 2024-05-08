# import streamlit as st
# import pandas as pd
# import numpy as np
# from scipy.sparse import csr_matrix
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# # import seaborn as sns

# # Load your datasets
# movies = pd.read_csv("movies.csv")
# ratings = pd.read_csv("ratings.csv")

# # Create the final dataset
# final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
# final_dataset.fillna(0, inplace=True)

# # Group by users and movies to count the number of votes
# no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
# no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# # Keep users with more than 50 votes
# final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# # Create a CSR matrix
# csr_data = csr_matrix(final_dataset.values)
# final_dataset.reset_index(inplace=True)

# # Fit the NearestNeighbors model
# knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# knn.fit(csr_data)

# # Define the function to get movie recommendations
# def get_movie_recommendation(movie_name):
#     n_movies_to_recommend = 10
#     movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    
#     if len(movie_list):
#         movie_idx = movie_list.iloc[0]['movieId']
#         movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
#         distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
#         rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]
#         recommend_frame = []
        
#         for val in rec_movie_indices:
#             movie_idx = final_dataset.iloc[val[0]]['movieId']
#             idx = movies[movies['movieId'] == movie_idx].index
#             recommend_frame.append({
#                 'Title': movies.iloc[idx]['title'].values[0],
#                 'Distance': val[1]
#             })
            
#         df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
#         return df
    
#     return "No movies found. Please check your input."

# # Streamlit user interface
# st.title("Movie Recommendation System")

# # Input box for the movie name
# movie_name = st.text_input("Enter a movie name to get recommendations:", "")

# # Button to get recommendations
# if st.button("Get Recommendations"):
#     if movie_name:
#         recommendations = get_movie_recommendation(movie_name)
#         if isinstance(recommendations, str):
#             st.write(recommendations)
#         else:
#             st.write("Here are your movie recommendations:")
#             st.dataframe(recommendations)
#     else:
#         st.write("Please enter a movie name.")

import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Load your datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Create the final dataset
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

# Group by users and movies to count the number of votes
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# Keep users with more than 50 votes
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Create a CSR matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Fit the NearestNeighbors model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Define the function to get movie recommendations
def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, regex=False)]
    
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[1:]
        recommend_frame = []
        
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({
                'Title': movies.iloc[idx]['title'].values[0],
                'Distance': val[1]
            })
            
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_recommend + 1))
        return df
    
    return "No movies found. Please check your input."

# Streamlit user interface
st.title("Movie Recommendation System")

# Input box for the movie name
movie_name = st.text_input("Enter a movie name to get recommendations:", "")

# Button to get recommendations
if st.button("Get Recommendations"):
    if movie_name:
        recommendations = get_movie_recommendation(movie_name)
        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write("Here are your movie recommendations:")
            st.dataframe(recommendations)

            # Plot a bar graph to show the "distance" of the recommended movies
            st.write("Distance of the recommended movies:")
            fig, ax = plt.subplots()
            ax.bar(recommendations["Title"], recommendations["Distance"], color='skyblue')
            ax.set_xlabel("Movies")
            ax.set_ylabel("Distance (Cosine Similarity)")
            ax.set_title("Recommended Movies vs. Distance")
            ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better visibility
            st.pyplot(fig)  # Render the plot in Streamlit
    else:
        st.write("Please enter a movie name.")

