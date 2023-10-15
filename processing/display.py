import os
from processing import preprocess
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Main():

    def __enter__(self):
        # Initialization code, if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Cleanup code, if needed
        pass

    def __init__(self):
        self.new_df = None
        self.movies = None
        self.movies2 = None

    def getter(self):
        return self.new_df, self.movies, self.movies2

    def get_df(self):
        pickle_file_path = r'Files/new_df_dict.pkl'

        # Checking if preprocessed dataframe already exists or not
        if os.path.exists(pickle_file_path):

            # Read the Pickle file and load the dictionary -- 3 times
            # For the movies dataframe
            pickle_file_path = r'Files/movies_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)

            self.movies = pd.DataFrame.from_dict(loaded_dict)

            # Now, for the movies2 doing the same work
            pickle_file_path = r'Files/movies2_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict_2 = pickle.load(pickle_file)

            self.movies2 = pd.DataFrame.from_dict(loaded_dict_2)

            # Now, For new_df
            pickle_file_path = r'Files/new_df_dict.pkl'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_dict = pickle.load(pickle_file)

            self.new_df = pd.DataFrame.from_dict(loaded_dict)

        else:
            self.movies, self.new_df, self.movies2 = preprocess.read_csv_to_df()

            # Converting to pickle file (dumping file)
            # Convert the DataFrame to a dictionary

            #  Now, doing for the movies dataframw
            movies_dict = self.movies.to_dict()

            pickle_file_path = r'Files/movies_dict.pkl'
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(movies_dict, pickle_file)

            #  Now, doing for the movies2 dataframe
            movies2_dict = self.movies2.to_dict()

            pickle_file_path = r'Files/movies2_dict.pkl'
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(movies2_dict, pickle_file)

            # For the new_df
            df_dict = self.new_df.to_dict()

            # Save the dictionary to a Pickle file
            pickle_file_path = r'Files/new_df_dict.pkl'
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(df_dict, pickle_file)

    def vectorise(self, col_name):
        # Model to vectorise the words using CountVectorizer (Bag of words)
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vec_tags = cv.fit_transform(self.new_df[col_name]).toarray()
        sim_bt = cosine_similarity(vec_tags)
        return sim_bt

    def get_similarity(self, col_name):
        pickle_file_path = fr'Files/similarity_tags_{col_name}.pkl'
        if os.path.exists(pickle_file_path):
            pass
        else:
            similarity_tags = self.vectorise(col_name)

            # Converting to pickle file (dumping file)
            # Save the dictionary to a Pickle file
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(similarity_tags, pickle_file)

    def main_(self):
        # This is to make sure that resources are available.
        self.get_df()
        self.get_similarity('tags')
        self.get_similarity('genres')
        self.get_similarity('keywords')
        self.get_similarity('tcast')
        self.get_similarity('tprduction_comp')