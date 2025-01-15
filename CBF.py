import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import ssl
import nltk
from rake_nltk import Rake

# Ensure SSL issues are bypassed
ssl._create_default_https_context = ssl._create_unverified_context

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')


class Content_Based_Filtering:

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df.dropna(inplace=True)  # Remove rows with missing values
        print(f"Data loaded. Shape: {self.df.shape}")

    def pre_process(self):
        """Preprocess the DataFrame."""
        self.df = self.df[["Title", "Genre", "Description", "Director", "Actors"]]

        # Process Actors column
        self.df["Actors"] = self.df["Actors"].map(lambda x: x.split(',')[:3])

        # Process Genre column
        self.df["Genre"] = self.df["Genre"].map(lambda x: x.lower().split(','))

        # Process Director column
        self.df["Director"] = self.df["Director"].map(lambda x: x.split(" "))

        # Convert Actors and Director columns to lowercase
        for index, row in self.df.iterrows():
            self.df.at[index, "Actors"] = [x.lower().replace(" ", '') for x in row["Actors"]]
            self.df.at[index, "Director"] = ''.join(row["Director"]).lower()

        # Extract keywords using Rake
        self.df["Key_words"] = ""
        for index, row in self.df.iterrows():
            plot = row["Description"]
            r = Rake()
            r.extract_keywords_from_text(plot)
            key_words_dict_scores = r.get_word_degrees()
            self.df.at[index, "Key_words"] = list(key_words_dict_scores.keys())

        # Drop the Description column
        self.df.drop(columns=["Description"], inplace=True)

        # Generate the bag_of_words column
        self.df["bag_of_words"] = ""
        columns = ["Genre", "Actors", "Director", "Key_words"]
        for index, row in self.df.iterrows():
            words = ""
            for col in columns:
                if isinstance(row[col], list):
                    words += ' '.join(row[col]) + ' '
                else:
                    words += row[col] + ' '
            self.df.at[index, "bag_of_words"] = words.strip()

        print(f"Data after preprocessing:\n{self.df.head()}")

    def generate_count_matrix(self):
        """Generate the count matrix and cosine similarity."""
        count = CountVectorizer()
        self.count_matrix = count.fit_transform(self.df["bag_of_words"])
        self.cosine_sim = cosine_similarity(self.count_matrix, self.count_matrix)

        # Reset the index to ensure it matches the original DataFrame
        self.df.reset_index(inplace=True, drop=False)
        self.indices = pd.Series(self.df.index, index=self.df["Title"])

        print(f"Count matrix and cosine similarity generated.")

    def recommend(self, title, top_n=10):
        """Recommend movies based on cosine similarity."""
        # Ensure case-insensitive matching
        if title not in self.indices.index:
            print(f"Error: Title '{title}' not found in dataset.")
            return []

        # Get the index of the movie
        idx = self.indices[title]

        # Get similarity scores for all movies
        scores = pd.Series(self.cosine_sim[idx]).sort_values(ascending=False)

        # Get the indices of the top N similar movies
        top_indices = list(scores.iloc[1:top_n+1].index)

        # Return the titles of the recommended movies
        recommended_movies = self.df.loc[top_indices, "Title"].tolist()
        return recommended_movies


# Instantiate the class
cbf = Content_Based_Filtering("imdb_movie_dataset.csv")

# Preprocess the data
cbf.pre_process()

# Generate the count matrix and cosine similarity
cbf.generate_count_matrix()

# Get recommendations for a specific movie
recommendations = cbf.recommend("Sicario", top_n=10)
print(f"Top 10 recommended movies based on your choice: {recommendations}")