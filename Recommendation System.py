import json
import os.path

from flask import Flask, request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import pandas as pd
import re

app = Flask(__name__)
print('0')
if not os.path.exists("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv"):
    filename = "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"
    col_list = ['name', 'id', 'eng_categories', 'reviews.rating', 'reviews.username', 'eng_description']
    reviews_data = pd.read_csv(filename, usecols=col_list, sep=';', encoding='UTF-8')
    reviews_data.columns = ['product_id', 'product_name', 'categories', 'rating', 'user', 'description']
    c = 0
    am_user = []
    d1 = reviews_data.copy()
    for index, i in enumerate(reviews_data['user']):
        if 'customer' in i or 'user' in i or 'Customer' in i or 'User' in i or 'amazoncustomer' in i or 'amazonuser' in i or 'Amazoncustomer' in i or 'Amazonuser' in i:
            d1 = d1.drop(index)
            am_user.append(i)

    d1.to_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv',
              encoding='utf-8', index=False)
else:
    if os.path.exists("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv"):
        os.remove("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")


def normalize(value, old_max, old_min, new_max=10.0, new_min=0.0):
    """
        This functions used to normalize the ratings to new values
    """
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    return (((value - old_min) * new_range) / old_range) + new_min


def check_name(x):
    words = x.split()
    return all(x[0].isupper() and x[1].islower() for x in words if len(x) > 1)


def clean(x):
    if x.istitle() or check_name(x):
        return x.replace(" ", "").lower()
    else:
        return x.lower().strip()


def clean_text_non_english(string):
    pattern = re.compile('[^A-z0-9 ]+')
    q = re.sub("[^a-zA-Z0-9]+", " ", string)
    return q


def get_favorite_products(user_id, ratings_df):
    favorites = ratings_df[(ratings_df['user'] == user_id) & (ratings_df['rating'] >= 3.5)].sort_values(by='rating',
                                                                                                        ascending=False)[
        'product_id']
    return set(favorites if type(favorites) == pd.Series else [favorites])


class ContentBasedRecommender:
    MODEL_NAME = 'Content-Based'

    def __init__(self, items_ids, items_matrix, training_data, testing_data):
        self.items_ids = items_ids
        self.items_matrix = items_matrix
        self.training_data = training_data
        self.testing_data = testing_data

    def get_model_name(self):
        return self.MODEL_NAME

    def get_item_profile(self, item_id):
        idx = self.items_ids.index(item_id)
        return self.items_matrix[idx].toarray().reshape(-1)

    def get_items_profiles(self, ids):
        items_profiles = np.array([self.get_item_profile(x) for x in ids])
        return items_profiles

    def build_users_profile(self, user_id):
        user_df = self.training_data[self.training_data['user'] == user_id]
        user_items_profiles = self.get_items_profiles(user_df['product_id'].values)
        user_items_ratings = np.array(user_df['rating'].values).reshape(-1, 1)
        user_profile = np.sum(np.multiply(user_items_profiles, user_items_ratings), axis=0) / np.sum(user_items_ratings)

        if len(user_profile) == 0:
            return np.zeros([1, self.items_matrix.shape[1]], dtype='float')
        elif (np.sum(user_items_ratings)) == 0:
            return np.zeros([1, self.items_matrix.shape[1]], dtype='float')

        else:
            user_profile = np.sum(np.multiply(user_items_profiles, user_items_ratings), axis=0) / np.sum(
                user_items_ratings)
            return user_profile

    def get_similar_items_to_user_profile(self, user_id, topn=1000):
        user_profile = self.build_users_profile(user_id).reshape(1, -1)
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(user_profile, self.items_matrix.toarray())
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self.items_ids[i], cosine_similarities[0, i]) for i in similar_indices],
                               key=lambda x: -x[1])
        # print(similar_items, self.training_data[self.training_data['user'] == user_id]['product_name'].values)
        # print(len(similar_items))
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10):
        similar_items = self.get_similar_items_to_user_profile(user_id)
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['product_id', 'predicted_rating']) \
            .head(topn)

        recommendations_df['predicted_rating'] = recommendations_df['predicted_rating'].apply(
            lambda x: normalize(x, 1.0, 0.0))

        return recommendations_df


def get_recommendations(name, cosine_sim, df1):
    idx = []
    # Get the index of the food that matches the title
    # idx =df1.index[name in a for a in df1['categories'].values]
    cat = df1['information'].unique()
    for index, a in enumerate(cat):
        # print(
        if name in a:
            idx.append(index)
            # print(index)
    # Get the pairwsie similarity scores of all dishes with that food
    sim_scores = list(enumerate(cosine_sim[idx[0]]))

    # Sort the dishes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar dishes
    sim_scores = sim_scores[1:]
    # Get the food indices
    food_indices = [i[0] for i in sim_scores]
    d = []
    for i in range(len(food_indices)):
        var = df1['information'].unique()[food_indices[i]]
        dd = df1[df1['information'] == var]
        for j in dd['product_id']:
            if j in d:
                continue
            d.append(j)
    return d


def cf_preparation(d1):
    users_items_pivot_matrix_df = d1.pivot_table(index='user', columns='product_id', values='rating').fillna(0)
    users_items_pivot_matrix = users_items_pivot_matrix_df.values
    # The number of factors to factor the user-item matrix.
    NUMBER_OF_FACTORS_MF = 15
    # Performs matrix factorization of the original user item matrix
    # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
    U, sigma, Vt = svds(users_items_pivot_matrix, k=NUMBER_OF_FACTORS_MF)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=users_items_pivot_matrix_df.columns,
                            index=users_items_pivot_matrix_df.index)
    preds_df = preds_df.apply(
        lambda x: normalize(x, all_user_predicted_ratings.max(), all_user_predicted_ratings.min()))
    return preds_df


class CFRecommender:
    MODEL_NAME = 'Collaborative Filtering'

    def __init__(self, predictions_df):
        self.predictions_df = predictions_df

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        # Get and sort the user's predictions
        sorted_user_predictions = self.predictions_df.loc[user_id].sort_values(ascending=False)
        recommendations = {'product_id': sorted_user_predictions.index,
                           'predicted_rating': sorted_user_predictions.values}
        recommendations_df = pd.DataFrame(recommendations)
        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations_df = recommendations_df[~recommendations_df['product_id'].isin(items_to_ignore)] \
            .sort_values('predicted_rating', ascending=False) \
            .head(topn)

        return recommendations_df


class HybridRecommender:
    MODEL_NAME = 'Hybrid'

    def __init__(self, cb_model, c_model):
        self.content_model = cb_model
        self.colla_model = c_model

    def get_model_name(self):
        return self.MODEL_NAME

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        content_model_df = self.content_model.recommend_items(user_id, items_to_ignore, topn)
        c_model_df = self.colla_model.recommend_items(user_id, items_to_ignore, topn)

        recommendations_df = content_model_df.append(c_model_df) \
            .sort_values('predicted_rating', ascending=False) \
            .head(topn)
        return recommendations_df


def popularity_pre(d1):
    populartiy = d1.groupby('product_id').agg({'rating': ['mean', 'count']}).reset_index()
    populartiy.columns = ['product_id', 'ratings_mean', 'ratings_count']
    # populartiy['year'] = d1[d1['product_id'].isin(populartiy['product_id'])].values
    populartiy.sort_values(by='ratings_mean', ascending=False)

    return populartiy


class PopularityRecommender:
    MODEL_NAME = 'Popularity'

    def __init__(self, popularities_df):
        self.popularities_df = popularities_df

    def get_model_name(self):
        return self.MODEL_NAME

    def weighted_rating(self, x, m, C):
        v = x['ratings_count']
        R = x['ratings_mean']
        # Calculation based on the IMDB formula
        return (v / (v + m) * R) + (m / (m + v) * C)

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        C = self.popularities_df['ratings_mean'].mean()

        self.popularities_df['predicted_rating'] = self.popularities_df.apply(lambda x: self.weighted_rating(x, 3.5, C),
                                                                              axis=1)

        recommendations_df = self.popularities_df[~self.popularities_df['product_id'].isin(items_to_ignore)] \
            .sort_values('predicted_rating', ascending=False) \
            .head(topn)

        return recommendations_df


print('00')


@app.route('/insert_rate', methods=["POST", "GET"])
def insert():
    new_product_name = request.form['name']
    new_product_id = request.form['id']
    new_product_rate = request.form['rating']
    new_product_username = request.form['user_name']
    new_product_desc = request.form['description']
    new_product_category = request.form['categories']
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf8')
    gg = data.loc[(data['user'] == new_product_username) & (data['product_id'] == int(new_product_id)), 'rating']
    if len(gg) != 0:
        data.loc[(data['user'] == new_product_username) & (
                data['product_name'] == new_product_name), 'rating'] = int(new_product_rate)
        data.to_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf-8',
                    index=False)
    else:

        data.loc[len(data.index)] = [int(new_product_id), new_product_name, new_product_category, int(new_product_rate),
                                     new_product_username, new_product_desc]
        data.to_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf-8',
                    index=False)
    return 'true'


@app.route('/delete_rate', methods=["POST", "GET"])
def delete():
    new_product_id = request.form['id']
    new_product_username = request.form['user_name']
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf8')
    new_product_id = int(new_product_id)
    gg = data.loc[(data['user'] == new_product_username) & (data['product_id'] == int(new_product_id)), 'rating']
    if len(gg) == 0:
        return 'false'
    data = data.drop(gg.index[0])
    data.to_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf-8',
                index=False)
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf8')
    gg = data.loc[(data['user'] == new_product_username) & (data['product_id'] == int(new_product_id)), 'rating']
    if len(gg) == 0:
        return 'true'
    else:
        return 'false'


@app.route('/delete_product', methods=["POST", "GET"])
def delete_product():
    new_product_id = request.form['id']
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf8')
    new_product_id = int(new_product_id)
    gg = data[data['product_id'] == int(new_product_id)]
    if len(gg) == 0:
        return 'false'
    data = data[~data['product_id'].isin(gg['product_id'])]
    data.to_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf-8',
                index=False)
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv', encoding='utf8')
    gg = data.loc[(data['product_id'] == int(new_product_id))]

    if len(gg) == 0:
        return 'true'
    else:
        return 'false'


@app.route('/recommend_product_out', methods=["POST", "GET"])
def recommend_out():
    user_name = request.form['user_name']
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv',
                       encoding='utf8')
    data.insert(0, 'information', data["product_name"] + data["categories"] + data["description"])
    data['information'] = data['information'].apply(lambda x: clean(x))
    data['information'] = data['information'].apply(lambda x: clean(x))
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['information'].unique())
    cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)
    if user_name not in list(set(data['user'].values)):
        popularity = popularity_pre(data)
        popularity_recommender = PopularityRecommender(popularity)
        ids = popularity_recommender.recommend_items(user_name)
        result = data[data['product_id'].isin(ids['product_id'].values)]['product_name'].unique()
        list_of_product_ids = ids['product_id'].values
        list_of_product_ids = list_of_product_ids.tolist()
        lists = result.tolist()
        print(len(list_of_product_ids))
        json_str = json.dumps(list_of_product_ids)
        return json_str
    else:
        items_to_ignore = get_favorite_products(user_name, data)
        preds_df = cf_preparation(data)
        cf_recommender_model = CFRecommender(preds_df)
        content_based_recommender_model_soup = ContentBasedRecommender(data['product_id'].unique().tolist(),
                                                                       tfidf_matrix,
                                                                       data, data)
        hybridrecommender1 = HybridRecommender(content_based_recommender_model_soup, cf_recommender_model)
        ids = hybridrecommender1.recommend_items(user_name, items_to_ignore)['product_id'].values
        result = data[data['product_id'].isin(ids)]['product_name'].unique()
        list_of_product_ids = ids.tolist()
        lists = result.tolist()
        print(len(list_of_product_ids))
        json_str = json.dumps(list_of_product_ids)
        return json_str


@app.route('/recommend_product_in', methods=["POST", "GET"])
def recommend_in():
    new_product_id = request.form['id']
    data = pd.read_csv('Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19_updated.csv',
                       encoding='utf8')
    data['product_name'] = data['product_name'].apply(lambda x: clean(x))
    data["categories"] = data['categories'].apply(lambda x: clean(x))
    data["description"] = data['description'].apply(lambda x: clean(x))

    data.insert(0, 'information', data["product_name"] + data["categories"] + data["description"])
    new_product_id = int(new_product_id)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['information'].unique())
    cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)
    product_name = data.loc[(data['product_id'] == int(new_product_id)), 'product_name']
    product_name = clean(product_name.values[0])
    # print(product_name)
    a = get_recommendations(product_name, cosine_sim1, data)
    # list_of_product_ids = data[data['product_name'].isin(a)]['product_id'].unique()
    # list_of_product_ids = list_of_product_ids.tolist()
    # a = a.tolist()
    json_str = json.dumps(a)
    return json_str


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
