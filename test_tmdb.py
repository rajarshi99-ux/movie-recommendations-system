import sys
sys.path.insert(0, './src')
from tmdb_helper import get_poster_url
from adaptive_recommender import AdaptiveRecommender
rec = AdaptiveRecommender()
for q in ['Inception', 'The Dark Knight Rises', 'Heat', 'Need for Speed', 'Death Wish', 'Training Day']:
    res = rec.get_movie_details(q)
    print(q, ':', get_poster_url(res))
