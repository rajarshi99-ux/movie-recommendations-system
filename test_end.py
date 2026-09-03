import sys
sys.path.insert(0, './src')
from tmdb_helper import get_poster_url
from adaptive_recommender import AdaptiveRecommender
rec = AdaptiveRecommender()
for q in ['The Dark Knight', 'Inception']:
    print('--- ' + q + ' ---')
    for r in rec.recommend(q, 3):
        print(r['title'] + ': ' + str(get_poster_url(r)))
