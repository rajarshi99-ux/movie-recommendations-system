import sys, os
import requests
sys.path.insert(0, os.path.abspath('src'))
from recommendation_engine import MovieRecommender
from tmdb_helper import get_poster_url_with_path

rec = MovieRecommender()
for title in ['Inception', 'Minority Report', 'The Dark Knight', 'Toy Story']:
    details = rec.get_movie_details(title)
    poster_path = details.get('poster_path')
    url = get_poster_url_with_path(poster_path, title)
    print(f'Title: {title}')
    print(f'Local poster_path: {poster_path}')
    print(f'Generated URL: {url}')
    
    try:
        r = requests.get(url, timeout=3)
        print(f'HTTP: {r.status_code}, Content-Type: {r.headers.get("Content-Type")}')
    except Exception as e:
        print(f'HTTP Error: {e}')
    print('-'*50)
