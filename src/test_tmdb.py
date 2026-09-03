import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError("TMDB_API_KEY not found in .env")

url = "https://api.themoviedb.org/3/search/movie"

params = {
    "api_key": API_KEY,
    "query": "Inception"
}

response = requests.get(url, params=params, timeout=10)

print("Status:", response.status_code)

data = response.json()

if response.status_code != 200:
    print(data)
    raise SystemExit()

results = data.get("results", [])

print("Movies found:", len(results))

for movie in results[:5]:
    print()
    print("Title:", movie.get("title"))
    print("TMDB ID:", movie.get("id"))
    print("Poster:", movie.get("poster_path"))

    poster_path = movie.get("poster_path")

    if poster_path:
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
        print("Poster URL:", poster_url)