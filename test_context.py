import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "backend", "src"))
# pyrefly: ignore [missing-import]
from adaptive_recommender import AdaptiveRecommender
# pyrefly: ignore [missing-import]
from tmdb_helper import get_poster_url

rec = AdaptiveRecommender()
idx, row = rec._resolve_title("Toy Story")
recs = rec.recommend("Toy Story", 10)

print(f"Toy Story main poster: {get_poster_url(rec.metadata.iloc[idx].to_dict())}")

for i, r in enumerate(recs):
    print(f"Rec {i+1} : {r['title']} -> {get_poster_url(r)}")
