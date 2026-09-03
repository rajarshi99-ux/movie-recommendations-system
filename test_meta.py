import pandas as pd
df = pd.read_csv('dataset/movies_cleaned.csv')
movies = ['Inception', 'The Dark Knight', 'The Dark Knight Rises', 'Heat', 'Need for Speed', 'Death Wish', 'Training Day', 'The Arrival', 'Arrival', 'Interstellar', 'Toy Story']
for t in movies:
    r = df[df['title'].str.lower() == t.lower()]
    for idx, row in r.iterrows():
        print(row['title'] + ' | ID: ' + str(row['id']) + ' | Poster: ' + str(row['poster_path']))
