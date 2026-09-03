import requests, json
with open('meta.json', 'r') as f:
    data = json.load(f)
for m in data:
    url = 'https://image.tmdb.org/t/p/w500' + m['poster_path']
    try:
        res = requests.head(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=2.0)
        print(m['title'] + ' | ' + str(res.status_code))
    except Exception as e:
        print(m['title'] + ' | Error')
