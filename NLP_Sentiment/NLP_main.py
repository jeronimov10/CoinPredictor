import feedparser

from transformers import pipeline

ticker = "BTC-USD"
keyword = 'BTC'

pipe = pipeline("text-classification", model="ProsusAI/finbert")

rss_url = f'https://finance.yahoo.com/rss/headline?s={ticker}'

feed = feedparser.parse(rss_url)

total_score = 0

num_articles = 0

for i, entry in enumerate(feed.entries):

    if keyword.lower() not in entry.summary.lower():
        continue


    print(f'Title: {entry.title}')
    print(f'Link: {entry.link}')
    print(f'Published: {entry.published}')
    print(f'Summary: {entry.summary}')

    sentimiento = pipe(entry.summary)[0]

    print(f'Sentimiento {sentimiento["label"]}, score: {sentimiento["score"]}')
    print('-' * 40)

    if sentimiento['label'] == 'positive':
        total_score += sentimiento['score']
        num_articles += 1
    elif sentimiento['label'] == 'negative':
        total_score -= sentimiento['score']
        num_articles += 1


final_score = total_score / num_articles

print(f'Sentimiento general: {"Positivo" if final_score >= 0.15 else "Negativo" if final_score <= -0.15 else "Neutral"} {final_score}')



