from newsapi import NewsApiClient
import pandas as pd

api = NewsApiClient(api_key='5eebff47dd2c48b799fa4c1206966735')

global_news = api.get_everything(q='global',
                                 sources='bbc-news,the-verge',
                                 domains='bbc.co.uk',
                                 language='en')

article_list = global_news['articles']
global_headlines = [x['title'] for x in article_list]
global_articles = [x['description'] for x in article_list]

data = pd.DataFrame({'headline': global_headlines, 'text': global_articles})
data.to_csv('data/news_data.csv', index=False)
