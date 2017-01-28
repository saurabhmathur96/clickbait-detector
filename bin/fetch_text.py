import requests
import itertools
import tqdm

ARTICLES_URL = "https://newsapi.org/v1/articles"
API_KEY = "" # API Key here

SOURCES_URL = "https://newsapi.org/v1/sources?language=en"
sources_response = requests.get(SOURCES_URL).json()
all_source_details = sources_response["sources"]

clickbait_sources = open("data/clickbait.sources.txt").read().strip().split()
genuine_sources = open("data/genuine.sources.txt").read().strip().split()

def find_details(id):
    return next((x for x in all_source_details if x["id"] == id), None)

clickbait_details = filter(bool, [find_details(id) for id in clickbait_sources])
genuine_details = filter(bool, [find_details(id) for id in genuine_sources])

def fetch_headlines(source_details):
    source_id, sort_bys_available = source_details["id"], source_details["sortBysAvailable"]
    def fetch_articles(source_id, sort_by, api_key):
        response = requests.get(ARTICLES_URL, { "source": source_id, "sortBy": sort_by, "apiKey": api_key}).json()
        return response["articles"]
    articles = [fetch_articles(source_id, sort_by, API_KEY) for sort_by in sort_bys_available]
    titles = [article["title"] for article in itertools.chain.from_iterable(articles)]
    return list(set(titles))

clickbait_headlines = list(itertools.chain.from_iterable(fetch_headlines(details) for details in tqdm.tqdm(clickbait_details, desc="fetching clickbait")))
genuine_headlines = list(itertools.chain.from_iterable(fetch_headlines(details) for details in tqdm.tqdm(genuine_details, desc="fetching genuine")))

open("data/clickbait.txt", "w").write("\n" + "\n".join(clickbait_headlines).encode("ascii", "ignore"))
open("data/genuine.txt", "w").write("\n" + "\n".join(genuine_headlines).encode("ascii", "ignore"))