import requests
import tqdm

BASE_URL="http://www.buzzfeed.com/api/v2/feeds/index"


with open("data/clickbait.txt", "a+") as outfile:

    for page in tqdm.tqdm(range(0, 30)):
        response = requests.get(BASE_URL, { "p": page }).json()
        titles = [each["title"].encode("ascii", "ignore") for each in response["buzzes"]]
        outfile.write("\n" + "\n".join(titles))