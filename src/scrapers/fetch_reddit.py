import requests
import tqdm
import time

BASE_URL = "https://www.reddit.com/r/savedyouaclick/hot.json"
params = {
    "sort": "hot",
    "after": ""
}
headers = {
    "User-agent": "clickbait-detector scraper"
}
titles = list()
with open("data/clickbait-reddit.txt", "w") as outfile:
    for i in tqdm.tqdm(range(10)):
        response = requests.get(BASE_URL, params=params, headers=headers)
        time.sleep(1)
        if response.status_code == 200:
            response = response.json()
            params["after"] = response["data"]["after"]
            titles += [each["data"]["title"].split("|")[0].encode("ascii", "ignore") for each in response["data"]["children"]]
    outfile.write("\n".join(titles))
