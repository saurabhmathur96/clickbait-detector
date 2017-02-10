import requests
import tqdm
import time

BASE_URL = "https://www.reddit.com/r/savedyouaclick/top.json"
params = {
    "sort": "top",
    "t": "all",
    "after": ""
}
headers = {
    "User-agent": "clickbait-detector scraper v1.0"
}
titles = list()
with open("clickbait-top-reddit.txt", "w") as outfile:
    for i in tqdm.tqdm(range(100)):
        response = requests.get(BASE_URL, params=params, headers=headers)
        time.sleep(2)
        if response.status_code == 200:
            response = response.json()
            params["after"] = response["data"]["after"]
            titles += [each["data"]["title"].encode("ascii", "ignore").replace("\n", "").split("|")[0] for each in response["data"]["children"]]
	else:
	    time.sleep(10)
	    print "error"
    outfile.write("\n".join(titles))

