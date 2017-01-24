import requests
import tqdm

BASE_URL = "http://content.guardianapis.com/search"
params = {
    "from-date": "2014-07-01",
    "to-date": "2017-01-23",
    "api-key": "", # API Key here
    "page": 1
}

with open("data/genuine.txt", "a+") as outfile:

    for page in tqdm.tqdm(range(1, 301), desc="fetching headlines from guardian"):
        params["page"] = page
        response = requests.get(BASE_URL, params=params).json()
        results = response["response"]["results"]
        titles = [result["webTitle"].encode("ascii", "ignore").replace("\n", "") for result in results]
        outfile.write("\n" + "\n".join(titles))
