from bs4 import BeautifulSoup
import glob
import tqdm


with open("data/genuine.txt", "a+") as outfile:

    for filename in tqdm.tqdm(glob.glob("data/feed/*.xml")):
        with open(filename) as f:
            soup = BeautifulSoup(f.read())
            titles = [each.find("title").text.encode("ascii", "ignore") for each in soup.find_all("item")]
            outfile.write("\n" + "\n".join(titles))