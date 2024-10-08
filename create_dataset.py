from firecrawl import FirecrawlApp
import os

entity = "movie"
fire_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
source_path = "dataset/source"

def create_dataset(url: str, file_name: int):
    scrape_result = fire_app.scrape_url(
        url, params={"formats": ["markdown"], "excludeTags": ["img", "video"]}
    )["markdown"]
    with open(f"dataset/article/{entity}/{file_name}.txt", "w") as f:
        f.write(scrape_result)


if __name__ == "__main__":
    with open(source_path + "/" + entity + ".txt", "r") as f:
        urls = f.readlines()
    for i, url in enumerate(urls):
        create_dataset(url, i)