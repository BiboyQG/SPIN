from firecrawl import FirecrawlApp
import os

fire_app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

def create_dataset(url: str, file_name: int):
    scrape_result = fire_app.scrape_url(
        url, params={"formats": ["markdown"], "excludeTags": ["a", "img", "video"]}
    )["markdown"]
    with open(f"dataset/article/{file_name}.txt", "w") as f:
        f.write(scrape_result)


if __name__ == "__main__":
    urls = [
        "https://www.caranddriver.com/reviews/a62019773/2025-volvo-ex90-drive/",
        "https://www.caranddriver.com/reviews/a62043196/2025-hyundai-tucson-hybrid-drive/",
        "https://www.caranddriver.com/reviews/a62017885/2025-volkswagen-jetta-drive/",
        "https://www.caranddriver.com/reviews/a62021190/2025-volkswagen-jetta-gli-test/",
        "https://www.caranddriver.com/reviews/a60777159/2025-genesis-gv80-suv-drive/",
        "https://www.caranddriver.com/reviews/a61828804/2024-mercedes-amg-gt63-coupe-test/",
        "https://www.caranddriver.com/reviews/a61689886/2025-lexus-ux300h-hybrid-test/",
        "https://www.caranddriver.com/reviews/a61557962/2024-toyota-land-cruiser-first-edition-test/",
        "https://www.caranddriver.com/reviews/a61069740/2025-honda-civic-hybrid-prototype-drive/",
        "https://www.caranddriver.com/reviews/a60499379/2025-toyota-camry-drive/"
    ]
    for i, url in enumerate(urls):
        create_dataset(url, i)