import os
from icrawler.builtin import BingImageCrawler

# === Configuration ===
categories = ["vegetables", "dairy food", "canned food", "baked food", "meat", "dry food"]
images_per_category = 30

# Ensure base data directory exists
os.makedirs("data/images", exist_ok=True)

for cat in categories:
    # normalize folder name
    folder = cat.replace(" ", "_")
    out_dir = os.path.join("data/images", folder)
    # create each category folder (this will also create any missing parents)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {images_per_category} images for '{cat}' into '{out_dir}'...")
    crawler = BingImageCrawler(storage={'root_dir': out_dir})
    crawler.crawl(keyword=cat, max_num=images_per_category)

print("Done!")
