import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import os
import argparse
import requests
from tqdm import tqdm

IS1_URL = "https://prts.wiki/w/%E5%88%BB%E4%BF%84%E6%9F%8F%E7%9A%84%E7%81%B0%E8%95%88%E8%BF%B7%E5%A2%83/%E6%94%B6%E8%97%8F%E5%93%81%E5%9B%BE%E9%89%B4"
IS2_URL = "https://prts.wiki/w/%E5%82%80%E5%BD%B1%E4%B8%8E%E7%8C%A9%E7%BA%A2%E5%AD%A4%E9%92%BB#%E9%95%BF%E7%94%9F%E8%80%85%E5%AE%9D%E7%9B%92%EF%BC%88%E6%94%B6%E8%97%8F%E5%93%81%E4%B8%80%E8%A7%88%EF%BC%89"
IS3_URL = "https://prts.wiki/w/%E6%B0%B4%E6%9C%88%E4%B8%8E%E6%B7%B1%E8%93%9D%E4%B9%8B%E6%A0%91/%E7%94%9F%E7%89%A9%E5%88%B6%E5%93%81%E9%99%88%E8%AE%BE"
IS4_URL = "https://prts.wiki/w/%E6%8E%A2%E7%B4%A2%E8%80%85%E7%9A%84%E9%93%B6%E5%87%87%E6%AD%A2%E5%A2%83/%E4%BB%AA%E5%BC%8F%E7%94%A8%E5%93%81%E7%B4%A2%E5%BC%95"
IS5_URL = "https://prts.wiki/w/%E8%90%A8%E5%8D%A1%E5%85%B9%E7%9A%84%E6%97%A0%E7%BB%88%E5%A5%87%E8%AF%AD/%E6%83%B3%E8%B1%A1%E5%AE%9E%E4%BD%93%E5%9B%BE%E9%89%B4" 
IS6_URL = "https://prts.wiki/w/%E5%B2%81%E7%9A%84%E7%95%8C%E5%9B%AD%E5%BF%97%E5%BC%82/%E7%8F%8D%E7%8E%A9%E9%9B%86%E5%86%8C"
ITEM_SELECTOR = "table.wikitable.logo"

options = Options()
options.add_argument("--headless")  # 如果你想要背景執行可以打開
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 10)

def IS_collect(url):
    items_data = []
    try:
        driver.get(url)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ITEM_SELECTOR)))
        items = driver.find_elements(By.CSS_SELECTOR, ITEM_SELECTOR)

        for item in items:
            record = {}
            # 1. No.、名稱、圖片（跟之前一樣）
            record["no"]   = item.find_element(By.XPATH, ".//tr[1]/th[1]").text.strip()
            record["name"] = item.find_element(By.XPATH, ".//tr[1]/th[2]").text.strip()
            img_el = item.find_element(By.XPATH, ".//tr[2]/td/img")
            record["img_url"] = img_el.get_attribute("data-src") or img_el.get_attribute("src")
        
            items_data.append(record)

    finally:
        driver.quit()
        return items_data

def png_collect(items: list):
    print(f"Collecting {len(items)} collectibles pictures...")
    for item in tqdm(items):
        img_url = item.get("img_url")
        # img_url example: "//torappu.prts.wiki/assets/roguelike_topic_itempic/rogue_4_relic_legacy_9.png"
        if img_url:
            img_url = "https:" + img_url
            # img_path = assets/roguelike_topic_itempic/rogue_4_relic_legacy_9.png
            img_path = img_url[img_url.find("assets/"):]
            # curl -o ./asset/roguelike_topic_itempic/rogue_4_relic_legacy_9.png https://torappu.prts.wiki/assets/roguelike_topic_itempic/rogue_4_relic_legacy_9.png
            os.makedirs(os.path.dirname(f"./{img_path}"), exist_ok=True)
            response = requests.get(img_url)
            if response.status_code == 200:
                with open(f"./{img_path}", "wb") as f:
                    f.write(response.content)
            else:
                print(f"Failed to download image from {img_url}, status code: {response.status_code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect collectibles data and pictures from PRTS IS pages.")
    parser.add_argument("--is1", action="store_true", help="Collect collectibles from #1: Ceobe's Fungimist")
    parser.add_argument("--is2", action="store_true", help="Collect collectibles from #2: Phantom & Crimson Solitaire")
    parser.add_argument("--is3", action="store_true", help="Collect collectibles from #3: Mizuki & Caerula Arbor")
    parser.add_argument("--is4", action="store_true", help="Collect collectibles from #4: Expeditioner's Jǫklumarkar")
    parser.add_argument("--is5", action="store_true", help="Collect collectibles from #5: Sarkaz's Furnaceside Fables")
    parser.add_argument("--is6", action="store_true", help="Collect collectibles from #6: Sui's Garden of Grotesqueries")
    args = parser.parse_args()
    if not (args.is1 or args.is2 or args.is3 or args.is4 or args.is5 or args.is6):
        print("Please specify at least one IS to collect collectibles from.")
        exit(1)
    # mkdir ./assets
    os.makedirs("./assets", exist_ok=True)
    if args.is1:
        items = IS_collect(IS1_URL)
        with open("./assets/IS1_Collectibles.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        png_collect(items)
    if args.is2:
        items = IS_collect(IS2_URL)
        with open("./assets/IS2_Collectibles.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        png_collect(items)
    if args.is3:
        items = IS_collect(IS3_URL)
        with open("./assets/IS3_Collectibles.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        png_collect(items)
    if args.is4:
        items = IS_collect(IS4_URL)
        with open("./assets/IS4_Collectibles.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        png_collect(items)
    if args.is5:
        items = IS_collect(IS5_URL)
        with open("./assets/IS5_Collectibles.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        png_collect(items)
    if args.is6:
        items = IS_collect(IS6_URL)
        with open("./assets/IS6_Collectibles.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=4)
        png_collect(items)