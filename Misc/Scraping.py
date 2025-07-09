from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
import pandas as pd
import time

# adjust path if needed
PATH = 'c:\\Users\\Jay Tai\\Downloads\\edgedriver_win32\\msedgedriver.exe'
service = Service(PATH)
driver = webdriver.Edge(service=service)

driver.get("https://rollcall.com/factbase/trump/topic/social/")
time.sleep(5)  # wait for JS to load content

posts = []
while True:
    items = driver.find_elements(By.CSS_SELECTOR, ".post-list-item")
    for item in items:
        try:
            ts = item.find_element(By.TAG_NAME, "time").get_attribute("datetime")
            content = item.find_element(By.CSS_SELECTOR, ".post-content").text
            posts.append({'timestamp': ts, 'content': content})
        except Exception:
            continue
    # Try to click “Load more” if present
    try:
        load_more = driver.find_element(By.LINK_TEXT, "Load More")
        load_more.click()
        time.sleep(3)
    except:
        break

driver.quit()

df = pd.DataFrame(posts)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.to_csv("factbase_trump_posts.csv", index=False)
print("Scraped", len(df), "posts")
