from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import pandas as pd

PATH = 'c:\\Users\\Jay Tai\\Downloads\\edgedriver_win32\\msedgedriver.exe'
service = Service(PATH)
options = webdriver.EdgeOptions()
options.add_argument('--headless')
driver = webdriver.Edge(service=service, options=options)

driver.get("https://rollcall.com/factbase/trump/topic/social/")

time.sleep(5)

results = pd.DataFrame(columns=['datetime', 'text', 'isRetweet'])

def get_results(stop):
    result_container = driver.find_element(By.CSS_SELECTOR, '.page.page-id-749153.wp-embed-responsive.factbase-twitter').find_element(By.ID, 'app').find_element(By.CLASS_NAME, 'p-0.m-0.leading-4.text-black.align-baseline.border-0.duration-300').find_element(By.ID, 'main').find_element(By.ID, 'app').find_element(By.CSS_SELECTOR, '.w-full.px-4.md\\:px-0').find_element(By.CSS_SELECTOR, '.m-auto.bg-white\\/80.p-4.shadow-xl.rounded-2xl').find_element(By.CSS_SELECTOR, '.max-w-7xl.mx-auto.px-4.md\\:px-0').find_element(By.CLASS_NAME,'results-container').find_elements(By.CSS_SELECTOR,'.block.mb-8.rounded-xl.border')
    for i in result_container:
        result_datetime = i.find_element(By.CSS_SELECTOR, '.p-4').find_element(By.CSS_SELECTOR, '.flex-1').find_element(By.CSS_SELECTOR, '.flex.flex-col.md\\:flex-row.items-start').find_element(By.CSS_SELECTOR, '.flex.items-center').find_element(By.CSS_SELECTOR, '.text-sm.text-gray-500').find_elements(By.CSS_SELECTOR, '.hidden.md\\:inline')[1].get_attribute("textContent")
        result_text = i.find_element(By.CSS_SELECTOR, '.p-4').find_element(By.CSS_SELECTOR, '.flex-1').find_element(By.CSS_SELECTOR, '.text-sm.font-medium.whitespace-pre-wrap.leading-relaxed').get_attribute("textContent")  
        rt = result_text.startswith("RT")

        if result_datetime == "" or result_text == "":
            continue

        if result_text not in results['text'].values and 'https://' not in result_text: 
            results.loc[len(results)] = [result_datetime, result_text, rt]
    
        if stop in result_datetime:
            print("Reached the end date")
            return False
        
    return True

stop = "November 6, 2024"
c = True

try:
    while c:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)  # wait for new content to load
        c = get_results(stop)

except Exception as e:
    print("An error occurred:", e)

finally:
    print("Saving results and quitting driver...")
    print(results.tail())
    results.to_csv('data/News Articles/trump_social_results2.csv', index=False)
    driver.quit()




