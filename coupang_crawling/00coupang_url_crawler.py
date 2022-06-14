from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import chromedriver_autoinstaller
import subprocess
import time
import warnings ; warnings.filterwarnings(action='ignore')

############## 디버거 크롬 코드 #####################
subprocess.Popen(r'C:\Program Files\Google\Chrome\Application\chrome.exe --remote-debugging-port=9222 --user-data-dir="C:\chrometemp"') # 디버거 크롬 구동

option = Options()
option.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

chrome_ver = chromedriver_autoinstaller.get_chrome_version().split('.')[0]
try:
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=option)
except:
    chromedriver_autoinstaller.install(True)
    driver = webdriver.Chrome(f'./{chrome_ver}/chromedriver.exe', options=option)
driver.implicitly_wait(10)
####################################################

f = open('./coupang_url_list', 'w', encoding='utf-8')
url_list=[]
query = "스마트폰"

# 아래 url은 쿠팡 웹사이트 좌측에 위치한 필터에서 가전디지털 -> 휴대폰 / 새 상품 선택한 상태의 url
# 쿠팡은 최대 27페이지의 상품 페이지만을 제공
for i in range(1,28):
    
    url = 'https://www.coupang.com/np/search?rocketAll=false&q='+query+'&brand=&offerCondition=NEW&filter=&availableDeliveryFilter=&filterType=&isPriceRange=false&priceRange=&minPrice=&maxPrice=&page='+str(i)+'&trcid=&traid=&filterSetByUser=true&channel=user&backgroundColor=&searchProductCount=99011&component=497144&rating=0&sorter=scoreDesc&listSize=36'
    driver.get(url)
    time.sleep(2)

    # 쿠팡 페이지당 36개 상품 정렬
    for j in range(1,37):
        titles = driver.find_element(by=By.XPATH, value='/html/body/div[3]/section/form/div[2]/div[2]/ul/li['+str(j)+']/a')
        title = titles.get_attribute('href')
        # url_list.append(title)
        f.write((title) + '\n')

print("url 수집 완료, 해당 url 데이터 크롤링")

