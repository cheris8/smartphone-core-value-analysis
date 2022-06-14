import pandas as pd
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from fake_useragent import UserAgent
from tqdm import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

path = './chromedriver.exe'
#'coupang_url_list는 00coupang_url_crawler.py에서 추출된 상품 url 리스트
f = open('coupang_url_151')

a=0

ua = UserAgent()
user_agent = ua.random
headers = {'User-Agent': user_agent}
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument(f'user-agent={user_agent}')
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)
chrome_options.add_experimental_option("prefs", {"prfile.managed_default_content_setting.images": 2})
driver = webdriver.Chrome(path, chrome_options=chrome_options)
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument",
                       {"source": """ Object.defineProperty(navigator, 'webdriver', { get: () => undefined }) """})
driver.maximize_window()

# login
loginurl = 'https://login.coupang.com/login/login.pang'
driver.get(url=loginurl)
time.sleep(2)
id_input = driver.find_element(by=By.XPATH, value='//*[@id="login-email-input"]')
id_input.send_keys('user ID')  #######쿠팡 아이디(이메일)
pw_input = driver.find_element(by=By.XPATH, value='//*[@id="login-password-input"]')
pw_input.send_keys('password')  ######쿠팡 비밀번호
driver.find_element(by=By.XPATH, value='/html/body/div[1]/div/div/form/div[5]/button').click()
driver.implicitly_wait(10)
time.sleep(10)

urls = f.readlines()
for url in urls:

    ####################### 링크 불러오기 ########################
    driver.get(url)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    ######화면 Zoom out######

    # driver.execute_script("document.body.style.zoom='35%'")

    ###################### 스크롤내리기 ##########################
    driver.execute_script("window.scrollTo(0, window.scrollY + 600);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, window.scrollY + 600);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, window.scrollY + 500);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, window.scrollY + 500);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, window.scrollY + 500);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, window.scrollY + 500);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, window.scrollY + 500);")
    time.sleep(2)
    driver.execute_script("window.scrollTo(0, window.scrollY + 500);")
    time.sleep(2)

    ###################### 상품평 page 추출 ##############################
    page_sub = soup.find_all('span', {'class': 'count'})
    for x in page_sub:
        page = x.get_text()

    page_index_last = page.rfind('개 상품평')
    total = page[:page_index_last].strip().replace(',', '').replace('(', '').replace(')', '')

    # 리뷰 1000개 이상 시 최대 페이지 수가 200 페이지
    if int(total) > 1000:
        page = 200
    else:
        page = int(total) // 5 + 1

    product = [];
    price = [];
    company = [];
    rating = [];
    review = []
    count = 3

    error_count = 0
    # 상품명
    product_sub = soup.find_all('h2', {'class': 'prod-buy-header__title'})
    for x in product_sub:
        # f.write((x.get_text()) + '\t')
        product.append(x.get_text())

    # 가격
    temp = []
    price_sub = soup.find_all('span', {'class': 'total-price'})
    for x in price_sub:
        temp.append(str(x))

    if temp:
        price_sub = temp[0]
        first_index = price_sub.find('<strong>')
        first_index += 8
        last_index = price_sub.rfind('<span class')
        # f.write((price_sub[first_index:last_index]) + '\t')
        price.append(price_sub[first_index:last_index])
    else:
        price.append('none')

    # 제조회사
    company_sub = soup.find_all('a', {'class': 'prod-brand-name'})
    for x in company_sub:
        x = str(x)
        index_first = x.find('data-brand-name')
        index_first += len('data-brand-name="')
        index_last = x.find('data-coulog')
        if x[index_first:index_last - 2] == '':  # 제조회사가 없는 경우
            # f.write(('none') + '\t')
            company.append('none')
        else:
            # f.write((x[index_first:index_last - 2]) + '\t')
            company.append(x[index_first:index_last - 2])

    # # 리뷰 최신순으로 변경 (default 베스트순)
    # best2new = driver.find_element(by=By.XPATH, value='//*[@id="btfTab"]/ul[2]/li[2]/div/div[6]/section[2]/div[1]/button[2]')
    # best2new.click()
    # driver.implicitly_wait(10)

    for y in tqdm(range(1, page + 1)):
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # 별점
        rating_sub = soup.find_all('div', {'class': 'sdp-review__article__list__info__product-info__star-gray'})
        for x in rating_sub:
            x = str(x)
            index_first = x.find('data-rating')
            index_first += len('data-rating="')
            index_last = x.find('style')
            # f.write((x[index_first:index_last - 2]) + '\t')
            rating.append(x[index_first:index_last - 2])

        # 리뷰
        review_sub = soup.find_all('article', {'class': 'sdp-review__article__list js_reviewArticleReviewList'})

        for x in review_sub:
            temp_1 = x.find('div', {'class': 'sdp-review__article__list__review__content js_reviewArticleContent'})
            if temp_1 is not None:
                # f.write((temp.get_text()) + '\t')
                review.append(temp_1.get_text())
            else:
                # f.write(('none') + '\t')
                review.append('none')

        # xpath 지정: 쿠팡의 주소는 button[] 내의 숫자의 차이로 xpath가 바뀜.
        xpath = '//*[@id="btfTab"]/ul[2]/li[2]/div/div[6]/section[4]/div[3]/button[' + str(count) + ']'
        xpath_1 = '//*[@id="btfTab"]/ul[2]/li[1]/div/div[6]/section[4]/div[3]/button[' + str(count) + ']'

        # 마지막 페이지에는 다음으로 넘기는 버튼이 없음
        # 에러 발생하는 이유 확인이 안됨
        if y != page:
            # page 넘기기
            if temp:
                try:
                    next_page = driver.find_element(by=By.XPATH, value=xpath)
                except:
                    print('에러1')
                    error_count += 1
                    continue
                time.sleep(2)
            else:
                try:
                    next_page = driver.find_element(by=By.XPATH, value=xpath_1)
                except:
                    print("에러2")
                    error_count += 1
                    continue
                time.sleep(2)

            # 10 페이지 단위로 button[]사이의 값이 바뀜. 범위 : 2 ~ 12
            if count == 12:
                count = 3  # 첫번째 페이지 다음부터 눌러주면 되므로 3으로 초기화
            count += 1

            # 이동 ~
            try :
                next_page.send_keys(Keys.ENTER)
            except:
                continue



    product *= len(rating)
    price *= len(rating)
    company *= len(rating)



    # 합치기
    df = pd.DataFrame({'상품명': product,
                       '가격': price,
                       '제조회사': company,
                       '별점': rating,
                       '리뷰': review})
    #####수정 필요#####

    name = 'coupang_result_' + str(a) +'.xlsx'
    # 파일 저장
    df.to_excel(name)
    # df.to_csv(name, sep = '\t')
    a += 1

