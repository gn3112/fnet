from selenium import webdriver
from bs4 import BeautifulSoup

wd = webdriver.Chrome("/Users/georgesnomicos/chromedriver")

wd.get("https://www.carrefour.fr/r/fruits-et-legumes/fruits")

button = wd.find_element_by_class_name("a-button.is-secondary")
button.click()

html = wd.page_source

soup = BeautifulSoup(html, "html.parser")

print(soup.text)

wd.quit()