from bs4 import BeautifulSoup
import requests
import io
from PIL import Image
import os

main_categories = ["fruits-et-legumes"]
sub_categories = {"fruits-et-legumes":["fruits"]
        }

class WebItems():
    def __init__(self, URL):
        self.URL = URL

    def search_by_category(self, main_category, sub_category, image_resolution=580):
        page = requests.get(self.URL+'r/'+main_category+'/'+sub_category)
        soup = BeautifulSoup(page.content, "html.parser")
        all_items = soup.find_all('article')
        
        all_text = []
        all_img = []
        for item in all_items:
            text = item.find_all('h2')[0].text.strip()
            all_text.append(text)

            img_pil = self._get_pil_image_from_article_tag(item, text, image_resolution)
            all_img.append(img_pil)

        return all_text, all_img

    def search_by_name(self, name):
        URL = self.URL + "s?q=" + name.replace(' ', '+')
        page = requests.get(URL)
        soup = BeautifulSoup(page.content, "html.parser")
        first_item = soup.find('article')
        text = first_item.find('h2').text.strip()

        img_pil = self._get_pil_image_from_article_tag(first_item, text, 580)

        return text, img_pil

    def _get_pil_image_from_article_tag(self, article_tree, text, image_resolution):
        imgs_with_tag = article_tree.find_all("img")

        for img in imgs_with_tag:
            if img['alt'].strip() == text:
                break
        
        img280_url = self.URL+img["src"]
        img1500_url = img280_url.replace("280x280","1500x1500")

        img_bytes = requests.get(img1500_url).content

        return Image.open(io.BytesIO(img_bytes))

if __name__ == "__main__":
    URL = "https://www.carrefour.fr/"
    carrefour = WebItems(URL)
    
    SAVE_PATH = '/Users/georgesnomicos/fridge_net/industrial_items'

    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)

    items_trial = ["Beurre tendre doux ELLE VIRE", "Fromages frais bio nature CARREFOUR BIO", 
    "Fromage à tartiner nature PHILADELPHIA", "Mozzarella GALBANI", "Lait bio entier LACTEL", 
    "beurre president", "yop fraise", "Chocolat au lait et céréales CRUNCH", "Chiffonnade de jambon de parme NEGRONI",
    "Compotes pomme poire ANDROS"]

    for name in items_trial:
        text, img_pil = carrefour.search_by_name(name)
        img_pil.save(os.path.join(SAVE_PATH,text.replace(' ',"_"))+'.jpg')
