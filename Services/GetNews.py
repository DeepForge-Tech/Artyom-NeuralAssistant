import requests
from bs4 import BeautifulSoup
from loguru import logger
import json
import os

class ParseNews:
    def __init__(self,ProjectDir) -> None:
        self.NewsURL = "https://lenta.ru/parts/news"
        self.itNewsURL = "https://3dnews.ru/news"
        self.NewsArray = []
        self.itNewsArray = []
        self.ProjectDir = ProjectDir

    def News(self):
        r = requests.get(self.NewsURL).text
        soup = BeautifulSoup(r, 'html.parser')
        content = soup.find_all("h3",class_ = 'card-full-news__title')
        for span in content:
            text = span.text.replace('"','')
            # text = await self.FilteringTransforms(text,to_words=True)
            self.NewsArray.append(text)
        file = open(os.path.join(self.ProjectDir,"AssistantSettings/News.json"),"w",encoding="utf-8")
        json.dump(self.NewsArray,file,indent=2,ensure_ascii=False,sort_keys=True)
        file.close()

    def it_news(self):
        r = requests.get(self.itNewsURL).text
        soup = BeautifulSoup(r, 'html.parser')
        content = soup.find_all("a",class_='entry-header')
        for span in content:
            text = span.text.replace('"','')
            # text = await self.FilteringTransforms(text,to_words=True)
            self.itNewsArray.append(text)
        file = open(os.path.join(self.ProjectDir,"AssistantSettings/IT_News.json"),"w",encoding="utf-8")
        json.dump(self.itNewsArray,file,ensure_ascii=False,sort_keys=True, indent=2)
        file.close()

    def StartParse(self):
        self.News()
        self.it_news()

if __name__ == "__main__":
    Parse = ParseNews(os.path.dirname(os.path.realpath(__file__)))
    Parse.News()
    # Parse.StartParse()