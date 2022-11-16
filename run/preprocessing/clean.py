import re
import emoji
from bs4 import BeautifulSoup


def clean_text(text):
    text = emoji.demojize(text) # Remove emojis
    text = re.sub(r'\:(.*?)\:','',text) 
    text = str(text).lower() # Make text lowercase
    text = re.sub('\[.*?\]', '', text) # Remove text inside square brackets
    text = BeautifulSoup(text, 'lxml').get_text() # Remove html
    text = re.sub('https?://\S+|www\.\S+', '', text) # Remove links
    text = re.sub('<.*?>+', '', text) # Remove text inside angular brackets
    text = re.sub('\n', '', text) # Remove newlines
    text = re.sub('\w*\d\w*', '', text) # Remove words that contain numbers
    text = re.sub(r"[^a-zA-Z?.!,¿']+", " ", text)
    return text


def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, f' {p} ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''} 
    for s in specials:
        text = text.replace(s, specials[s])
    return text


def remove_space(text):
    text = text.strip()
    text = text.split()
    return " ".join(text)

