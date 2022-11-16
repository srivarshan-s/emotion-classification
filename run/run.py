from preprocessing.contraction import clean_contractions
from preprocessing.misspelling import correct_spelling
from preprocessing.clean import *
from preprocessing.punctuations import *

import torch
import pickle
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class CustomModel(torch.nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.roberta = AutoModel.from_pretrained('roberta-base')
        self.fc = torch.nn.Linear(768, 7)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask=mask,
                                   token_type_ids=token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


class BertDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.max_len = max_len
        self.text = text
        self.tokenizer = tokenizer

    def __len__(self):
        return 1

    def __getitem__(self, index):
        text = self.text
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }


def preprocess_pipeline(text, punct_list, punct_dict):
    text = clean_text(text)
    text = clean_contractions(text)
    text = clean_special_chars(text, punct_list, punct_dict)
    text = correct_spelling(text)
    text = remove_space(text)
    return text


def predict(text):
    model = pickle.load(open("models/cust_bert_model.pkl", "rb"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    dataset = BertDataset(text, tokenizer, 200)
    data_loader = DataLoader(dataset, batch_size=1,
                             num_workers=4, shuffle=False, pin_memory=True)
    model = model.to(device)
    fin_outputs = []
    for _, data in enumerate(data_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        outputs = model(ids, mask, token_type_ids)
        fin_outputs.extend(torch.sigmoid(
            outputs).cpu().detach().numpy().tolist())
    fin_outputs = fin_outputs[0]
    pred = np.argmax(fin_outputs)
    labels = ['anger', 'disgust', 'fear',
              'joy', 'sadness', 'surprise', 'neutral']
    return labels[pred]


def main():
    print("Enter your string:", end=" ")
    user_string = str(input())
    user_string = preprocess_pipeline(user_string, punct, punct_mapping)
    print(user_string)
    pred = predict(user_string)
    print(pred)


if __name__ == "__main__":
    main()
