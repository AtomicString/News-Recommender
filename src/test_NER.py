import numpy as np
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer
from functools import partial


def test_func(model, tokenizer, tag_values, test_sentence):
    tokenized_sentence = tokenizer.encode(test_sentence)
    input_ids = torch.tensor([tokenized_sentence]).cuda()
    ner_tokens = []
    ner = []

    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    # join bpe split tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag_values[label_idx])
            new_tokens.append(token)

    for token, label in zip(new_tokens, new_labels):
        # print("{}\t{}".format(label, token))
        if label in ['B-LOC', 'B-MISC', 'B-ORG', 'I-LOC', 'I-MISC', 'I-ORG', 'I-PER']:
            ner_tokens.append(label)
            ner.append(token)

    return ner_tokens, ner


data = pd.read_csv("data/eng_train.txt",
                   encoding="utf-8", sep=" ", header=None, skip_blank_lines=False)
data.columns = ["Word", "POS", "IOB tags", "Tag"]
tag_values = list(set(data["Tag"].values))
tag_values.append("PAD")


tokenizer = BertTokenizer.from_pretrained(
    'bert-base-cased', do_lower_case=False)

model = pickle.load(open('models/model.pickle', 'rb'))

get_test_BIO = partial(test_func, model=model,
                       tokenizer=tokenizer, tag_values=tag_values)

if __name__ == "__main__":
    print(get_test_BIO(
        test_sentence="Elections Live: Nitin Gadkari, Sharad Pawar cast vote in Maharashtra"))
