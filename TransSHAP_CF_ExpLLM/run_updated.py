import pandas as pd
from nltk.tokenize import TweetTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import torch


print("USING the ProsusAI/finbert_____\n")
# Bag of words
train_data = pd.read_csv("./models/English_tweet_label.csv")
train_data = list(train_data["Tweet text"])
tknzr = TweetTokenizer()

bag_of_words = set([xx for x in train_data for xx in tknzr.tokenize(x)])

from transformers import BertTokenizer, BertForSequenceClassification

class SCForShap(BertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None,):
        output = super().forward(input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, labels)
        return output[0]

###### ProsusAI/finbert" is a fine-tuned model trained on financial text data
pretrained_model = "ProsusAI/finbert"
tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
model = SCForShap.from_pretrained(pretrained_model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)




# Test texts
#t1 = "Why is it that 'fire sauce' isn't made with any real fire? Seems like false advertising." #neutral
t1 = "The company's revenue dropped significantly this quarter, raising concerns among investors." #neutral
#t2 = "Being offensive isn't illegal you idiot." #negative
t2 = "The company had too much debt and went bankrupt." #negative
t3 = "Loving the first days of summer! <3" #positive
#t3 = "Due to mounting losses, the firm announced widespread layoffs"

t4 ="The company's shares plummeted after reporting a massive quarterly loss."
t5 ="The product recall caused significant reputational damage and financial losses."
t4 = "I hate when people put lol when we are having a serious talk ."   #negative
t5 = "People are complaining about lack of talent. Takes a lot of talent to ignore objectification and degradation #MissAmerica" #neutral
t6 = "Shit has been way too fucked up, for way too long. What the fuck is happening" #negative
t7 = "@Lukc5SOS bc you're super awesome😉" #positive
t8 = "RT @JakeBoys: This show is amazing! @teenhoot family are insane 😍" #positive
t9 = "So kiss me and smile for me 😊💗 http://t.co/VsRs8KUmOP"
# t6 = "RT @deerminseok: YOU'RE SO BEAUTIFUL. I MISS YOU SO MUCH. http://t.co/VATdCVypqC" #positive
# t7 = "😄😃😄 love is in the air ❤️❤️❤️ http://t.co/y1RDP5EdwG" #positive

tt = "the company went bankrupt."
tt2 = " he hate cat"
t11= "company lost a lot of money this year."
t12="The stock price closed unchanged from the opening value."
t13="Market trends remained steady throughout the trading session."
t14="The product launch was a complete failure."
t15="Profits hit a record high this quarter."


texts = [t12, t13, t14]



import shap
import random
import numpy as np
import logging
import matplotlib.pyplot as plt
from explainers.SHAP_for_text import SHAPexplainer
from explainers import visualize_explanations
logging.getLogger("shap").setLevel(logging.WARNING)
shap.initjs()

words_dict = {0: None}
words_dict_reverse = {None: 0}
for h, hh in enumerate(bag_of_words):
    words_dict[h + 1] = hh
    words_dict_reverse[hh] = h + 1


predictor = SHAPexplainer(model, tokenizer, words_dict, words_dict_reverse, device)

train_dt = [predictor.split_string(x) for x in train_data]  

idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)### used to as the background data

explainer = shap.KernelExplainer(model=predictor.predict, data=shap.kmeans(idx_train_data, k=20)) #10

texts_ = [predictor.split_string(x) for x in texts][:3] 
idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)  


predictions_list = []

# Extract and visualize SHAP values
for ii in range(len(idx_texts)):
    print(f"START COMPUTE DATASET___{ii + 1}___\n")
        
    # Original text (for label prediction)
    original_text = texts[ii]

    # Directly tokenize the original text using BERT tokenizer
    f, pred_f = predictor.get_prediction_label(original_text)

    # Store the predicted class index in the predictions list
    predictions_list.append(pred_f)
    
    t = idx_texts[ii]

    perturbed_data = np.array(predictor.generate_purturb(t))

    shap_to_use = perturbed_data.reshape(1, -1)  
    #print(f"check the shap_to_use shape___________ {shap_to_use.shape}")   #(1, 97)
    #print(f"WITHN THE FOR _____IN RUN.PY__________check the perturbed_data shape {shap_to_use}")


    #print("NOW STARTING COMPUTE SHAP_VALUES_______\n")
    shap_values = explainer.shap_values(X=shap_to_use,  nsamples=1000) 
    #print(f"check the shap_values shape______________ {shap_values.shape}")  #check the shap_values shape (1, 97, 3) [Single input sample, number of tokens, Number of output classes]


    selected_shap_values = shap_values[0][:, pred_f]  
    #print(selected_shap_values)


    ###Visualize using the correct SHAP values
    visualize_explanations.joint_visualization(
        texts_[ii],
        selected_shap_values,
        ["Negative", "Neutral", "Positive"][int(pred_f)],
        f[0][pred_f],
        ii
    )


 





