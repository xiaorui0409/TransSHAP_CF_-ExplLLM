import numpy as np
import torch
from nltk.tokenize import TweetTokenizer
import logging
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)


class SHAPexplainer:

    ##### Purpose of the  "words_dict_reverse" ; "Words_dict_reverse"
    def __init__(self, model, tokenizer, words_dict, words_dict_reverse, device):
        self.model = model
        self.tokenizer = tokenizer
        self.tweet_tokenizer = TweetTokenizer()
        self.words_dict = words_dict
        self.words_dict_reverse = words_dict_reverse
        self.device = device
        self.word_idx = None  


    ############################## Updated predict Function####################################
    def predict(self, indexed_words):
        # self.model.to(self.device)
        #print(f"____PREDICT_FUNCTION____FIRST time_______CHECK THE indexed_words____{indexed_words}")  ###2D array
        #print(f"____PREDICT_FUNCTION____FIRST time_______CHECK THE indexed_words shape____{indexed_words.shape}")  # ##(10, 97)
       
        sentences = [[self.words_dict[xx] if xx != 0 else "" for xx in x] for x in indexed_words]
        #print(f"____PREDICT_FUNCTION____FIRST time_______sentences____{sentences}")

        # Flatten sentences to process individual strings
        flat_sentences = [" ".join(sentence) for sentence in sentences]
       
        ###Tokenize text into word-level tokens using the TweetTokenizer.
        texts_convert = [self.split_string(x) for x in flat_sentences]

        #### Convert text_tokens into numerical representation using dt_to_idx
        perturb_convert_texts, _ = self.dt_to_idx(texts_convert, max_seq_len=None)

        Bert_indexed_tokens = np.array([self.generate_purturb(x) for x in perturb_convert_texts])  #[batch_number,   max_lenght]

        indexed_tokens = Bert_indexed_tokens  ####Bert_indexed_tokens:list
        tokens_tensor = torch.tensor(indexed_tokens).to(self.device)
        #print(f"____PREDICT_FUNCTION____check the indexed_token before fed into the Bertmodel___{tokens_tensor.shape}")  #(10, 512)

        attention_mask = (tokens_tensor!=0).to(self.device)

        ####step4: compute output
        with torch.no_grad():
            outputs = self.model(input_ids=tokens_tensor, attention_mask=attention_mask)
            logits = outputs
            #print(f"Logits: {logits}")
            perturbed_predictions = logits.detach().cpu().numpy()
        
        final =[self.softmax(x) for x in perturbed_predictions]
        #print("*"*100)
        #print("Final shape:__", np.array(final).shape)

        return np.array(final)


    ###### Visulzation purpose
    def get_prediction_label(self, original_text):

        #print(f"____get_prediction_label_______CHECK THE indexed_words____{original_text}")  ###2D array  (1, 97)
        #print(f"____PREDICT_original____FIRST time_______CHECK THE indexed_words shape____{original_text.shape}")  # ##_torch.Size([1, 97])
    
        # Directly tokenize the original text using BERT tokenizer
        inputs = self.tokenizer(original_text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        
        # Get prediction (label) from original text
        with torch.no_grad():
            outputs = self.model(**inputs)

        f = outputs.cpu().numpy()  # shape (1, 3) if num_labels=3
        f1 = self.softmax(f)
        #print("Full output of f:", f1)
        pred_f = np.argmax(f1[0])   # predicted class
        #print("pred_f:____",pred_f)  ## int_number
        #print("get_prediction_label_Final shape:__", type(pred_f))
        #raise SystemExit("Breaking execution after stop check")
    
        return f1, pred_f



    def softmax(self, it):
        exps = np.exp(np.array(it))
        return exps / np.sum(exps)

    def split_string(self, string):
        data_raw = self.tweet_tokenizer.tokenize(string)
        data_raw = [x for x in data_raw if x not in ".,:;'"]
        return data_raw

    def tknz_to_idx(self, train_data, MAX_SEQ_LEN=None):
        tokenized_nopad = [self.tokenizer.tokenize(" ".join(text)) for text in train_data]
        if not MAX_SEQ_LEN:
            MAX_SEQ_LEN = min(max(len(x) for x in train_data), 512)
        tokenized_text = [['[PAD]', ] * MAX_SEQ_LEN for _ in range(len(tokenized_nopad))]
        for i in range(len(tokenized_nopad)):
            tokenized_text[i][0:len(tokenized_nopad[i])] = tokenized_nopad[i][0:MAX_SEQ_LEN]
        indexed_tokens = np.array([np.array(self.tokenizer.convert_tokens_to_ids(tt)) for tt in tokenized_text])
        return indexed_tokens, tokenized_text, MAX_SEQ_LEN

    def dt_to_idx(self, data, max_seq_len=None):
        idx_dt = [[self.words_dict_reverse[xx] for xx in x] for x in data]
        if not max_seq_len:
            max_seq_len = min(max(len(x) for x in idx_dt), 512)
        for i, x in enumerate(idx_dt):
            if len(x) < max_seq_len:
                idx_dt[i] = x + [0] * (max_seq_len - len(x))
        return np.array(idx_dt), max_seq_len
    

    def generate_purturb(self, indexed_words):
        #print(f"__GENERATE_PURTURB__Original indexed words: {indexed_words}")
        #print(f"____GENERATE_PURTURB____FIRST time_______CHECK THE indexed_words shape____{indexed_words.shape}")  # ##(97,)

        perturbed_sentences = []
    

        for self.word_idx, token_id in enumerate(indexed_words):  # Track the current word index
        
            if token_id == 0:
                perturbed_sentences.append("")  # Keep padding
            elif np.random.rand() < 0.3:  # Replace 30% of tokens
                perturbed_sentences.append("[MASK]")  # Use BERT's [MASK] token
            else:
                perturbed_sentences.append(self.words_dict[token_id])  # Original token
        

        #print(f"__GENERATE_PURTURB__Perturbed sentences: {perturbed_sentences}") 
        #print(f"__GENERATE_PURTURB__Perturbed sentences type: {type(perturbed_sentences)}")##list


        # Join perturbed words into a single sentence
        perturbed_sentences = " ".join(perturbed_sentences).strip()
        #print(f"__GENERATE_PURTURB__NEW___Perturbed sentences___NEW______: {perturbed_sentences}") 

        MAX_SEQ_LEN = 128
       
        Bert_indexed_tokens, _, _ = self.tknz_to_idx([perturbed_sentences], MAX_SEQ_LEN=MAX_SEQ_LEN)

        return np.array(Bert_indexed_tokens[0])  












    
 
