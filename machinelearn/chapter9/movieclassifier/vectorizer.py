from sklearn.feature_extraction.text import HashingVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(__file__)

# 停用词
stop = pickle.load(open(os.path.join(cur_dir,"pkl_objects",'stopwords.pkl'),'rb'))

def tokenizer(text): # 数据清洗
    text = re.sub("<[^>]*>","",text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())
    text = re.sub('[\W]+'," ",text.lower()+" ".join(emoticons).replace('-',''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# 分档分解器
vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)