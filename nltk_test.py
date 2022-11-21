import nltk
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd
import string

#print(brown.categories())
#print(brown.fileids())

#nltk.download('omw-1.4')
#nltk.download('averaged_perceptron_tagger')
#情态动词列表，可以用于分析不同问题中用词的频率差异
modals= ['can','could','may','might','must','will']
data = []
for corpus in brown.categories():
    fdist = nltk.FreqDist([w.lower() for w in brown.words(categories=corpus)])
    for m in modals:
        data.append([corpus, m , fdist[m]])
        #print(m,':',fdist[m])

#借用其他工具可以生成表格便于分析
df = pd.DataFrame(data, columns=['categories', 'modals', 'value'])
res = df.pivot(index='categories', columns='modals', values='value')
print(res)
'''
结果为
['how are you going today?', 
'The weather is good.', 
'Do you want to go out with me?']
'''
#对段落进行分析
paragraph = "how are you going today? The weather is good. Do you want to go out with me?"
tokenized_text = nltk.tokenize.sent_tokenize(paragraph)
print(tokenized_text)#分句
#去掉句子中的标点符号，并替换为空格
for paragraph in tokenized_text:
    newsent = paragraph.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    print(newsent)

#直接对段落进行处理,去掉停用词以及标点符号
#nltk.download('stopwords')
stop_words = stopwords.words("english")
word_tokens = nltk.tokenize.word_tokenize(paragraph.strip())
filtered_word = [w for w in word_tokens if not w in stop_words]
#print(stop_words)
print("word_tokens:" , word_tokens)
print("filtered_word:" , filtered_word)

# 词汇规范化
lem = WordNetLemmatizer() # 词性还原
stem = PorterStemmer() # 提取词干
word = 'stationary'
lemmatized_word = lem.lemmatize(word)
stemmed_word = stem.stem(word)#此处没有输出想象中的结果
print("Lemmatized word is", lemmatized_word)
print("Stemmed word is", stemmed_word)

sent = "Albert Einstein was born in Ulm, Germany in 1879."
tokens = nltk.word_tokenize(sent)
tags = nltk.pos_tag(tokens)
print(tags)

