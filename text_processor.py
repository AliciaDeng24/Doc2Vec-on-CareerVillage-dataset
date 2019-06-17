from init import *
from nltk.corpus import stopwords

class text_prossessor():
    '''
    Preprocess text body (A string)

    '''

    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def prossessor(self, txt_str):
        '''
        Input: an non-prossessed string
        Output: string being stemmed and with stopwords, html links, and tags removed
        '''

        result = []

        # split and cast to lower case
        text = re.sub(r'<[^>]+>', ' ', str(txt_str))
        text = re.sub(r'(html|org|dr|www|com)', '', text)

        for word in re.split('[^a-zA-Z]', str(text).lower()):
            # remove non-alphabetic and stop words
            if word.isalpha() and word not in self.stopwords:
                if word not in result:
                    new_word = self.stemmer.stem(word)
                # use stemmed version of word
                result.append(new_word)
        return ' '.join(result)

#         # Remove tags at the end
#         new_txt = re.sub(r'\s*(\r|\n|#|<p>|</p>|<li>|<em>|<ol>|</li>|</ol>|</em>|\(|\))+(\w|\s)*', '', str(txt_str))

#         # Remove all website links and punctuations
#         new_txt = re.sub(r'(html|org|dr|www|com)', '', new_txt)
#         new_txt = re.sub(r'(<|>)?https?:?//(www\.)?\w+\/?./?\w*/?\./?\w*', '', new_txt)
#         new_txt = re.sub(r"(\'|\.|\/|\,|\?|\&|\:|\-)", ' ', new_txt).split()

#         # Remove all stopwords, stemming
#         for word in new_txt:
#             if word.lower() not in self.stopwords:
#                 if word.lower() not in result:
#                     # new_word = self.stemmer.stem(word.lower())
#                     result.append(word.lower())

#         return ' '.join(result)
