from init import *
from text_processor import text_prossessor
from data_processor import DataPrePro
from nltk.corpus import stopwords

professionals_final = pd.read_csv('professionals_final.csv')
questions_final = pd.read_csv('questions_final.csv')

# A. Remove all duplicate tags in profession and question datasets
split_fun = lambda x: x if pd.isna(x) else ','.join(set(x.split(',')))
professionals_final['tags_tag_name'] = professionals_final['tags_tag_name'].apply(split_fun)
questions_final['question_author_tags'] = questions_final['question_author_tags'].apply(split_fun)
questions_final['answers_author_tags'] = questions_final['answers_author_tags'].apply(split_fun)
questions_final['question_tags'] = questions_final['question_tags'].apply(split_fun)

# Initiate text_prossessor
tp = text_prossessor()

col_ques = ['professionals_industry', 'professionals_headline', 'questions_body',\
            'questions_title', 'answers_body', 'question_tags', 'question_author_tags',\
            'answers_author_tags']
for col in col_ques:
    questions_final[col] = questions_final[col].apply(tp.prossessor)

col_prof = ['professionals_industry', 'professionals_headline', 'tags_tag_name']
for col in col_prof:
    professionals_final[col] = professionals_final[col].apply(tp.prossessor)


# Define a function using doc2vec to train word embeddings
def train_doc2vec(df, tgt_feature, feature_lists, dim, epoch):
    '''
    Create tagged documents using tgt_feature as tags along with feature_lists from df.
    df - dataframe containing text data
    tgt_feature - name of column
    feature_lists - list of strings containing names of all text columns
    dim - dimension of the doc2vec output
    epoch - a hyperparameter used in Doc2Vec

    return:
    d2v_dict: a dict whose keys are input texts and values are corresponding embedding vectors of dimension dim
    doc2vec: Dov2Vec object
    '''

    tagged_doc = []
    for fea in feature_lists:
        if fea != tgt_feature:
            tagged_doc += [TaggedDocument(data[fea].split(), [data[tgt_feature]]) for i, data in df[[tgt_feature, fea]].drop_duplicates().iterrows()]
        else: # when fea is indeed tgt_feature
            tagged_doc += [TaggedDocument(data.split(), [data]) for data in df[fea].drop_duplicates()]

    doc2vec = Doc2Vec(tagged_doc, vector_size=dim, dm=0, min_count=1, epoches=epoch, worker=2)
    d2v_dict = {doc2vec.docvecs.index2entity[i]: doc2vec.docvecs.vectors_docs[i] for i in range(len(doc2vec.docvecs.index2entity))}
    return d2v_dict, doc2vec

# Word Embeddings visualization - Used t-SNE for dimension reduction
# Word Embeddings on tags & industries via t-SNE

def tsne_scatterplot(emb_dict, annotations, amt, plt_title):
    '''
    Reduce the embedded vectors to a lower dimension for visualization

    emb_dict: a dictionary whose {key: value} pairs are {texts: embedded vectors}
    annotations: pd.series of tag names associated to texts data
    amt: number of items to plot
    plt_title: a string

    return:
    a scatterplot with tag names as annotations
    '''

    most_freq = set(annotations.value_counts()[:amt].index)
    emb_vectors = {item: emb_dict[item] for item in most_freq}

    stacked_tags = np.vstack(emb_vectors.keys())
    stacked_vectors = np.vstack(emb_vectors.values())

    tsne_vectors = TSNE(n_components=2).fit_transform(stacked_vectors)

    X = tsne_vectors[:, 0]
    Y = tsne_vectors[:, 1]

    _, ax = plt.subplots(figsize=(16,16))
    plt.scatter(X, Y, alpha=0.7)
    ax.set_title(plt_title)

    for i, name in enumerate(stacked_tags):
        ax.annotate(name[0], (X[i], Y[i]), )

    plt.show()

# Train word embeddings on questions' tags and professionals' industries

target_fea = 'question_tags'
features = ['questions_body', 'answers_body', 'answers_author_tags', 'question_author_tags', 'question_tags', 'professionals_industry', 'professionals_headline']
pro_tags_dict, professional_tags_emb = train_doc2vec(train_dataset, target_fea, features, dim=15, epoch=10)
industry_dic, industry_tag_emb = train_doc2vec(train_dataset, 'professionals_industry', features, dim=15, epoch=10)

# visualizations (Will be presented in a separate notebook file)
tsne_scatterplot(pro_tags_dict, train_dataset.question_tags, 100, "t-SNE on professionals' tags")
tsne_scatterplot(industry_dic, train_dataset.professionals_industry, 100, "t-SNE on industries")



# Train questions embedding vectors based on question titles, tags, and bodies

# Train professional embedding vectors based on their characteristics (tags followed, industry, headline)
