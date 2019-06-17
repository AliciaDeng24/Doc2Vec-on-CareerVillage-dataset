from init import *
from text_processor import text_prossessor
from data_processor import DataPrePro
from nltk.corpus import stopwords
from NN_Models import *

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


# Initiate data_processor
dp = DataPrePro()

# Define embedded dimensions
emb_dim_ques = 20
emb_dim_pro = 20

# Neural network layers dimensions
dims_list = [64, 64, 32, 32]

# Split 10% of the total dataset as test set
random.seed(100)
N = len(questions_final)
train_rows = random.sample(list(range(N)), round(0.9*N))
test_rows = list(set(list(range(N))) - set(train_rows))

train_dataset = questions_final.iloc[train_rows, :]
test_dataset = questions_final.iloc[test_rows, :]


# Train questions embedding vectors based on question titles, tags, and bodies
train_dataset['question_all'] = train_dataset['questions_body'] + ' ' + train_dataset['questions_title'] + ' ' + train_dataset['question_tags']
ques_dict, ques_d2v = train_doc2vec(train_dataset, 'questions_id', ['question_all'], dim=emb_dim_ques, epoch=15)

# Train professional embedding vectors based on their characteristics (tags followed, industry, headline)
prof_fea = ['professionals_industry', 'professionals_headline','tags_tag_name']
pro_dict, _ = train_doc2vec(all_professionals, 'professionals_id', prof_fea, dim=emb_dim_pro, epoch=15)

# Append scaled characteristics to professional embedded vectors (measurement of activity)
# Training set
ques_input, pro_input, ques_ids, pro_ids = dp.preprocess(pro_dict, ques_dict, professionals_final, train_dataset, emb_dim_ques, emb_dim_pro)

# A. Fit a neural network using pre-defined Function

# Note that this neural network takes embedded question vectors as inputs and embedded professional vectors as output
model_1 = nn_model(ques_input, pro_input, dims_list)
model_1.compile(optimizer=Adam(lr=0.001), loss='mean_absolute_error', metrics=['accuracy'])
model_1.fit(ques_input, pro_input, validation_split=0.1, epoch=20)

# Transform test_set with scaled characteristics
# Here, we use the pretrained Doc2Vec object to project embedded vectors for test questions
test_input_df, test_input_mtx = transform_test(test_dataset, ques_d2v, True)

# Predict using trained neural network
prediction_1 = model_1.predict(test_input_mtx)

# Recommender System
# For each question, choose 10 candidate professionals whose embedded vectors are closest to projected vector
recommended_prof = find_all_closest(pro_ids, pro_input, ques_ids, prediction)
