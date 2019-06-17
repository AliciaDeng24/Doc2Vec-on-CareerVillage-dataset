# Define a function that transform the test dataset into desired format

def transform_test(test_set, pre_trained_d2v, use_infer=True):
    '''
    Compute question vectors based on indicator use_infer.
    If True, then question vectors in test set will be inferred from pre-trained doc2vec object
    If False, then train a separate doc2vec on test questions

    return:
    array of size (len(test_set), 50) as input array of neural network model
    '''

    # Concatenate whole text
    test_set['question_all'] = test_set['questions_body'] + ' ' + test_set['questions_title'] + ' ' + test_set['question_tags']
    test_set['question_all'].apply(tp.prossessor)

    if use_infer:
        # Initiate transformed dictionary
        test_dict = dict()
        for idx, row in test_set[['questions_id', 'question_all']].drop_duplicates(subset='questions_id').iterrows():
            test_dict[row['questions_id']] = pre_trained_d2v.infer_vector(row['question_all'].split())

    else:
        # Use test_set to train a new doc2vec object
        test_dict, _ = train_doc2vec(test_set, 'questions_id', ['question_all'], dim=50, epoch=30)

    test_input_1 = pd.DataFrame(test_dict).T
    test_input_2 = test_input_1.values

    return test_input_1, test_input_2

# Define a function that pinpoints the 'k-most-similar-professionals' via cosine similarity

def find_closest(pro_ids, pro_trained, nn_vector, k=avg_num_ans):
    '''
    Find k closest (most similar) professionals based on neural network output vectors and doc2vec embeddings

    pro_ids: professionals ids
    pro_trained: pre-trained doc2vec embedding vectors
    nn_vector: predicted neural network vector
    k : num of neighbours

    return:
    list of indices indicating professionals
    '''

    dictionary = dict(zip(pro_ids, pro_trained))
    nearest_indice = dict()

    for key, vec in dictionary.items():
        # Reshape into 2D arrays and find cosine similarity
        cos_sim = cosine_similarity(vec.reshape(-1,1).T, nn_vector.reshape(-1,1).T)
        nearest_indice[key] = cos_sim

    # Sort dictionary based on values
    sorted_dict = sorted(nearest_indice.items(), key=lambda kv: kv[1], reverse=True)
    nearest_k = [tup[0] for tup in sorted_dict[0:k]]

    # This list contains the closest k professionals embedding vectors given the predicted ques embedding
    return nearest_k

def find_all_closest(pro_ids, pro_trained, questions_ids, nn_outputs):
    total = nn_outputs.shape[0]

    recommending_dict = {}

    for pos in range(total):
        nearest = find_closest(pro_ids, pro_trained, nn_outputs[pos])
        recommending_dict[questions_ids[pos]] = nearest

    return recommending_dict
