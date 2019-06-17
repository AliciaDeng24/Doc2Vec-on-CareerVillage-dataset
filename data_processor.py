from init import *

# Create data preprocess class

class DataPrePro(object):
    '''
    Preprocessing given data (professional, questions) into doc2vec arrays form
    '''

    def __init__(self):
        '''
        '''

        # Scaler
        self.scaler = MinMaxScaler()

    def scale_series(self, series):
        '''
        Scale series using MinMixScaler
        '''

        scaled_array = self.scaler.fit_transform(series.values.reshape((-1,1)))
        ret = pd.Series(scaled_array.reshape(-1))

        return ret

    def preprocess(self, pro, ques, all_prof_df, all_ques_df, dim_1, dim_2):
        '''
        Merge pro with que, together with other generated features

        Return:
        ques_input - Numpy array of embedded vectors for question
        pro_input - Numpy array of corresponding embedded vectors for each
        professional who has answered this question
        ids - Pandas series of professional_ids in order of questions
        they've answered
        '''

        pro_col = {x: 'pro_array_{}'.format(x) for x in list(range(dim_2))}
        que_col = {y: 'que_array_{}'.format(y) for y in list(range(dim_1))}

        prof_df = pd.DataFrame(pro).T.rename(pro_col, axis='columns').reset_index()
        ques_df = pd.DataFrame(ques).T.rename(que_col, axis='columns').reset_index()

        # Form new pro dataframe
        new_pro_df = prof_df.merge(all_prof_df[['professionals_id','how_many_questions_answered', \
                                           'comment_count', 'length_since_joined', 'answers_date_added', \
                                           'date_since_last_reply']],
                                 left_on='index', right_on='professionals_id', how='right')
        new_pro_df.drop(columns=['index', 'answers_date_added'], inplace=True)
        new_pro_df = new_pro_df.fillna(0)

        # Form new ques dataframe
        new_que_df = ques_df.merge(all_ques_df[['questions_id', 'answers_author_id']],
                                   left_on='index', right_on='questions_id', how='left')
        new_que_df.drop(columns='index', inplace=True)
        new_que_df = new_que_df.drop_duplicates(subset='questions_id')

        # Scale length_since_joined and days_since_answers added

        col_to_scale = ['how_many_questions_answered','comment_count', 'length_since_joined', 'date_since_last_reply']
        for col in col_to_scale:
            new_pro_df[col] = self.scale_series(new_pro_df[col])

        # Merge two datasets together and form numpy arrays for NN
        new_df = new_que_df.merge(new_pro_df, left_on='answers_author_id', right_on='professionals_id', how='left')
        ids = new_df['answers_author_id']
        new_df.drop(columns=['answers_author_id','professionals_id', 'questions_id'], inplace=True)

        # Split dataframe into neural network inputs
        ques_input = new_df.iloc[:, 0:dim_1].values
        pro_input = new_df.iloc[:, dim_1:].values

        return ques_input, pro_input, ids
