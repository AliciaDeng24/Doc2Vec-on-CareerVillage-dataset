from init import *
from text_processor import text_prossessor

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
