from init import *

# Read in all dataset
emails = pd.read_csv('emails.csv')
questions = pd.read_csv('questions.csv')
question_scores = pd.read_csv('question_scores.csv')
answers = pd.read_csv('answers.csv')
answer_scores = pd.read_csv('answer_scores.csv')
comments = pd.read_csv('comments.csv')
group_mem = pd.read_csv('group_memberships.csv')
groups = pd.read_csv('groups.csv')
tags = pd.read_csv('tags.csv')
tag_users = pd.read_csv('tag_users.csv')
tag_questions = pd.read_csv('tag_questions.csv')
matches = pd.read_csv('matches.csv')
professionals = pd.read_csv('professionals.csv')
students = pd.read_csv('students.csv')
school_mem = pd.read_csv('school_memberships.csv')

# Feature engineering

# A. Professional Dataset
# All users (professsionals + users) who are eligible for answering questions
num_profes = professionals.shape[0]
original_profes_id = professionals['professionals_id'].unique().tolist()
answered_user_id = answers['answers_author_id'].unique().tolist()
intersection_prof = list(set(original_profes_id) & set(answered_user_id))
users_notprof_cananswer = pd.Series(list(set(answered_user_id) - set(intersection_prof)))
len(users_notprof_cananswer)

# Construct new professional dataset
users_notprof_df = pd.DataFrame(data = users_notprof_cananswer, columns = ['professionals_id'])
users_notprof_df['professionals_location'] = [np.nan] * len(users_notprof_cananswer)
users_notprof_df['professionals_industry'] = [np.nan] * len(users_notprof_cananswer)
users_notprof_df['professionals_headline'] = [np.nan] * len(users_notprof_cananswer)
users_notprof_df['professionals_date_joined'] = [np.nan] * len(users_notprof_cananswer)

all_professionals = pd.concat([professionals, users_notprof_df], axis=0)
print(all_professionals.shape)

# Num of questions each professional has answered & Num of comments each professional has professionals_date_joined
answer_user_count = answers['answers_author_id'].value_counts()
answer_user_count = pd.DataFrame(answer_user_count)
answer_user_count['professionals_id'] = answer_user_count.index
answer_user_count.columns = ['how_many_questions_answered','professionals_id']
answer_user_count.reset_index(inplace=True)
answer_user_count.drop('index',axis = 1, inplace=True)
all_professionals = all_professionals.merge(answer_user_count, on = 'professionals_id', how = 'left')

comment_count = pd.DataFrame(comments['comments_author_id'].value_counts()).reset_index()
comment_count.columns = ['professionals_id', 'comment_count']
all_professionals = all_professionals.merge(comment_count, on='professionals_id', how = 'left')
all_professionals[['how_many_questions_answered', 'comment_count']] = all_professionals[['how_many_questions_answered', 'comment_count']].fillna(0)


# Concatenate tags with users and questions
tag_users.columns = ['tags_tag_id', 'tag_users_user_id']
all_tags_users = tag_users.merge(tags, on = 'tags_tag_id')
all_tags_users.sort_values(by='tag_users_user_id').head()
user_tags_df = all_tags_users.groupby(by='tag_users_user_id', as_index=False).aggregate(lambda x: ' '.join(x))

# Attach tags to all professionals
all_professionals = all_professionals.merge(user_tags_df,left_on='professionals_id',right_on='tag_users_user_id',how='left')
all_professionals.drop(columns=['tag_users_user_id'],inplace=True)

# Time since professionals joined & Date of last reply by each professional
all_professionals['professionals_date_joined'] = pd.to_datetime(all_professionals['professionals_date_joined'])
all_professionals['length_since_joined'] = datetime(2019,4,1) - all_professionals['professionals_date_joined']
all_professionals['length_since_joined'] = all_professionals['length_since_joined'].apply(lambda x: x.days)

last_reply = pd.DataFrame(answers.groupby(by='answers_author_id').answers_date_added.max())
all_professionals = all_professionals.merge(last_reply, left_on='professionals_id', right_on='answers_author_id', how='left')
all_professionals['answers_date_added'] = pd.to_datetime(all_professionals['answers_date_added'])
all_professionals['date_since_last_reply'] = datetime(2019,4,1) - all_professionals['answers_date_added']
all_professionals['date_since_last_reply'] = all_professionals['date_since_last_reply'].apply(lambda x: np.nan if pd.isnull(x) else x.days)


# B. Questions dataset
questions = questions.merge(question_scores.rename({'id': 'questions_id'}, axis=1), on='questions_id', how='left')

# Append tags for each question
all_tags = tag_questions.merge(tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')
all_tags_by_question = pd.DataFrame(all_tags.groupby(by = 'tag_questions_question_id').apply(lambda x: ','.join(x)))
all_tags_by_question.columns = ['questions_id', 'tags_tag_name']
questions = questions.merge(all_tags_by_question, on = 'questions_id', how = 'left')

# How many answers each question has
answer_counts = pd.DataFrame(answers.groupby(by='answers_question_id').answers_id.count()).reset_index()
answer_counts.head()

# Time when first answer of each question was posted
# Time delta - the length for each question to receive an answer from professional
answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'])
questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'])
answer_first_answer = pd.DataFrame(answers.groupby(by='answers_question_id').min()).reset_index()
answer_first_answer.drop(columns='answers_id', inplace=True)
answer_first_answer.columns = ['questions_id','first_author','first_answer_added_date']
questions = questions.merge(answer_first_answer, on = 'questions_id',how='left')

questions['time_delta'] = questions['first_answer_added_date'] - questions['questions_date_added']
questions['time_delta'] = questions['time_delta'].apply(lambda x: x.days)

# Append (both question and professional who gave the first answer) authors' locations
questions_with_loc = questions.merge(students, left_on='questions_author_id', right_on='students_id', how='left')
questions_with_loc = questions_with_loc.merge(all_professionals, left_on='first_author', right_on='professionals_id', how='left')
questions_with_loc.drop(columns=['students_id', 'professionals_date_joined',
       'how_many_questions_answered', 'comment_count', 'tags_tag_name_y'], inplace=True)

# Append group information for question author (students)
group_info = groups.merge(group_mem, left_on='groups_id', right_on='group_memberships_group_id', how='inner')
questions_with_loc_group = questions_with_loc.merge(group_info, left_on='questions_author_id', right_on='group_memberships_user_id', how='left')
questions_with_loc_group.drop(columns=['group_memberships_group_id','professionals_id','groups_id','group_memberships_user_id'],inplace=True)

# Merge answers to question_scores
questions_answers = questions_with_loc_group.merge(answers, left_on='questions_id', right_on='answers_question_id')
questions_answers.drop(columns=['first_author', 'first_answer_added_date', 'answers_id', 'answers_question_id', 'answers_date_added_y'], inplace=True)
# Tags for professional and students
questions_final = questions_answers.merge(user_tags_df, left_on='questions_author_id', right_on='tag_users_user_id', how='left')
questions_final = questions_final.merge(user_tags_df, left_on='answers_author_id', right_on='tag_users_user_id', how='left')
questions_final.drop(columns=['tag_users_user_id_x', 'tag_users_user_id_y'], inplace=True)
questions_final.columns = ['questions_id', 'questions_author_id', 'questions_date_added',
       'questions_title', 'questions_body', 'score', 'question_tags','time_delta',
       'students_location', 'students_date_joined', 'professionals_location',
       'professionals_industry', 'professionals_headline', 'length_since_joined', 'answers_date_added',
       'date_since_last_reply', 'groups_group_type', 'answers_author_id',
       'answers_body', 'question_author_tags', 'answers_author_tags']

# Save final question and professional datasets into csv for furture use
questions_final.to_csv('questions_final.csv')
all_professionals.to_csv('professionals_final.csv')
