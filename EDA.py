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

# All users (professsionals + users) who are eligible for answering questions
num_profes = professionals.shape[0]
original_profes_id = professionals['professionals_id'].unique().tolist()
answered_user_id = answers['answers_author_id'].unique().tolist()
intersection_prof = list(set(original_profes_id) & set(answered_user_id))
