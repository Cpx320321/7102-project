import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Load the pre-trained job and course matrices
job_matrix = np.load('program_matrix.npy')
course_matrix = np.load('program_matrix.npy')

# Load the job and course data as pandas dataframes
job_data = pd.read_csv(r'C:\Users\...\job_description.csv')
course_data = pd.read_csv(r'C:\Users\...\program.csv')

import torch
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embedding(text):
    # Truncate text to maximum length of 512 tokens
    if len(text)>512:
        text = text[:512]
    # Tokenize the text
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    # Convert the tokenized sequence to a PyTorch tensor
    input_tensor = torch.tensor([input_ids])
    # Generate the BERT embeddings
    with torch.no_grad():
        outputs = model(input_tensor)
        embeddings = outputs[0][0][1:-1]  # ignore the special tokens and take the embeddings of the original text only
    # Convert the embeddings to a 100-dimensional vector
    embeddings_100d = torch.mean(embeddings, dim=0)[:100].numpy()
    return embeddings_100d

def cosine_similarity(v1, v2):
    # Calculate the cosine similarity between two vectors
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Combine the 'program_overview' and 'course_description' columns into a single text column
courese_data1 = course_data['program_overview'] + course_data['course_description']
# Create a list of course names by combining the 'university' and 'program_name' columns
university = list(course_data['university'])
program_name = list(course_data['program_name'])
courese_name = [str(s1) + ' ' + str(s2) for s1, s2 in zip(university, program_name)]
courese_name = courese_name[0:90]  # Only keep the first 90 courses for this example

courese_name = list(courese_name)
courese_data1 = list(courese_data1)

nan_indices = []
for i, value in enumerate(courese_data1):
    # Record the indices of any NaN values in the course data
    if type(value) == float:
        nan_indices.append(i)

# Remove the corresponding elements from the course names and descriptions lists
for index in sorted(nan_indices, reverse=True):
    del courese_name[index]

for index in sorted(nan_indices, reverse=True):
    del courese_data1[index]

# Separate the job data into two categories: engineering and social work jobs
Engineering=job_data[job_data['Category'] == 'Engineering Jobs']
Social = job_data[job_data['Category'] == 'Social work Jobs']
Engineering_Description = list(Engineering['FullDescription'])
Social_Description = list(Social['FullDescription'])
Description = Engineering_Description + Social_Description
Engineering_Title = list(Engineering['Title'])
Social_Title = list(Social['Title'])
Title = Engineering_Title + Social_Title

sim= []
tit = []
for course in courese_data1:
    s_list = []
    course = str(course)
    # Convert the course string to a tensor using the embedding function
    course_tensor = embedding(course)
    # Calculate the cosine similarity between the course tensor and each job vector
    for i in range(job_matrix.shape[0]):
        s_list.append(cosine_similarity(job_matrix[i], course_tensor))

    # Find the top 5 elements and their indices in s_list
    max_list = sorted(enumerate(s_list), key=lambda x: x[1], reverse=True)[:6]
    max_value = [x[1] for x in max_list]
    max_index = [x[0] - 1 for x in max_list]
    similarity = []
    title = []
    #description = []
    # Add the top 5 job titles and similarities to the title and similarity lists
    for i in range(5):
        similarity.append(max_value[i+1])
        title.append(Title[max_index[i+1]])
    sim.append(similarity)
    tit.append(title)

# Calculate the mean similarity for each course
mean1=np.sum(np.array(sim), axis=1)/5
df = pd.DataFrame()
#df[courese_name] = courese_name
# Loop through the top 5 jobs and add them to the DataFrame
for i in range(5):
    if i == 0:
        df['input'] = courese_data1
    job_name = f'course {i+1}'
    simlarity = f'simlarity {i+1}'
    df[job_name] = np.array(tit)[:,i]
    df[simlarity] = np.array(sim)[:,i]
    if i == 4:
        df['mean similarity'] = mean1

# Save the results to an Excel file
df.to_excel('course_to_job.xlsx', index=False)
