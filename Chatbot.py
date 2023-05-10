# Import necessary libraries
import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
import numpy as np
import warnings
import torch
warnings.filterwarnings('ignore')

# Load pre-trained BERT model and tokenizer
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


# Define function to compute BERT embeddings
def embedding(text):
    if len(text)>512: # limit text to 512 tokens to fit within BERT's input size
        text = text[:512]
    input_ids = tokenizer.encode(text, add_special_tokens=True) # encode text as BERT tokens
    input_tensor = torch.tensor([input_ids]) # convert tokenized sequence to PyTorch tensor
    with torch.no_grad():
        outputs = model(input_tensor) # generate BERT embeddings
        embeddings = outputs[0][0][1:-1]  # ignore special tokens and take embeddings of original text only
    embeddings_100d = torch.mean(embeddings, dim=0)[:100].numpy() # convert embeddings to a 100-dimensional vector
    return embeddings_100d

# Define function to compute cosine similarity between two vectors
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Define function to compute top 5 most similar job titles to a given course
def similarity(course):
    s_list = []
    course_tensor = embedding(course)
    for i in range(Description_matrix.shape[0]):
        s_list.append(cosine_similarity(Description_matrix[i], course_tensor))

    # find the position of the top 5 values
    max_list = sorted(enumerate(s_list), key=lambda x: x[1], reverse=True)[:5]
    max_value = [x[1] for x in max_list]
    max_index = [x[0] - 1 for x in max_list]
    similarity = []
    title = []
    for i in range(5):
        similarity.append(max_value[i])
        title.append(Title[max_index[i]])
    return similarity,title

# Define function to display chatbot response
def send_message():
    message = input_box.get()
    similarity1,title = similarity(message)
    response = str(similarity1)
    title = str(title)
    chat_box.configure(state='normal')
    chat_box.insert(tk.END, "course: " + message + "\n")
    chat_box.insert(tk.END, "similarity: " + response + "\n")
    chat_box.insert(tk.END, "job: " + title + "\n")
    chat_box.configure(state='disabled')
    input_box.delete(0, tk.END)


# main function
if __name__ == "__main__":
    # ask user what they want to predict: job or course
    choose = input('what do you want to predict? job or course:')

    # if user chooses job
    if choose == 'job':

        # Load the pre-computed job description embeddings
        Description_matrix = np.load('job_matrix.npy')
        # read job description data from csv file
        df = pd.read_csv(
            r'C:\Users\不知道叫什么\Desktop\HKU sme2\ARIN7102 Applied data mining and text analytics\project/job_description.csv')

        # filter job descriptions by category (engineering and social work)
        Engineering = df[df['Category'] == 'Engineering Jobs']
        Social = df[df['Category'] == 'Social work Jobs']

        # extract full descriptions and titles of job postings
        Engineering_Description = list(Engineering['FullDescription'])
        Social_Description = list(Social['FullDescription'])
        Description = Engineering_Description + Social_Description
        Engineering_Title = list(Engineering['Title'])
        Social_Title = list(Social['Title'])
        Title = Engineering_Title + Social_Title

        # create tkinter window
        window = tk.Tk()
        window.title("Chatbot")
        window.geometry("500x500")

        # create scrolledtext widget to display chat history
        chat_box = scrolledtext.ScrolledText(window, state='disabled')
        chat_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # create entry widget for user input
        input_box = tk.Entry(window)
        input_box.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        input_box.bind("<Return>", lambda event: send_message())

        # start tkinter mainloop
        window.mainloop()

    # if user chooses course
    else:
        # Load the pre-computed job description embeddings
        Description_matrix = np.load('course_matrix.npy')
        # read program data from csv file
        df = pd.read_csv(
            r'C:\Users\不知道叫什么\Desktop\HKU sme2\ARIN7102 Applied data mining and text analytics\project/program.csv')

        # extract university and program names and combine them into a list of titles
        university = list(df['university'])
        program_name = list(df['program_name'])
        Title = [str(s1) + ' ' + str(s2) for s1, s2 in zip(university, program_name)]

        # create tkinter window
        window = tk.Tk()
        window.title("Chatbot")
        window.geometry("500x500")

        # create scrolledtext widget to display chat history
        chat_box = scrolledtext.ScrolledText(window, state='disabled')
        chat_box.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # create entry widget for user input
        input_box = tk.Entry(window)
        input_box.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        input_box.bind("<Return>", lambda event: send_message())

        # start tkinter mainloop
        window.mainloop()