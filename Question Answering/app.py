import streamlit as st
import numpy as np 
import pandas as pd 
import json 

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import textwrap
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm.notebook import tqdm
import time
import wikipedia
import functools

MODEL_NAME= "t5-base"

device = torch.device('cpu')

@st.cache(allow_output_mutation=True)
def load_qa_model():
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    checkpoint = torch.load('model.tar', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model= model.to(device)
    tokernizer= T5Tokenizer.from_pretrained(MODEL_NAME)
    return model, tokernizer



def generate_answer(data):
    source_encoding = tokernizer(
            data['question'],
            data['context'],
            max_length=369,
            padding= "max_length",
            truncation= "only_second",
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors= "pt"
        ).to(device)
    
    generated_ids = model.generate(
        input_ids=source_encoding['input_ids'],
        attention_mask= source_encoding['attention_mask'],
        num_beams= 1,
        max_length=80,
        length_penalty=1.0,
        early_stopping=True,
        use_cache= True
    )
    preds = [
        tokernizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    return "".join(preds)

def conditional_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if config["framework"] == "pt":
            qa = st.cache(func)(*args, **kwargs)
        else:
            qa = func(*args, **kwargs)
        return qa

    return 

# @conditional_decorator
def get_wiki_paragraph(query: str) -> str:
    st.write(query)
    results = wikipedia.search(query)
    try:
        summary = wikipedia.summary(results[0], sentences=10)
    except wikipedia.DisambiguationError as e:
        ambiguous_terms = e.options
        return wikipedia.summary(ambiguous_terms[0], sentences=10)
    return summary


# context = ""
if __name__ == "__main__":
    model, tokernizer= load_qa_model()
    # topic = st.text_input("Enter a topic: ")
    # topic_button = st.button("Get Topic")

    # if topic_button:
    #     if topic != '':
    #         result = wikipedia.search(topic)
    #         add_selectbox = st.selectbox("What article do you like:", result)
    #         get_summery_button = st.button("Get summery")

    #         if get_summery_button:
    #             page = wikipedia.page(add_selectbox)
    #             context = page.summary
            
    #             context_para = st.write(context)


    #             st.title("Ask questions about Wikipedia summery")
    #             question = st.text_input("Questions from this article?")
    #             button = st.button("Get me Answers")
    #             with st.spinner("Discovering Answers.."):
    #                 if button:
    #                     print("hi") 
    #                     if question!= '':
    #                         data = { 'context': context, 'question': question }
    #                         answer = generate_answer(data)
    #                         wait_flag= True
    #                         st.write(answer)
    
    topic = st.text_input("Enter a topic: ", "")
    paragraph_slot = st.empty()

    if topic:
        result = wikipedia.search(topic)
        add_selectbox = st.selectbox("What article do you like:", result)
        page = wikipedia.page(add_selectbox)
        context = page.summary
        context_para = st.write(context)
        question = st.text_input("QUESTION", "")
        
        if question != "":
            data = {"question": question,
                    "context": context}
            try:
                answer = generate_answer(data)
                st.success(answer)
            except Exception as e:
                print(e)
                st.warning("You must provide a valid wikipedia paragraph")