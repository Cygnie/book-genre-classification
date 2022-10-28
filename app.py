import numpy as np
import streamlit as st
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")

# Load the Tokenizer from hugging face hub
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the model
model = TFDistilBertForSequenceClassification.from_pretrained("model.bin")

label = open("label.txt","r").read()
id2label = {key:value for key,value in enumerate(label.split("\n"))}

def predict(summary,id2label):

    tokenized_data = TOKENIZER([summary], truncation = True, max_length = 128, padding= True, return_tensors = "tf")

    prediction = model(tokenized_data).logits

    classifications = np.argmax(prediction, axis=1)

    output = id2label[classifications[0]]
    return output

def main():
    st.set_page_config(page_title="Book Genre Classification")
    st.markdown("<h1 style='text-align:center;'>Book Genre Classification</h1>",unsafe_allow_html=True)
    st.write("Please fill in the text field with your book summary")
    summary = st.text_area("summary")

    result=""
    if st.button("Predict"):
        result=predict(summary,id2label)
    st.success('The output is {}'.format(result))

if __name__ == '__main__':
    main()
    