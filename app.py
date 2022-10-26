from flask import Flask, request,app,jsonify,url_for,render_template
import numpy as np
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification



app = Flask(__name__)

# Load the Tokenizer from hugging face hub
TOKENIZER = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the model
model = TFDistilBertForSequenceClassification.from_pretrained("book_class_saved_model_2022-10-23 19_46_55.232799.bin")

label = open("label.txt","r").read()
id2label = {key:value for key,value in enumerate(label.split("\n"))}


@app.route("/")
def homePage():
    return render_template("home.html")

@app.route('/predict_api',methods=["POST"])

def predict_api():
    data = request.json["data"]
    print(data)

    tokenized_data = TOKENIZER(list(data.values()), truncation = True, max_length = 128, padding= True, return_tensors = "tf")

    prediction = model(tokenized_data).logits

    classifications = np.argmax(prediction, axis=1)
    print(classifications)

    output = id2label[classifications[0]]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
    