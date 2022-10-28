# Book Genre Classification
![image](https://user-images.githubusercontent.com/75915883/198703771-e8faf717-85de-4146-81db-7b4175b886ce.png)

The main aim here is to predict the genre of the book based on the synopsis of the book.

You can also find the same project as a [Colab notebook]() on this link.

### Software and Tools Requirements

1. [Github Account](https://github.com)
2. [Heroku Account](https://heroku.com)
3. [VS Code IDE](https://code.visualstudio.com/)
4. [Git CLI](https://cli.github.com/)
5. [Comet.ml](https://comet_ml.com)

# Dataset

* Download the dataset for custom training. You can also find more details about the dataset at this link.
  * https://www.kaggle.com/datasets/athu1105/book-genre-prediction

# Download pre-trained(State-of-the-art) model.

* Download the Transformers model manually: In this project I chose **DistilBert**
* https://huggingface.co/models

# Installation

* You can create a new environment for this project. If you want to make edits to the project and try new models, I recommend creating a new environment.


## Create a new enviroment

```
conda create -p venv python==3.7 -y
```

## Download the required libraries

```
pip install -r requirements.txt
```

# Inference Demo
* Testing with your book summary. (It may take some time, please wait.)
* **If the required libraries are not available, you will get an error. Please install required libraries**

```
streamlit run app.py
```

# Author
* [Furkan CEYRAN](https://github.com/Cygnie)