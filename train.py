import comet_ml

experiment = comet_ml.Experiment(

api_key="<YOUR APİ KEY>",
project_name="<YOUR PROJECT NAME>",
workspace="<YOUR WORKSPACE>",
)
experiment.set_name("<YOUR EXPERIMENT NAME>")

import argparse
import datetime
import random
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf # Tensorflow
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification # Transformers lib from hugging face
from datasets import load_dataset # Hugging face Transformers Lib Dataset Module
from transformers import DataCollatorWithPadding, create_optimizer
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter


# Comet ML
from comet_ml import Experiment

#create an experiment with your api key


def parse_args():

    parser = argparse.ArgumentParser()
    
    # CLI args
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=128)
    
    parser.add_argument('--epochs', 
                        type=int, 
                        default=50)
    
    parser.add_argument('--freeze_pretrained_layer', 
                        type=eval, 
                        default=True)
    
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=5e-5)
    
    parser.add_argument('--max_seq_length', 
                        type=int, 
                        default=128)

    parser.add_argument('--train_data', 
                        type=str, 
                        default="cleaned_book_data.csv")
        
    parser.add_argument('--test_data', 
                        type=str, 
                        default="book_test_data.csv")
    
    parser.add_argument('--model_name', 
                        type=str, 
                        default=f"model/book_class_saved_model_{str(datetime.datetime.today())}.bin")

    parser.add_argument('--model_checkpoint', 
                        type=str, 
                        default="distilbert-base-uncased")

    return parser.parse_args()


########################################################### Tools and variables ################################################################

def load_and_prepare_dataset(
        dataset,
        model,
        TOKENIZER,
        MAX_LENGTH,
        BATCH_SİZE
        ):
    
    def prepare_dataset_features (example):
      return TOKENIZER(
        example["summary"],
        truncation = True,
        max_length = MAX_LENGTH
        )
        
    dataset = dataset["train"].train_test_split(train_size = 0.8, shuffle =True, seed = 42 )
    
    
    tokenized_dataset = dataset.map(
                                        prepare_dataset_features,
                                        batched=True,
                                        num_proc=3,
                                        )
    
    tokenized_dataset = tokenized_dataset.remove_columns(["summary"])

    data_collator = DataCollatorWithPadding(tokenizer=TOKENIZER, return_tensors="tf")
    
    tf_train_dataset = model.prepare_tf_dataset(
                                                tokenized_dataset['train'],
                                                shuffle=True,
                                                batch_size=BATCH_SİZE,
                                                collate_fn=data_collator,
                                                tokenizer=TOKENIZER
                                                )
                                            
    tf_validation_dataset = model.prepare_tf_dataset(
                                                    tokenized_dataset['test'],
                                                    shuffle=False,
                                                    batch_size=BATCH_SİZE,
                                                    collate_fn=data_collator,
                                                    tokenizer=TOKENIZER
                                            )
    
    return tf_train_dataset,tf_validation_dataset
    


    ################################################################################## MAİN ##################################################################################




if __name__ == '__main__':

    args = parse_args()

    MODEL_CHECKPOİNT = args.model_checkpoint
    MODEL_NAME = args.model_name

    dataset = load_dataset("csv",data_files=args.train_data)

    # HYPERPARAMETER 
    NUM_CLASS = len(np.unique(dataset["train"]["label"]))
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    MAX_LENGTH = args.max_seq_length
    LEARNING_RATE = args.learning_rate
    
    CLASS_WEIGHTS ={ key:len(dataset["train"]["label"])/(len(np.unique(dataset["train"]["label"]))*value) for key,value in Counter(dataset["train"]["label"]).items()}



    model = None
    TOKENIZER = None   
    successful_download = False

    retries = 0
    
    while (retries < 5 and not successful_download):
        try:
            # Download Pretrained Tokenizer 
            TOKENIZER = DistilBertTokenizer.from_pretrained(MODEL_CHECKPOİNT)

            
            # Download Pretrained Model
            model = TFDistilBertForSequenceClassification.from_pretrained(
                                                                MODEL_CHECKPOİNT,
                                                                num_labels = NUM_CLASS
                                                              )
    
            successful_download = True
            print('Sucessfully downloaded after {} retries.'.format(retries))
        
        except:
            retries = retries + 1
            random_sleep = random.randint(1, 30)
            print('Retry #{}.  Sleeping for {} seconds'.format(retries, random_sleep))
            time.sleep(random_sleep)
 
    if not model:
         print('Not properly initialized...')

    tf_train_dataset,tf_validation_dataset = load_and_prepare_dataset(
    dataset,
    model,
    TOKENIZER,
    MAX_LENGTH,
    BATCH_SIZE
    )
   
    batches_per_epoch = len(dataset["train"]) // BATCH_SIZE
    total_train_steps = int(batches_per_epoch * NUM_EPOCHS)
  
    if args.freeze_pretrained_layer:
        model.layers[0].trainable = False

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=LEARNING_RATE,
    end_learning_rate=0.,
    decay_steps=total_train_steps
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    
    
    model.compile(
                  optimizer=optimizer,
                  loss=loss_func,
                  metrics= ['sparse_categorical_accuracy'],
                  )

    ################################################################################## TRAIN MODEL ##################################################################################
    checkpoint_filepath = 'checkpoint/'

    early_stop_callback = tf.keras.callbacks.EarlyStopping(patience=3)

    save_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                                        filepath=checkpoint_filepath,
                                                        save_weights_only=True,
                                                        monitor='val_sparse_categorical_accuracy',
                                                        mode='max',
                                                        save_best_only=True),

    my_callbacks = [early_stop_callback,save_checkpoint]

    with experiment.train():
        hist = model.fit(
        tf_train_dataset,
        validation_data=tf_validation_dataset,
        epochs= NUM_EPOCHS,
        callbacks=my_callbacks,
        class_weight = CLASS_WEIGHTS
        )

    train_metrics = {
      'loss':hist.history["loss"][-1],
      'val_loss':hist.history["val_loss"][-1],
      'accuracy':hist.history["sparse_categorical_accuracy"][-1],
      'val_accuracy':hist.history["val_sparse_categorical_accuracy"][-1],
    }
    experiment.log_metrics(train_metrics,prefix ="train")

    comet_params={
        'Batch Size':BATCH_SIZE,
        'Epochs':NUM_EPOCHS,
        'Optimizer':optimizer,
        'Loss Function':loss_func,
        'Learning Rate:':LEARNING_RATE,
        'Max Length':MAX_LENGTH
        }
      ####### CLASS_WEİGHT KULLANARAK TEST YAP ##################

    ################################################################################## EVAULATE MODEL ##################################################################################

    try:
        test_data = load_dataset("csv",data_files = args.test_data)
    except:
        print("Something went wrong when reading to the test file")
    
    def tokenize_dataset(dataset):
        encoded = TOKENIZER(
            dataset["summary"],
            padding=True,
            truncation=True,
            return_tensors='np',
        )
        return encoded.data

    tokenized_test_dataset = {
        split: tokenize_dataset(test_data[split]) for split in test_data.keys()
    }

    y_test = np.array(test_data["train"]["label"])


    # Predictions 
    preds = model.predict(tokenized_test_dataset["train"])['logits']
    probabilities = tf.nn.softmax(preds)
    class_preds = np.argmax(probabilities, axis=1)

    def log_classification_report(y_true, y_pred):
        report = classification_report(y_true, y_pred, output_dict=True)
        for key, value in report.items():
          if key == "accuracy":
            experiment.log_metric(key, value)
          else:
            experiment.log_metrics(value, prefix=f'{key}')


    with experiment.test():
        loss, accuracy = model.evaluate(tokenized_test_dataset["train"],y_test,batch_size = BATCH_SIZE)
        log_classification_report(y_test, class_preds)

    test_metrics = {
        'loss':loss,
        'accuracy':accuracy
    }
    
    experiment.log_metrics(test_metrics,prefix ="test")

    cm = comet_ml.ConfusionMatrix()
    cm.compute_matrix(y_test, class_preds)

    experiment.log_confusion_matrix(matrix=cm)
    experiment.log_parameters(comet_params, prefix = "train")
    experiment.log_dataset_hash(tf_train_dataset)

    model.save_pretrained(MODEL_NAME)
    experiment.log_model("Text Classification Model", MODEL_NAME)

    experiment.end()
    
    
    
    
    
    
    
    