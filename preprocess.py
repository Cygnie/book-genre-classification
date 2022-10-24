import pandas as pd
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split 



def parse_args():

    parser = argparse.ArgumentParser()
    
    # CLI args
    
    parser.add_argument('--train_data_path', 
                        type=str, 
                        default="data.csv")
    
    parser.add_argument('--train_test_split', 
                        type=float, 
                        default=0.2)

    return parser.parse_args()

    
if __name__ == '__main__':

    args = parse_args()

    try:
        df = pd.read_csv(args.train_data_path)
    except:
        print("Data failed to initialize properly")

    print("Target class {} \n ".format(len(df.genre.value_counts())))

    print(f"Target Distributrion: {df.genre.value_counts()}")

    df.drop(["index","title"],inplace = True, axis = 1)
    df.rename(columns = {'genre':"label"},inplace = True)

    id2label = {key:value for key,value in enumerate(df["label"].unique())}

    label2id = {value:key for key,value in id2label.items()}

    print("id2label ",id2label,"\n")
    print("label2id",label2id)

    df.replace(label2id,inplace = True)
    
    print(f"Target Value Counts:\n{df.label.value_counts()}")

    try:
        with open("label.txt","w") as f:
            for item in id2label.values():
                f.write(f"{item}\n")
    except:
        print("Something went wrong when writing to the file\tFailed to write book_label file ")


    # Train - Test split ==> 0.2
    train_df, test_df = train_test_split(df, test_size = args.train_test_split, shuffle = True, stratify=df["label"])

    train_df.to_csv("cleaned_book_data.csv",index = False)
    test_df.to_csv("book_test_data.csv",index = False)

    

