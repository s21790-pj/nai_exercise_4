import csv
import pandas as pd

from training_models.svc_model import svc_model
from training_models.tree_model import tree_model


"""
Authors: Kamil Kaczówka, Szymon Olkiewicz.
To run application you have to install requirements by pip install -r requirements.txt 

Application printing in console classificator performance by test and training dataset.

"""



def sonar():
    # Load data without headers.
    sonar_data = pd.read_csv('datasets/sonar.csv', header=None)

    """
        Split the data in training and testing subsets.
    """
    X = sonar_data.values[:, :60]
    y = sonar_data.values[:, 60]

    #declaring svc and tree model
    svc_model(X, y)
    print("\n")
    tree_model(X, y)


def diabetes():
    X = []
    y = []

    # Load data
    with open('datasets/diabetes.csv', 'r') as newFile:
        plots = csv.reader(newFile)
        has_header = csv.Sniffer().has_header(newFile.read(1024))
        newFile.seek(0)  # on the beginning of file
        if has_header:
            next(plots)
        for row in plots:
            X.append(row[0:8])
            y.append(row[8])

    # declaring svc and tree model
    svc_model(X, y)
    print("\n")
    tree_model(X, y)



if __name__ == "__main__":
    print("Dataset from MLM \"Sonar Dataset\"\n=====")
    sonar()
    print("\n")
    print("Own diabetes dataset\n=====")
    diabetes()
