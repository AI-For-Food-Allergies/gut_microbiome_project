# Main script to run train + evaluation

from data_loading import load_data
from modules.classifier import MicrobiomeClassifier
from train import train_classifier
from evaluation import evaluate_classifier

def __init__():
    # load data
    data = load_data()
    # load model
    model = MicrobiomeClassifier()
    # train model
    train_classifier()
    # evaluate model
    evaluate_classifier()
    # save model
    save_model()
    ...