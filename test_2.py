import pandas as pd
import os
ProjectDir = os.path.dirname(os.path.realpath(__file__))
Dataset = pd.read_csv(os.path.join(ProjectDir,'Datasets/CommunicationDataset.tsv'),sep = '\t')
for value in Dataset.reply[:128]:
    print(value)