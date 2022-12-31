import json
import os
symbols = ['"','?','!','/']
# Подготовка датасета
ProjectDir = os.getcwd()
DatasetFile = open(os.path.join(ProjectDir,"Datasets/ArtyomDataset_2.json"),"r",encoding="utf-8")
Dataset = json.load(DatasetFile)
DatasetFile.close
test = {}
AdditionalDatasetFile = open(os.path.join(ProjectDir,"Datasets/RuBQ_2.0.json"),"r",encoding="utf-8")
AdditionalDataset = json.load(AdditionalDatasetFile)
AdditionalDatasetFile.close()

ArtyomSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/ArtyomAnswers_2.json"),"r",encoding="utf-8")
ArtyomSettings = json.load(ArtyomSettingsFile)
ArtyomSettingsFile.close()

SettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/Settings_2.json"),"r",encoding="utf-8")
Settings = json.load(SettingsFile)
SettingsFile.close()

for Group in AdditionalDataset:
    for symbol in symbols:
        Question = (Group["question_text"]).replace(symbol,"")
        Answer = (Group["answer_text"]).replace(symbol,"")
    Dataset["dataset"].update(
        {
            Answer: [Question]
        }
    )
    ArtyomSettings.update(
        {
            Answer:[Question]
        }
    )
    LatestInt = -1
    for value in Settings["CATEGORIES"]:
        LatestInt += 1
    # LatestInt_2 = LatestInt
    LatestInt += 1
    Settings["CATEGORIES"].update(
        {
            Answer: LatestInt
        }
    )

    Settings["CATEGORIES_TARGET"].update(
        {
            LatestInt:Answer
        }
    )

DatasetFile = open(os.path.join(ProjectDir,"Datasets/ArtyomDataset_2.json"),"w",encoding="utf-8")
json.dump(Dataset,DatasetFile,ensure_ascii=False, indent=2)
DatasetFile.close

ArtyomSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/ArtyomAnswers_2.json"),"w",encoding="utf-8")
json.dump(ArtyomSettings,ArtyomSettingsFile,ensure_ascii=False, indent=2)
ArtyomSettingsFile.close()

SettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/Settings_2.json"),"w",encoding="utf-8")
json.dump(Settings,SettingsFile,ensure_ascii=False, indent=2)
SettingsFile.close()

