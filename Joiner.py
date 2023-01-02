import json
import os

symbols = ['"','?','!','/']
# Подготовка датасета
ProjectDir = os.getcwd()
DatasetFile = open(os.path.join(ProjectDir,"Datasets/ArtyomDataset.json"),"r",encoding="utf-8")
Dataset = json.load(DatasetFile)
DatasetFile.close

AdditionalDatasetFile = open(os.path.join(ProjectDir,"Datasets/RuBQ_2.0.json"),"r",encoding="utf-8")
AdditionalDataset = json.load(AdditionalDatasetFile)
AdditionalDatasetFile.close()

AdditionalDatasetFile = open(os.path.join(ProjectDir,"Datasets/RuBQ_2.0_test.json"),"r",encoding="utf-8")
AdditionalDataset = json.load(AdditionalDatasetFile)
AdditionalDatasetFile.close()

ArtyomSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/ArtyomAnswers.json"),"r",encoding="utf-8")
ArtyomSettings = json.load(ArtyomSettingsFile)
ArtyomSettingsFile.close()

SettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/Settings.json"),"r",encoding="utf-8")
Settings = json.load(SettingsFile)
SettingsFile.close()

def RuBQ():
    LatestInt = -1
    for value in Settings["CATEGORIES"]:
        LatestInt += 1
    print(LatestInt)
    for Group in AdditionalDataset:
        # for symbol in symbols:
        Question = (Group["question_text"]).replace('"',"")
        Answer = (Group["answer_text"]).replace('"',"")
        Dataset["dataset"].update(
            {
                Answer:{
                    "questions": [Question]
                }
            }
        )
        ArtyomSettings.update(
            {
                Answer: [Answer]
            }
        )
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



    for value in Settings["CATEGORIES"]:
        LatestInt += 1
    print(LatestInt)
    for Group in AdditionalDataset:
        # for symbol in symbols:
        Question = (Group["question_text"]).replace('"',"")
        Answer = (Group["answer_text"]).replace('"',"")
        Dataset["dataset"].update(
            {
                Answer:{
                    "questions": [Question]
                }
            }
        )
        ArtyomSettings.update(
            {
                Answer: [Answer]
            }
        )
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

def AddCategory(Name,Value):
    if not Name in Dataset["dataset"]:
        Dataset["dataset"].update(
            {
                Name:{
                    "questions": [Value]
                }
            }
        )
    elif Name in Dataset["dataset"]:
        Dataset["dataset"][Name].append(Value)

def AddValue(NameCategory,Value):
    if NameCategory in Dataset["dataset"]:
        Dataset["dataset"][NameCategory]["questions"].append(Value)
    elif not NameCategory in Dataset["dataset"]:
        print("Category is not found in dataset.")

def Save():
    DatasetFile = open(os.path.join(ProjectDir,"Datasets/ArtyomDataset_2.json"),"w",encoding="utf-8")
    json.dump(Dataset,DatasetFile,ensure_ascii=False, indent=2)
    DatasetFile.close

    ArtyomSettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/ArtyomAnswers_2.json"),"w",encoding="utf-8")
    json.dump(ArtyomSettings,ArtyomSettingsFile,ensure_ascii=False, indent=2)
    ArtyomSettingsFile.close()

    SettingsFile = open(os.path.join(ProjectDir,"NeuralNetworkSettings/Settings_2.json"),"w",encoding="utf-8")
    json.dump(Settings,SettingsFile,ensure_ascii=False, indent=2)
    SettingsFile.close()

while True:
    command = input(">>>")
    if command == "ac":
        category = input("Name Category:")
        value = input("Enter one value:")
        AddCategory(category,value)
    elif command == "av":
        category = input("Name Category:")
        value = input("Value:")
        AddValue(category,value)
    elif command == "rubq":
        RuBQ()
    elif command == "save":
        Save()
    elif command == 'exit':
        break

Save()
