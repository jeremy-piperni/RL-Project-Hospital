import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import os
import re

def remove_non_ascii(val):
    if isinstance(val, str):
        return re.sub(r'[^\x00-\x7F]+', '', val)
    return val

# Extract Examination information
url = "https://projecthospital.shoutwiki.com/wiki/Examination"
tables = pd.read_html(url)
table = tables[0]
examinations = table["Examination"]
examinations = examinations.apply(lambda x: x.lower().rstrip('"'))

# Extract Symptom information
url = "https://projecthospital.shoutwiki.com/wiki/Symptom"
tables = pd.read_html(url)
table = tables[0]
symptoms = table[[0,7,8,9]]
symptoms = symptoms.map(lambda x: x.lower() if isinstance(x, str) else x)
symptoms.columns = symptoms.iloc[0]
symptoms = symptoms[1:]
print(symptoms)

exam_list_main = set(examinations.dropna().str.strip())
exam_columns = ["examination 1", "examination 2", "examination 3"]
exam_list_symptoms = set(symptoms[exam_columns].stack().dropna().str.strip())
missing_exams = exam_list_main - exam_list_symptoms


# examinations to be removed, and examinations to be edited
temp = ["mycologic sampling"]
exams_to_remove = ["peritoneal fluid analysis - sampling", "cbc - sampling", "triage in reception", "biopsy - sampling", "csf analysis","mycologic sampling", "csf sampling", "microbial sampling",
                   "elastase test - sampling", "pcr - sampling", "differential diagnosis", "blood draw", "stool collecting", "serologic sampling", "urine analysis - sampling"]
exams_to_change = {
    "x-ray lower limb": "x ray lower limb",
    "blood test": "blood test testing",
    "urgent echo": "echo",
    "biopsy - testing": "biopsy testing",
    "urine analysis - testing": "urine sample analysis testing",
    "ct - enterography": "ct enterography",
    "x-ray head": "x ray head",
    "stool analysis": "stool analysis testing",
    "barium swallowing": "barrium swallow x ray",
    "skin allergy test": "skin injection test",
    "peritoneal fluid analysis - testing": "peritoneal fluid analysis testing",
    "fungal cultivation": "fungal cultivation testing",
    "pcr - testing": "pcr testing",
    "chest auscultation": "chest listening",
    "x-ray back": "x ray back",
    "x-ray upper limb": "x ray upper limb",
    "x-ray chest": "x ray torso",
    "cbc - testing": "cbc testing",
    "serologic testing": "serology testing testing",
    "blood analysis - icu": "urgent blood analysis",
    "elastase test - testing": "fecal elastase test testing",
    "physical examination": "physical and visual examination",
    "blood pressure measurement": "blood pressure and pulse measurement",
    "ps blood control": "urgent blood analysis",
    "csf analysis": "spinal fluid analysis testing",
    "microbial cultivation": "bacteria cultivation testing"
}

symptoms_to_change = {
    "compression wraps": "x ray torso",
    "bacteria cultivation sampling": "bacteria cultivation testing",
    "cc echo": "echo",
    "cc blood analysis": "urgent blood analysis",
    "pulse finding": "blood pressure and pulse measurement"
}

examinations = examinations.replace(exams_to_change)
examinations = examinations[~examinations.isin(exams_to_remove)]
symptoms = symptoms.replace(symptoms_to_change)

url = "https://projecthospital.shoutwiki.com/wiki/Diagnosis"

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

tables = soup.find_all('table')

data = []
for row in tables[0].find_all('tr')[1:]:
    first_td = row.find('td')
    if first_td:
        link_tag = first_td.find('a')
        title = first_td.get_text(strip=True)
        link = link_tag['href'] if link_tag and link_tag.has_attr('href') else None
        data.append({"Diagnosis": title, "Link": link})

diagnosis_emergency = pd.DataFrame(data)
print(diagnosis_emergency)


diagnosis_emergency["Symptoms"] = 0
for index, row in diagnosis_emergency.iterrows():
    end_link = row["Link"]
    url = "https://projecthospital.shoutwiki.com" + end_link
    tables = pd.read_html(url)
    table = tables[2]
    print(url)
    data = {}
    for index2, row2 in table.iterrows():
        symptom = remove_non_ascii(row2["Symptom"].lower())
        probability = row2["Probability"]
        data[symptom] = probability
    diagnosis_emergency.loc[index, "Symptoms"] = [data]
    time.sleep(random.uniform(1,3))

print(diagnosis_emergency)

diagnosis_emergency["Diagnosis"] = diagnosis_emergency["Diagnosis"].apply(lambda x: x.lower())
diagnosis_emergency["Diagnosis"] = diagnosis_emergency["Diagnosis"].apply(lambda x: remove_non_ascii(x))
diagnosis_emergency = diagnosis_emergency.drop(columns=["Link"])

script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "Examinations.csv")
new_path = os.path.abspath(new_path)
examinations.to_csv(new_path, index=False)

new_path = os.path.join(script_dir, os.pardir, "Data", "Symptoms.csv")
new_path = os.path.abspath(new_path)
symptoms.to_csv(new_path, index=False)

new_path = os.path.join(script_dir, os.pardir, "Data", "Emergency_Diagnosis.csv")
new_path = os.path.abspath(new_path)
diagnosis_emergency.to_csv(new_path, index=False)