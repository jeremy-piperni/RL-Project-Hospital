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
    "pulse finding": "blood pressure and pulse measurement",
    "inflammed sinuses": "inflamed sinuses",
    "breathing difficulties": "breathing problems",
    "t. pedis detected": "tinea pedis detected",
    "injury to the chest": "chest injury",
    "thickening and distortion of the nail": "nail thickening",
    "injury to the arm": "arm injury",
    "injury to the foot": "foot injury",
    "beef tapeworm present": "beef tapeworm detected",
    "painful cervical lymph nodes": "painful lymph nodes",
    "muscles and joints pain": "muscle and joint pain",
    "itching eye": "itchy eyes",
    "lung findings": "abnormal lung findings",
    "lactose intolerance detected": "lactose intolerance",
    "loss of apetite": "loss of appetite",
    "pork tapeworm present": "pork tapeworm detected",
    "otorhinolaryngological findings": "orl findings",
    "injury to the hand": "hand injury",
    "inability to move the joint": "joint immobility",
    "rsv and advs present": "rsv present",
    "hemoglobin low": "low hemoglobin",
    "sleeping difficulties": "sleeping problems",
    "raynaud's findings": "raynauds findings",
    "chr. fatigue syndrome": "chronic fatigue syndrome",
    "injury to the leg": "leg injury",
    "flatulence": "excessive flatulence",
    "long react time": "long reaction time",
    "ulcers in colon discovered": "colon ulcers",
    "crp high": "elevated crp",
    "persistent urge to urinate": "urinary urgency",
    "inflammed bilduct": "inflamed bilduct",
    "excessive burping": "excessive belching",
    "gallstones discovered": "gallstones identified",
    "tsh level high": "tsh - high level",
    "tsh level low": "tsh - low level",
    "varices in oesophagus": "varices in the oesophagus",
    "urethral stones detected": "ureter stone detected",
    "inflammed thyroid": "inflamed thyroid",
    "penetrating pancreas laceration": "pancreas laceration",
    "lft high level": "abnormal lft",
    "kidney tissue inflammed": "kidney tissue inflamed",
    "gastric ulcers discovered": "gastric ulcers identified",
    "tubulus inflammed": "tubulus inflammation",
    "peritoneal fluid infected": "peritoneal fluid shows infection",
    "pancreatic tissue inflammed": "pancreatic inflammation",
    "pancreatic elastase low level": "elastase - low level",
    "penetrating spleen rupture": "spleen rupture",
    "crohn disease discovered": "crohn's disease discovered",
    "e. histolytica discovered": "e.coli toxin detected",
    "bladder lining defected": "bladder lining defect",
    "viral infection of oesophagus": "oesophagus infection",
    "abdominal swelling": "swollen abdomen",
    "hearing difficulties": "hearing problems",
    "penetrating kidney laceration": "kidney laceration",
    "guillain-barre findings": "guillain-barre detected"

}

examinations = examinations.replace(exams_to_change)
examinations = examinations[~examinations.isin(exams_to_remove)]
symptoms = symptoms.replace(symptoms_to_change)
temp_symptoms = pd.DataFrame({"symptom":["influenza b detected","c.botulinum detected","tia detected","mastadenovirus b detected"],"examination 1":["serology testing testing","bacteria cultivation testing","mri","pcr testing"]})
symptoms = pd.concat([symptoms, temp_symptoms], ignore_index=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data", "Examinations.csv")
new_path = os.path.abspath(new_path)
examinations.to_csv(new_path, index=False)

new_path = os.path.join(script_dir, os.pardir, "Data", "Symptoms.csv")
new_path = os.path.abspath(new_path)
symptoms.to_csv(new_path, index=False)

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
        if symptom == "nasal sneezing":
            symptom = "sneezing"
        elif symptom == "c.jejuni detected":
            symptom = "stool containing bacterias"
        elif symptom == "campylobacter in stool":
            symptom = "stool containing bacterias"
        elif symptom == "breathing difficulties":
            symptom = "breathing problems"
        elif symptom == "sleeping problem":
            symptom = "sleeping problems"
        probability = int(row2["Probability"]) / 100
        data[symptom] = probability
    diagnosis_emergency.loc[index, "Symptoms"] = [data]
    time.sleep(random.uniform(1,3))

print(diagnosis_emergency)

diagnosis_emergency["Diagnosis"] = diagnosis_emergency["Diagnosis"].apply(lambda x: x.lower())
diagnosis_emergency["Diagnosis"] = diagnosis_emergency["Diagnosis"].apply(lambda x: remove_non_ascii(x))
diagnosis_emergency = diagnosis_emergency.drop(columns=["Link"])

new_path = os.path.join(script_dir, os.pardir, "Data", "Emergency_Diagnosis.csv")
new_path = os.path.abspath(new_path)
diagnosis_emergency.to_csv(new_path, index=False)

tables = soup.find_all('table')

data = []
for row in tables[1].find_all('tr')[1:]:
    first_td = row.find('td')
    if first_td:
        link_tag = first_td.find('a')
        title = first_td.get_text(strip=True)
        link = link_tag['href'] if link_tag and link_tag.has_attr('href') else None
        data.append({"Diagnosis": title, "Link": link})

diagnosis_gen_surgery = pd.DataFrame(data)
print(diagnosis_gen_surgery)


diagnosis_gen_surgery["Symptoms"] = 0
for index, row in diagnosis_gen_surgery.iterrows():
    end_link = row["Link"]
    url = "https://projecthospital.shoutwiki.com" + end_link
    tables = pd.read_html(url)
    table = tables[2]
    print(url)
    data = {}
    for index2, row2 in table.iterrows():
        symptom = remove_non_ascii(row2["Symptom"].lower())
        if symptom == "nasal sneezing":
            symptom = "sneezing"
        elif symptom == "c.jejuni detected":
            symptom = "stool containing bacterias"
        elif symptom == "campylobacter in stool":
            symptom = "stool containing bacterias"
        elif symptom == "breathing difficulties":
            symptom = "breathing problems"
        elif symptom == "sleeping problem":
            symptom = "sleeping problems"
        probability = int(row2["Probability"]) / 100
        data[symptom] = probability
    diagnosis_gen_surgery.loc[index, "Symptoms"] = [data]
    time.sleep(random.uniform(1,3))

print(diagnosis_gen_surgery)

diagnosis_gen_surgery["Diagnosis"] = diagnosis_gen_surgery["Diagnosis"].apply(lambda x: x.lower())
diagnosis_gen_surgery["Diagnosis"] = diagnosis_gen_surgery["Diagnosis"].apply(lambda x: remove_non_ascii(x))
diagnosis_gen_surgery = diagnosis_gen_surgery.drop(columns=["Link"])

new_path = os.path.join(script_dir, os.pardir, "Data", "Gen_Surgery_Diagnosis.csv")
new_path = os.path.abspath(new_path)
diagnosis_gen_surgery.to_csv(new_path, index=False)

tables = soup.find_all('table')

data = []
for row in tables[5].find_all('tr')[1:]:
    first_td = row.find('td')
    if first_td:
        link_tag = first_td.find('a')
        title = first_td.get_text(strip=True)
        link = link_tag['href'] if link_tag and link_tag.has_attr('href') else None
        data.append({"Diagnosis": title, "Link": link})

diagnosis_neuro = pd.DataFrame(data)
print(diagnosis_neuro)


diagnosis_neuro["Symptoms"] = 0
for index, row in diagnosis_neuro.iterrows():
    end_link = row["Link"]
    url = "https://projecthospital.shoutwiki.com" + end_link
    tables = pd.read_html(url)
    table = tables[2]
    print(url)
    data = {}
    for index2, row2 in table.iterrows():
        symptom = remove_non_ascii(row2["Symptom"].lower())
        if symptom == "nasal sneezing":
            symptom = "sneezing"
        elif symptom == "c.jejuni detected":
            symptom = "stool containing bacterias"
        elif symptom == "campylobacter in stool":
            symptom = "stool containing bacterias"
        elif symptom == "breathing difficulties":
            symptom = "breathing problems"
        elif symptom == "sleeping problem":
            symptom = "sleeping problems"
        elif symptom == "sleeping difficulties":
            symptom = "sleeping problems"
        elif symptom == "itching eye":
            symptom = "itchy eyes"
        elif symptom == "muscles and joints pain":
            symptom = "muscle and joint pain"
        elif symptom == "hearing difficulties":
            symptom = "hearing problems"
        probability = int(row2["Probability"]) / 100
        data[symptom] = probability
    diagnosis_neuro.loc[index, "Symptoms"] = [data]
    time.sleep(random.uniform(1,3))

print(diagnosis_neuro)

diagnosis_neuro["Diagnosis"] = diagnosis_neuro["Diagnosis"].apply(lambda x: x.lower())
diagnosis_neuro["Diagnosis"] = diagnosis_neuro["Diagnosis"].apply(lambda x: remove_non_ascii(x))
diagnosis_neuro = diagnosis_neuro.drop(columns=["Link"])

new_path = os.path.join(script_dir, os.pardir, "Data", "Neuro_Diagnosis.csv")
new_path = os.path.abspath(new_path)
diagnosis_neuro.to_csv(new_path, index=False)