import numpy as np
import pandas as pd
import gym
import random
from gym import spaces

class DiagnosisEnv(gym.Env):
    def __init__(self, df_symptoms, df_examinations, df_diagnoses, max_steps=10):
        super(DiagnosisEnv, self).__init__()

        self.symptoms = df_symptoms["symptom"].tolist()
        self.symptom_to_index = {s: i for i, s in enumerate(self.symptoms)}
        self.examinations = df_examinations["Examination"].tolist()
        self.exam_to_symptoms = self._build_exam_symptom_map(df_symptoms)

        self.diagnoses = self._parse_diagnoses(df_diagnoses)

        self.max_steps = max_steps
        self.observation_space = spaces.MultiBinary(len(self.symptoms))
        self.action_space = spaces.Discrete(len(self.examinations))

    def _build_exam_symptom_map(self, df_symptoms):
        mapping = {}
        for _, row in df_symptoms.iterrows():
            for col in ["examination 1","examination 2","examination 3"]:
                exam = row[col]
                if pd.notna(exam) and exam != "None":
                    mapping.setdefault(exam, []).append(row["symptom"])

        return mapping

    def _parse_diagnoses(self, df_diagnoses):
        diagnoses = []
        for _, row in df_diagnoses.iterrows():
            name = row["Diagnosis"]
            symptom_probs = eval(row["Symptoms"])
            if isinstance(symptom_probs, list) and len(symptom_probs) == 1 and isinstance(symptom_probs[0], dict):
                symptom_probs = symptom_probs[0]
            diagnoses.append((name, symptom_probs))
        return diagnoses

    def reset(self):
        self.steps = 0
        self.observed_symptoms = np.zeros(len(self.symptoms), dtype=int)
        self.chosen_diagnosis, symptom_probs = random.choice(self.diagnoses)

        self.true_symptoms = np.zeros(len(self.symptoms), dtype=int)
        for sym, prob in symptom_probs.items():
            if np.random.rand() < prob:
                self.true_symptoms[self.symptom_to_index[sym]] = 1
        self.exams_done = set()
        return self.observed_symptoms.copy()

    def step(self, action):
        exam = self.examinations[action]
        if exam in self.exams_done:
            guessed_diagnosis = self._diagnose()
            done = True
            reward = 1 if guessed_diagnosis == self.chosen_diagnosis else -1
            return self.observed_symptoms.copy(), reward, done, {
                "diagnosis": self.chosen_diagnosis,
                "forced": True,
                "guess": guessed_diagnosis
            }
        self.steps += 1
        self.exams_done.add(exam)
        for sym in self.exam_to_symptoms.get(exam, []):
            index = self.symptom_to_index[sym]
            self.observed_symptoms[index] = self.true_symptoms[index]

        done = self.steps >= self.max_steps
        reward = -0.1
        return self.observed_symptoms.copy(), reward, done, {"forced": False}

    def make_diagnosis(self, guess):
        done = True
        correct = guess == self.chosen_diagnosis
        if correct:
            reward = 1
        else:
            reward = -1
        return self.observed_symptoms.copy(), reward, done, {"diagnosis": self.chosen_diagnosis}

    def _diagnose(self):
        max_score = -1
        best_guess = None
        for diagnosis, symptom_probs in self.diagnoses:
            score = sum(
                self.observed_symptoms[self.symptom_to_index[sym]]
                for sym in symptom_probs.keys()
                if sym in self.symptom_to_index)
            if score > max_score:
                max_score = score
                best_guess = diagnosis
        return best_guess
    
    def get_true_diagnosis(self):
        return self.chosen_diagnosis

    def get_true_symptoms(self):
        return [
            symptom for symptom, idx in self.symptom_to_index.items()
            if self.true_symptoms[idx] == 1]

    def get_observed_symptoms(self):
        return [
            symptom for symptom, idx in self.symptom_to_index.items()
            if self.observed_symptoms[idx] == 1]