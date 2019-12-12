import pandas as pd


def strip_spaces(in_str):
    return in_str.strip()

def transform_salary(in_str):
    return 1 if in_str.strip('.') == ' >50K' else 0


def load_adult(path):
    dataset = pd.read_csv(path, names=["age", "workclass", "fnlwgt", "education", "education-num", 
                                              "marital-status", "occupation", "relationship", "race", 
                                              "sex", "capital-gain", "capital-loss", "hours-per-week", 
                                              "native-country", "salary"], converters={"workclass": strip_spaces, 
                                                                                       "education": strip_spaces, 
                                                                                       "marital-status": strip_spaces, 
                                                                                       "occupation": strip_spaces, 
                                                                                       "relationship": strip_spaces, 
                                                                                       "race": strip_spaces, 
                                                                                       "sex": strip_spaces, 
                                                                                       "native-country": strip_spaces, 
                                                                                       "salary": transform_salary})
    original_size = len(dataset)
    # make sure all values belong to their respective discrete set
    valid_workclass = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                       "Local-gov", "State-gov", "Without-pay", "Never-worked"]
    valid_education = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", 
                       "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters", 
                       "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
    valid_marital_status = ["Married-civ-spouse", "Divorced", "Never-married", 
                            "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
    valid_occupation = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", 
                        "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", 
                        "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
    valid_relationship = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    valid_race = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    valid_sex = ["Female", "Male"]
    valid_native_country = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", 
                            "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", 
                            "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", 
                            "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", 
                            "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", 
                            "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", 
                            "Peru", "Hong", "Holand-Netherlands"]
    rows_to_keep = [val in valid_workclass for val in dataset["workclass"]]
    dataset = dataset[rows_to_keep]
    rows_to_keep = [val in valid_education for val in dataset["education"]]
    dataset = dataset[rows_to_keep]
    rows_to_keep = [val in valid_marital_status for val in dataset["marital-status"]]
    dataset = dataset[rows_to_keep]
    rows_to_keep = [val in valid_occupation for val in dataset["occupation"]]
    dataset = dataset[rows_to_keep]
    rows_to_keep = [val in valid_relationship for val in dataset["relationship"]]
    dataset = dataset[rows_to_keep]
    rows_to_keep = [val in valid_race for val in dataset["race"]]
    dataset = dataset[rows_to_keep]
    rows_to_keep = [val in valid_sex for val in dataset["sex"]]
    dataset = dataset[rows_to_keep]
    rows_to_keep = [val in valid_native_country for val in dataset["native-country"]]
    dataset = dataset[rows_to_keep]
    size = len(dataset)

    # transform 'sex' and 'salary' columns to binary
    dataset['sex'] = dataset['sex'].map({'Female': 1, 'Male': 0}).astype(int)
    return dataset, size/original_size

def load_german(path):
    dataset = pd.read_csv(path, header=None, delimiter=r'\s+')
    dataset = dataset.rename(columns={12: 'Age', 20: 'credit_status'})

    # transform 'credit_status' column to binary
    dataset['credit_status'] = dataset['credit_status'].map({1: 0, 2: 1}).astype(int)
    return dataset

def encode(dataset_raw, encoders={}):
    dataset = dataset_raw.copy()
    for col, encoder in encoders.items():
        dataset[col] = encoder.fit_transform(dataset[col])
    return dataset

def decode(dataset_encoded, encoders={}):
    dataset = dataset_encoded.copy()
    for col, encoder in encoders.items():
        dataset[col] = encoder.inverse_transform(dataset[col])
    return dataset