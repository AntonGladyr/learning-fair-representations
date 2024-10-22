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
    return dataset, size/original_size

def load_german(path):
    dataset = pd.read_csv(path, header=None, delimiter=r'\s+')
    # transform output column to binary
    dataset.iloc[:, -1] = dataset.iloc[:, -1].map({1: 0, 2: 1}).astype(int)
    return dataset

def encode_adult(dataset_raw, dataset_raw_test):
    dataset = dataset_raw.copy()
    dataset_test = dataset_raw_test.copy()
    # transform 'sex' column to binary
    dataset['sex'] = dataset['sex'].map({'Female': 1, 'Male': 0}).astype(int)
    dataset_test['sex'] = dataset_test['sex'].map({'Female': 1, 'Male': 0}).astype(int)
    # 2-Quantile quantization
    continuous_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for i, col in enumerate(continuous_cols):
        median = dataset[col].median()
        # training set
        dataset.loc[dataset[col] < median, col] = 0
        dataset.loc[dataset[col] >= median, col] = 1

        # test set
        dataset_test.loc[dataset_test[col] < median, col] = 0
        dataset_test.loc[dataset_test[col] >= median, col] = 1
    return dataset, dataset_test

def encode_german(dataset_raw, dataset_raw_test):
    dataset = dataset_raw.copy()
    dataset_test = dataset_raw_test.copy()
    # transform 'Age' and column to binary
    age_index = 12
    dataset.loc[dataset[age_index] <= 25, age_index] = 1
    dataset.loc[dataset[age_index] > 25, age_index] = 0

    dataset_test.loc[dataset_test[age_index] <= 25, age_index] = 1
    dataset_test.loc[dataset_test[age_index] > 25, age_index] = 0
    # 2-Quantile quantization
    continuous_cols = [1, 4, 7, 10, 15, 17]
    for i, col in enumerate(continuous_cols):
        median = dataset[col].median()
        # training set
        dataset.loc[dataset[col] < median, col] = 0
        dataset.loc[dataset[col] >= median, col] = 1

        # test set
        dataset_test.loc[dataset_test[col] < median, col] = 0
        dataset_test.loc[dataset_test[col] >= median, col] = 1
    return dataset, dataset_test

def encode_german_all(dataset_raw):
    dataset = dataset_raw.copy()
    # transform 'Age' and column to binary
    age_index = 12
    dataset.loc[dataset[age_index] <= 25, age_index] = 1
    dataset.loc[dataset[age_index] > 25, age_index] = 0
    # 2-Quantile quantization
    continuous_cols = [1, 4, 7, 10, 15, 17]
    for i, col in enumerate(continuous_cols):
        median = dataset[col].median()
        # training set
        dataset.loc[dataset[col] < median, col] = 0
        dataset.loc[dataset[col] >= median, col] = 1
    return dataset