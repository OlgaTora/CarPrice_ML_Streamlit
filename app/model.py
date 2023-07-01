import pickle
import pandas as pd
from sklearn import preprocessing


def encoder_cat(data):
    cat_feat = data.select_dtypes(include='object')
    for columns in cat_feat:
        encoder = preprocessing.LabelEncoder()
        encoder = encoder.fit(data[columns])
        data[columns] = encoder.transform(data[columns])
#    encoder = category_encoders.OneHotEncoder(cols=cat_feat)
#    data = encoder.fit_transform(data)
    return data


def load_model(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def open_data(path="app/data/df_cars.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=['selling_price', 'torque'])
    return df


def preprocess_data(df: pd.DataFrame):
    encoder_cat(df)
    return df


def predict(df, path="app/data/model_lgbm.pickle"):
    model = load_model(path)
    prediction = model.predict(df)
    return prediction

