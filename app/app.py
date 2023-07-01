import pandas as pd
import streamlit as st
from PIL import Image
from model import preprocess_data, predict, open_data


def process_main_page():
    show_main_page()
    process_side_bar_inputs()


def show_main_page():
    image = Image.open('app/data/IMG_2887.jpg')
    image = image.resize((250, 250)).rotate(angle=270).effect_spread(3)
    st.set_page_config(
        layout='wide',
        initial_sidebar_state='auto',
        page_title='Car price prediction',
        page_icon=image,
    )

    st.title('Рекомендательный сервис по определению стоимости автомобиля')
    st.header('Попробуй - убедись!')
    st.image(image)
    st.markdown("""
        <style type="text/css">
        [data-testid=stSidebar] {
            background-color: rgb(235, 102, 0);
            color: #FFFFFF;
        }
        [data-testid=stButton] {
            color: rgb(1,1,100)
            }       
        </style>
    """, unsafe_allow_html=True)


def write_user_data(df):
    click = st.button('Ваши данные')
    if click:
        st.write('## Ваши данные')
        st.write(df)


def write_prediction(prediction):
    if prediction > 0:
        st.header('Рекомендуемая цена на данный автомобиль')
        st.write(prediction)
    else:
        st.write('## Введите запрос!')


def process_side_bar_inputs():
    st.sidebar.header('Заданные пользователем параметры')
    user_input_df = sidebar_input_features()
    write_user_data(user_input_df)
    df = open_data()
    df = pd.concat((user_input_df, df), axis=0)
    preprocessed_df = preprocess_data(df)
    user_df = preprocessed_df[:1]
    prediction = predict(user_df)
    write_prediction(prediction)


def sidebar_input_features():
    year = st.sidebar.slider("Год выпуска", min_value=1960, max_value=2023, value=1960, step=1)
    km_driven = st.sidebar.slider("Пробег в тыс км",
                                  min_value=0, max_value=1000, value=0, step=1)
    fuel = st.sidebar.selectbox("Топливо", ("Дизель", "Бензин", "Другое"))
    seller_type = st.sidebar.selectbox("Продавец", ("Частное лицо", "Дилер", "Официальный дилер"))
    transmission = st.sidebar.selectbox("Коробка передач", ("Автомат", "Ручная"))
    owner = st.sidebar.selectbox("Количество собственников", ("Один", "Два", "Три", "Четыре и более", "Тест-драйв"))
    mileage = st.sidebar.slider("Расход топлива", min_value=5, max_value=70, value=5, step=1)
    engine = st.sidebar.slider("Двигатель куб.см", min_value=500, max_value=5000, value=500, step=1)
    max_power = st.sidebar.slider("Мощность в лс", min_value=50, max_value=700, value=50, step=1)
    seats = st.sidebar.slider("Количество мест", min_value=1, max_value=14, value=1, step=1)
    mark = (st.sidebar.text_input("Марка автомобиля")).capitalize()
    model = (st.sidebar.text_input("Модель")).capitalize()

    translation = {
        'Дизель': 'Diesel',
        'Бензин': 'Petrol',
        'Другое': 'Other',
        'Автомат': 'Automatic',
        'Ручная': 'Manual',
        'Один': 'First Owner',
        'Два': 'Second Owner',
        'Три': 'Third Owner',
        'Четыре и более': 'Fourth & Above Owner',
        'Тест-драйв': 'Test Drive Car',
        'Частное лицо': 'Individual',
        'Дилер': 'Dealer',
        'Официальный дилер': 'Trustmark Dealer',
    }

    data = {
        'year': year,
        'km_driven': km_driven,
        'fuel': translation[fuel],
        'seller_type': translation[seller_type],
        'transmission': translation[transmission],
        'owner': translation[owner],
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats,
        'mark': mark,
        'model': model
    }

    df = pd.DataFrame(data, index=[0])
    return df


if __name__ == "__main__":
    process_main_page()

