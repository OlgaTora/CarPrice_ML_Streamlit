# CarPrice_ML_Streamlit

Запустить приложение можно по ссылке:
https://carprice.streamlit.app/

Данная работа выполнена в рамках буткема от AlEducation.

Задача: создать модель и реализовать ее в Streamlit.

Была проведена следующая работа с данными:
- пропуски/дубликаты
- созданы новые признаки mark и model
- перевести int64 в int8, где это возможно
- закодированы категориальные признаки
- значения в mileage приведены к одному знаменателю
- каждый признак проверен на выбросы и зависимость с целевым

Использовались три модели:
- Linear Regression
- Light GBM
- XGB Regression

Лучший результат показала Light GBM: R2 82.45
