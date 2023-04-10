# Итоговый проект первого года обучения <br /> (курс "Профессия Data Science")
## Тема: Агентство недвижимости

**Что необходимо сделать:** разработать сервис для предсказания стоимости домов на основе истории предложений.

### Описание структуры проекта:

* Каталог *[app](https://github.com/kpalych/fy_project/blob/master/app)* - файлы сервиса предсказаний цены стоимости домов
* Каталог *[model](https://github.com/kpalych/fy_project/blob/master/model)* - анализ, подготовка данных, подбор и обучение модели
* Каталог *[shared_libs](https://github.com/kpalych/fy_project/blob/master/shared_libs)* - библиотечные функции и данные, совместно используемые как на этапе подготовки модели, так и сервисом

### Каталог *app*:

* Файл *[predict_server.py](https://github.com/kpalych/fy_project/blob/master/app/predict_server.py)* - сервер сервиса предсказаний стоимости домов
* Файл *[test_client.py](https://github.com/kpalych/fy_project/blob/master/app/test_client.py)* - тестовый клиент для проверки работы сервиса (использует валидационный набор данных)

### Каталог *model*:

* Каталог *[data](https://github.com/kpalych/fy_project/blob/master/model/data)* - в данном каталоге находятся файлы с исходными данными для обучения модели
    * Файл *data.csv* - исходный набор данных
    * Файл *data_target_cleared.csv* - данные из которого удалены аномальные выбросы целевого значения и без валидационной набора
    * Файл *data_valid.csv* - валидационный набор (без целевого поля)
    * Файл *data_valid_target.csv* - целевое поле валидационного набора данных (для проверки качество предсказания сервиса)
    * Файл *uscities.csv* - данные по городам США (взят из проекта на сервисе kaggle.com ([источник](https://www.kaggle.com/datasets/sergejnuss/united-states-cities-database)))
* Файл *[index.ipynb](https://github.com/kpalych/fy_project/blob/master/model/index.ipynb)* - первичная оценка данных, очистка целевого параметра от выбросов, формирование основного и валидационного даборов данных
* Файл *[base_line.ipynb](https://github.com/kpalych/fy_project/blob/master/model/base_line.ipynb)* - предобработка базовых признаков, очистка от пропусков, формирование Base Line модели
* Файл *[extra_features.ipynb](https://github.com/kpalych/fy_project/blob/master/model/extra_features.ipynb)* - создание дополнительных признаков, обучение финальной модели модели
* Файл *[validate.ipynb](https://github.com/kpalych/fy_project/blob/master/model/validate.ipynb)* - тестирование финальной модели модели на валидационной выборке
* Файл *[population_features.ipynb](https://github.com/kpalych/fy_project/blob/master/model/population_features.ipynb)* - добавление признаков на основании стат. данных по городам США
* Файл *[coord_dicts.ipynb](https://github.com/kpalych/fy_project/blob/master/model/coord_dicts.ipynb)* - формирование кэша с геоданными (города, районы по ZIP коду, адреса объектов недвижимости)
* Файл *[model_selection.ipynb](https://github.com/kpalych/fy_project/blob/master/model/model_selection.ipynb)* - поиск наиболее подходящей модели предсказания

### Каталог *shared_libs*:

* Каталог *[data](https://github.com/kpalych/fy_project/blob/master/shared_libs/data)* - в данном каталоге находятся файлы с данными, используемыми как для обучения модели, так и для работы модели в составе сервиса
    * Каталог *data/models* - каталог, в котором хранятся сериализованные обученные модели
* Файл *[data_transform.py](https://github.com/kpalych/fy_project/blob/master/shared_libs/data_transform.py)* - библиотека с сервисными функциями для работы с данными как на этапе подготоки и обучения модели, так и для работы модели в составе сервиса

#### Загрузка Docker-образа сервиса предсказания цены недвижимости из репозитория:

`$ docker pull kpalych/fy_project`

#### Запуск сервиса:

`$ docker run -d --rm --name=fyp_container -p=80:80 kpalych/fy_project`

#### Тест сервиса:

`$ cd app`

`$ python test_client.py`

### Данные:

Данные для каталога `/model/data` можно скачать по [ссылке](https://disk.ya.ru/zsrytSEHGZDFrgb)

Данные для каталога `/shared_libs/data` можно скачать по [ссылке](https://disk.ya.ru/zsrytSEHGZDFrgb)

