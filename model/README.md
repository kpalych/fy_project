# Итоговый проект первого года обучения <br /> (курс "Профессия Data Science")
## Тема: Агентство недвижимости

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
