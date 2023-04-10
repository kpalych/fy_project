# Итоговый проект первого года обучения <br /> (курс "Профессия Data Science")
## Тема: Агентство недвижимости

### Каталог *app*:

* Файл *[predict_server.py](https://github.com/kpalych/fy_project/blob/master/app/predict_server.py)* - сервер сервиса предсказаний стоимости домов
* Файл *[test_client.py](https://github.com/kpalych/fy_project/blob/master/app/test_client.py)* - тестовый клиент для проверки работы сервиса (использует валидационный набор данных)

#### Загрузка Docker-образа сервиса предсказания цены недвижимости из репозитория:

`$ docker pull kpalych/fy_project`

#### Запуск сервиса:

`$ docker run -d --rm --name=fyp_container -p=80:80 kpalych/fy_project`

#### Тест сервиса:

`$ cd app`

`$ python test_client.py`