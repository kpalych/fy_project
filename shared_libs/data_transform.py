import numpy as np
import pandas as pd
import locale
import json
import re
import random
import pickle
import requests
import urllib.parse
from geopy.geocoders import Nominatim
from geocodio import GeocodioClient

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

locale.setlocale(locale.LC_ALL, 'en_US.UTF8')

osm_geolocator = Nominatim(user_agent='myapplication')
us_census_geocoder = GeocodioClient('USCensus API key')
Gmaps_API_key = 'GoogleMap API key'

def set_locale(loc_val, loc_type=locale.LC_ALL):
    locale.setlocale(loc_type, loc_val)


def target_str_to_float(value, print_error_value=False):
    """
    Функция конвертации целевой цены недвижимости во float

    Parameters:
    value (str): Исходная цена
    print_error_value (bool): Флаг вывода ошибок в консоль

    Returns:
    float: Цена в виде float или NaN
    """
    
    try:
        result = locale.atof(value.strip("$").strip("+"))
        return result
    except:
        if print_error_value:
            print(value)
        return np.NaN


def drop_not_informative_columns(df, cols):
    return df.drop(cols, axis=1)


def convert_to_json_str(value, noneValue=None):
    """
    Функция конвертации структурированных полей к строке, пригодной для JSON декодирования

    Parameters:
    value (str): Исходное значение
    noneValue (str): паттерн подстановки в качестве None значения

    Returns:
    str: Строка, пригодная для JSON декодирования
    """

    result = (
            str.replace(value, '"', "'")
                .replace("{'", '{"')
                .replace("['", '["')
                .replace("':", '":')
                .replace(": '", ': "')
                .replace("', ", '", ')
                .replace(", '", ', "')
                .replace("'}", '"}')
                .replace("']", '"]')
                .replace(': None,', ': "",' if noneValue is None else ': "'+noneValue+'",')
                .replace('", None, "', '", "", "')
                
        )
    
    return result

def get_subfact(value, sub_fact_label, need_print_error=False):
    """
    Функция выборки элемента из структуры поля homeFacts по имени

    Parameters:
    value (str): Значение поля homeFacts
    sub_fact_label (str): имя поля в структуре
    print_error_value (bool): Флаг вывода ошибок в консоль

    Returns:
    str: Значение поля структуры
    """

    try:
        src_value = value
        value = convert_to_json_str(value)
        facts = json.loads(value)['atAGlanceFacts']
        for fact in facts:
            if fact['factLabel'] == sub_fact_label:
                return np.NaN if fact['factValue'] == '' else fact['factValue']
            
        return np.NaN
    except Exception as e:
        if need_print_error:
            print(src_value)
            print(e)
            
        return np.NaN

def convert_str_to_school_rating(value):
    """
    Функция конвертации рейтинга школы в int

    Parameters:
    value (str): Значение рейтинга

    Returns:
    int: Рейтинг в виде int
    """

    value = value.upper()
    
    if value == '' or value == 'NA' or value == 'NR'  or 'NONE' in value:
        return 0
    
    if '/' in value:
        parts = value.split('/')
        return int(parts[0])
    else:
        return int(value)


def preprocess_grade_str(value):
    """
    Функция подготовки типа школы для послкдующей конвертации в категорию

    Parameters:
    value (str): Значение типа школы

    Returns:
    str: Очищенное значение
    """

    return value.replace(' to ', '-').replace('Preschool', 'PK').upper().replace('PK', '0').replace('K', '1').replace('–', '-')


def get_school_grade_category(grade_indx):
    """
    Функция преобразования индекса типа школы в категорию

    Parameters:
    grade_indx (int): Значение индекса типа школы

    Returns:
    str: Категория типа школы (PK, K, M, H)
    """

    grade_indx = int(grade_indx)
    
    if grade_indx == 0:
        return 'PK'
    elif grade_indx > 0 and grade_indx <= 5:
        return 'K'
    elif grade_indx > 5 and grade_indx <= 8:
        return 'M'
    else:
        return 'H'

def get_school_grades(str_value, pre_results=None):  
    """
    Функция рекурсивной выборки из данных о школе списка типов близлежащих школ

    Parameters:
    str_value (str): Данные о школах
    pre_results(list:str) Список ранее выбранных типов

    Returns:
    list(str): Список уникальных типов школ
    """
    
    str_value = str_value.strip().upper()
    
    if pre_results is None:
        pre_results = []
        
    if str_value == '' or str_value == 'N/A' or str_value == 'NA':
        return pre_results
    
    if ',' in str_value:
        parts = str_value.split(',')
        for part in parts:
            pre_results = get_school_grades(part, pre_results)
    else:
        if '-' in str_value:
            parts = str_value.split('-')
            
            lb = int(parts[0])
            ub = int(parts[1])
            
            for gi in range(lb, ub+1):
                pre_results.append(get_school_grade_category(gi))
        else:
            pre_results.append(get_school_grade_category(str_value))
    
    result = list(set(pre_results))
    result.sort()
    
    return result

def convert_school_distance_str_to_float(str_val):
    """
    Функция преобразования строки с расстоянием до школы в число типа Float

    Parameters:
    str_val (str): Данные о расстоянии

    Returns:
    float: Расстояние в милях
    """
  
    return locale.atof(str.upper(str_val).strip('MI').strip())

def get_schools_count(value, need_print_error=False):
    """
    Функция возвращает количество школ по строке описания близлежащих школ

    Parameters:
    value (str): Строка описания данных о школах

    Returns:
    int: Количество школ
    """
  
    try:
        src_value = value
        
        value = convert_to_json_str(value)
        schools = json.loads(value)

        return len(schools[0]['rating'])
    except Exception as e:
        if need_print_error:
            print(src_value)
            print(e)
            
        return np.NaN
    
def get_schools_avg_rate(value, need_print_error=False):
    """
    Функция вычисления средного рейтинга близлежащих школ

    Parameters:
    value (str): Строка описания данных о школах

    Returns:
    flaot: Средний рейтинг
    """
    
    try:
        src_value = value
        
        value = convert_to_json_str(value)
        schools = json.loads(value)

        all_ratings = []
        for rating in schools[0]['rating']:
            all_ratings.append(convert_str_to_school_rating(rating))

        return 0 if len(all_ratings) == 0 else np.array(all_ratings).mean()
    except Exception as e:
        if need_print_error:
            print(src_value)
            print(e)
            
        return np.NaN

def get_schools_grades_list(value, need_print_error=False):
    """
    Функция возвращает список уникальных типов школ

    Parameters:
    value (str): Строка описания данных о школах

    Returns:
    list(str): Список уникальных типов школ
    """
   
    try:
        src_value = value
        
        value = convert_to_json_str(value)
        schools = json.loads(value)

        all_schools_gr_cats = []
            
        for grade in schools[0]['data']['Grades']:
            grade_val = preprocess_grade_str(grade)
            
            all_schools_gr_cats = get_school_grades(grade_val, all_schools_gr_cats)

        return all_schools_gr_cats
    except Exception as e:
        if need_print_error:
            print(src_value)
            print(e)
            
        return np.NaN

def get_schools_min_distance(value, need_print_error=False):
    """
    Функция возвращает минимальное расстояние до школы

    Parameters:
    value (str): Строка описания данных о школах

    Returns:
    float: Минимальное расстояние в милях
    """
  
    try:
        src_value = value
        
        value = convert_to_json_str(value)
        schools = json.loads(value)

        dist_list = []
        for distance in schools[0]['data']['Distance']:
            dist_list.append(convert_school_distance_str_to_float(distance))

        return 0 if len(dist_list) == 0 else np.array(dist_list).min()
    except Exception as e:
        if need_print_error:
            print(src_value)
            print(e)
            
        return np.NaN

def get_schools_avg_distance(value, need_print_error=False):
    """
    Функция возвращает среднее расстояние до всех ближайших школы

    Parameters:
    value (str): Строка описания данных о школах

    Returns:
    float: Минимальное расстояние в милях
    """
  
    try:
        src_value = value
        
        value = convert_to_json_str(value)
        schools = json.loads(value)

        dist_list = []
        for distance in schools[0]['data']['Distance']:
            dist_list.append(convert_school_distance_str_to_float(distance))

        return 0 if len(dist_list) == 0 else np.array(dist_list).mean()
    except Exception as e:
        if need_print_error:
            print(src_value)
            print(e)
            
        return np.NaN

def convert_sqft_str_to_float(value, need_print_error=False):
    """
    Функция преобразования строки с площадью объекта в число типа float

    Parameters:
    value (str): Строка с площадью объекта

    Returns:
    float: Площадь объекта
    """
  
    try:
        parts = str.strip(value).split(' ')
            
        if len(parts) > 1:
            value = parts[len(parts)-2]
        else:
            value = parts[0]
            
        return locale.atof(value)
    except Exception as e:
        if need_print_error:
            print(value)
            print(e)
            
        return np.NaN

def get_pr_type_features_list():
    """
    Функция возвращает список имен признаков, извлекаемых из поля propertyType

    Returns:
    list(str): Список признаков
    """
  
    return [
        'SINGLE-FAMILY',
        'MULTI-FAMILY',
        'CONDOMINIMUM',
        'LAND',
        'TOWNHOUSE',
        'TRADITIONAL',
        'MODERN',
        'RANCH',
        'DETACHED'
    ]

pr_type_regexps = {
    'SINGLE-FAMILY': [
        r'(^|\W)single-family($|\W)',
        r'(^|\W)Single Family($|\W)'

    ],
    'MULTI-FAMILY': [
        r'(^|\W)multi-family($|\W)',
        r'(^|\W)Multi Family($|\W)'
    ],
    'CONDOMINIMUM': [
        r'(^|\W)condo($|\W)',
        r'(^|\W)Condominium($|\W)'
    ],
    'LAND': [
        r'(^|\W)lot/land($|\W)',
        r'(^|\W)Land($|\W)'
    ],
    'TOWNHOUSE': [
        r'(^|\W)townhouse($|\W)'
    ],
    'TRADITIONAL': [
        r'(^|\W)Traditional($|\W)'
    ],
    'MODERN': [
        r'(^|\W)Contemporary($|\W)',
        r'(^|\W)Modern($|\W)'
    ],
    'RANCH': [
        r'(^|\W)Ranch($|\W)'
    ],
    'DETACHED': [
        r'(^|\W)Single Detached($|\W)',
        r'(^|\W)Detached($|\W)'
    ]
}

def get_pr_type_feature(value, pr_type_feature):
    """
    Функция возвращает бинарное значение признака, извлекаемого из поля propertyType по имени

    Parameters:
    value (str): Значение поля propertyType
    pr_type_feature (str): Имя бинарного признака

    Returns:
    int: Значение бинарного признака
    """
  
    for re_pattern in pr_type_regexps[pr_type_feature]:
        if re.search(re_pattern, value, re.IGNORECASE) is not None:
            return 1
        
    return 0

    
def get_stories_features_list():
    """
    Функция возвращает список имен признаков, извлекаемых из поля stories

    Returns:
    list(str): Список признаков
    """
  
    return [
        '1-STORY',
        '2-STORY',
        '3-STORY',
        'OTHER-STORY'
    ]
    
story_pr_type_regexps = {
    '1-STORY': [
        r'(^|\W)One Story($|\W)',
        r'(^|\W)1 Story($|\W)',
        r'(^|\W)Single Level($|\W)'

    ],
    '2-STORY': [
        r'(^|\W)2 Story($|\W)',
        r'(^|\W)Two Story($|\W)',
        r'(^|\W)2 Stories($|\W)',
        r'(^|\W)Bi-Level($|\W)'
    ],
    '3-STORY': [
        r'(^|\W)3 Story($|\W)',
        r'(^|\W)3 Stories($|\W)',
        r'(^|\W)Tri-Level($|\W)'
    ],
    'OTHER-STORY': [
        r'(^|\W)8\+ Stories($|\W)'
    ]
}

def get_stroty_feature_by_pr_type(value, story_feature):
    """
    Функция возвращает бинарное значение признака, извлекаемого из поля stories по имени

    Parameters:
    value (str): Значение поля stories
    pr_type_feature (str): Имя бинарного признака

    Returns:
    int: Значение бинарного признака
    """

    for re_pattern in story_pr_type_regexps[story_feature]:
        if re.search(re_pattern, value, re.IGNORECASE) is not None:
            return 1
        
    return 0

story_story_regexps = {
    '1-STORY': [
        r'(^|\W)1 Story($|\W)',
        r'(^|\W)One Story($|\W)',
        r'(^|\W)1 Level($|\W)',
        r'(^|\W)One Level($|\W)',
        r'(^|\W)One($|\W)'

    ],
    '2-STORY': [
        r'(^|\W)2 Story($|\W)',
        r'(^|\W)2 Stories($|\W)',
        r'(^|\W)2 Level($|\W)',
        r'(^|\W)1.5 Story($|\W)',
        r'(^|\W)1.5 Level($|\W)',
        r'(^|\W)1.5 Stories($|\W)',
        r'(^|\W)Two($|\W)',
        r'(^|\W)Bi-Level($|\W)',
        r'(^|\W)1-2 Stories($|\W)'
    ],
    '3-STORY': [
        r'(^|\W)3 Story($|\W)',
        r'(^|\W)Tri-Level($|\W)',
        r'(^|\W)Tri Level($|\W)',
        r'(^|\W)2 Or More Stories($|\W)',
        r'(^|\W)3 Level($|\W)',
        r'(^|\W)3 Stories($|\W)'
    ],
    'OTHER-STORY': [
        r'(^|\W)4 Story($|\W)',
        r'(^|\W)Fourplex($|\W)',
        r'(^|\W)Three Or More($|\W)',
        r'(^|\W)3\+ Story($|\W)',
        r'(^|\W)3\+($|\W)'
    ]
}


def get_story_count_by_story(value):
    """
    Функция возвращает число этажей, извлекаемое из поля stories

    Parameters:
    value (str): Значение поля stories

    Returns:
    int: Число этажей (1, 2, 3, 4)
    """

    try:
        result = int(round(locale.atof(value)))
    except:
        result = None
        
    if result is not None:
        if result <= 0:
            result = 1
            
        return result
    
    f_list = get_stories_features_list()
    cnt = 1
    for f in f_list:
        for re_pattern in story_story_regexps[f]:
            if re.search(re_pattern, value, re.IGNORECASE) is not None:
                return cnt
            
        cnt += 1    
    
    return np.NaN

def clear_beds_from_sqr(value):
    """
    Функция проверяет наличие в строке из поля beds информации о площади объекта

    Parameters:
    value (str): Значение поля beds

    Returns:
    str: исходное значение или NaN
    """

    if re.search(r'(^|\W)acres($|\W)', value, re.IGNORECASE) is not None:
        return np.NaN
    if re.search(r'(^|\W)sqft($|\W)', value, re.IGNORECASE) is not None:
        return np.NaN
    
    return value

def convert_first_word_to_int(value):
    """
    Функция переводит первое слово из строки в число типа int

    Parameters:
    value (str): Строка

    Returns:
    int: число или NaN
    """
    
    value = str.strip(value)
    parts = value.split(' ')
    
    if len(parts) > 0:
        try:
            result = int(round(locale.atof(parts[0])))
            
            return result
        except:
            return np.NaN
    else:
        return np.NaN

def convert_last_word_to_int(value):
    """
    Функция переводит последнее слово из строки в число типа int

    Parameters:
    value (str): Строка

    Returns:
    int: число или NaN
    """

    value = str.strip(value)
    parts = value.split(' ')
    
    if len(parts) > 0:
        try:
            result = int(round(locale.atof(parts[len(parts)-1])))
            
            return result
        except:
            return np.NaN
    else:
        return np.NaN

def convert_beds_str_to_int(value):
    """
    Функция переводит поле beds в число типа int (берется первое слово строки)

    Parameters:
    value (str): Значение поля beds

    Returns:
    int: число или NaN
    """
    
    return convert_first_word_to_int(value)


def get_beds_features_list():
    """
    Функция возвращает список имен признаков, извлекаемых из поля beds

    Returns:
    list(str): Список признаков
    """

    return [
        '1-BEDROOM',
        '2-BEDROOMS',
        '3-BEDROOMS',
        '4-BEDROOMS',
        '5-BEDROOMS',
        '6-BEDROOMS',
        'OTHER-BEDROOMS'
    ]

def convert_baths_str_to_int(value):
    """
    Функция переводит поле baths в число типа int (берется первое или последнее слово строки)

    Parameters:
    value (str): Значение поля beds

    Returns:
    int: число или NaN
    """

    if re.search(r'(^|\W)Bathrooms:($|\W)', value, re.IGNORECASE) is not None:
        return convert_last_word_to_int(value)
    else:
        return convert_first_word_to_int(value)

def get_bathrooms_features_list():
    """
    Функция возвращает список имен признаков, извлекаемых из поля bathrooms

    Returns:
    list(str): Список признаков
    """

    return [
        '1-BATHROOM',
        '2-BATHROOMS',
        '3-BATHROOMS',
        '4-BATHROOMS',
        '5-BATHROOMS',
        '6-BATHROOMS',
        'OTHER-BATHROOMS'
    ]

#################################################################################################################################
# Группа функций, предназначенные для генерации случайного года, с соблюдением первоначально распределения в датасете
#################################################################################################################################

def get_nunuques_size(df, col_name, percent=50, start_n=10, step=1):
    result=start_n
    while True:
        cover = (df[col_name].value_counts() / df[~df[col_name].isna()].shape[0] * 100)[:result].sum()
        if cover >= percent:
            return result
        
        result += step
        
def get_weighted_nuniques_list(df, col_name, size, mul=10):
    subset = (df[col_name].value_counts() / df[~df[col_name].isna()].shape[0] * 100)[:size]
    weighted_values = []
    total = 0
    for indx, weight in np.round(subset * mul, 0).items():
        weighted_values.append({ 'value': indx, 'weight': int(weight) })
        total += weight
        
    weighted_list = []
    for item in weighted_values:
        for i in range(item['weight']):
            weighted_list.append(item['value'])
        
    return weighted_list

def get_weighted_random_item(weighted_list):
    return random.choice(weighted_list)

#################################################################################################################################
#################################################################################################################################
#################################################################################################################################

def get_cooling_features_list():
    """
    Функция возвращает список имен признаков, извлекаемых из поля cooling

    Returns:
    list(str): Список признаков
    """

    return [
        'CENTRAL_COOLING',
        'COOLING',
        'HEATING',
        'GAS',
        'ELECTRIC',
        'ZONED',
        'HEAT_PUMP',
        'WALL'
    ]
    
cooling_regexps = {
    'CENTRAL_COOLING': [
        r'(^|\W)Central($|\W)'
    ],
    'COOLING': [
        r'(^|\W)A/C($|\W)',
        r'(^|\W)AC($|\W)',
        r'(^|\W)Cooling($|\W)',
        r'(^|\W)Air Conditioning($|\W)',
        r'(^|\W)Air($|\W)',
        r'(^|\W)Refrigeration($|\W)'
    ],
    'HEATING': [
        r'(^|\W)Heating($|\W)'
    ],
    'GAS': [
        r'(^|\W)Gas($|\W)'
    ],
    'ELECTRIC': [
        r'(^|\W)Electric($|\W)'
    ],
    'ZONED': [
        r'(^|\W)Zoned($|\W)'
    ],
    'HEAT_PUMP': [
        r'(^|\W)Heat Pump($|\W)'
    ],
    'WALL': [
        r'(^|\W)Wall($|\W)'
    ]    
}

def get_cooling_feature(value, cooling_feature):
    """
    Функция возвращает бинарное значение признака, извлекаемого из поля cooling по имени

    Parameters:
    value (str): Значение поля stories
    pr_type_feature (str): Имя бинарного признака

    Returns:
    int: Значение бинарного признака
    """

    for re_pattern in cooling_regexps[cooling_feature]:
        if re.search(re_pattern, value, re.IGNORECASE) is not None:
            return 1
        
    return 0

def get_heating_features_list():
    """
    Функция возвращает список имен признаков, извлекаемых из поля heating

    Returns:
    list(str): Список признаков
    """

    return [
        'FORCED_AIR_HEAT',
        'OTHER_HEAT',
        'ELECTRIC',
        'GAS',
        'COOLING',
        'AIR_HEAT',
        'HEAT_PUMP',
        'CENTRAL_HEAT',
        'BASE_BOARD',
        'WALL',
        'ZONED',
        'HEATING'
    ]
    
heating_regexps = {
    'FORCED_AIR_HEAT': [
        r'(^|\W)Forced Air($|\W)'
    ],
    'OTHER_HEAT': [
        r'(^|\W)Other($|\W)'
    ],
    'ELECTRIC': [
        r'(^|\W)Central Electric($|\W)',
        r'(^|\W)Electric($|\W)',
    ],
    'GAS': [
        r'(^|\W)Gas($|\W)'
    ],
    'COOLING': [
        r'(^|\W)Air Conditioning($|\W)'
    ],
    'AIR_HEAT': [
        r'(^|\W)Air($|\W)'
    ],
    'HEAT_PUMP': [
        r'(^|\W)Heat Pump($|\W)'
    ],
    'CENTRAL_HEAT': [
        r'(^|\W)Central($|\W)'
    ],
    'BASE_BOARD': [
        r'(^|\W)Baseboard($|\W)'
    ],
    'WALL': [
        r'(^|\W)Wall($|\W)'
    ],
    'ZONED': [
        r'(^|\W)Zoned($|\W)'
    ],
    'HEATING': [
        r'(^|\W)Heating($|\W)'
    ],
}

def get_heating_feature(value, heating_feature):
    """
    Функция возвращает бинарное значение признака, извлекаемого из поля heating по имени

    Parameters:
    value (str): Значение поля stories
    pr_type_feature (str): Имя бинарного признака

    Returns:
    int: Значение бинарного признака
    """

    for re_pattern in heating_regexps[heating_feature]:
        if re.search(re_pattern, value, re.IGNORECASE) is not None:
            return 1
        
    return 0

def get_parking_features_list():
    """
    Функция возвращает список имен признаков, извлекаемых из поля parking

    Returns:
    list(str): Список признаков
    """

    return [
        'GARAGE',
        'ATTACHED',
        'DETACHED',
        'CARPOPT',
        'OFF_STREET',
        'ON_STREET',
        'PARKING',
        '1_PLACE',
        '2_PLACE',
        '3_PLACE',
        'MORE_PLACES'
    ]
    
parking_regexps = {
    'GARAGE': [
        r'(^|\W)Garage($|\W)'
    ],
    'ATTACHED': [
        r'(^|\W)Attached($|\W)'
    ],
    'DETACHED': [
        r'(^|\W)Detached($|\W)'
    ],
    'CARPOPT': [
        r'(^|\W)Carport($|\W)'
    ],
    'OFF_STREET': [
        r'(^|\W)Off Street($|\W)'
    ],
    'ON_STREET': [
        r'(^|\W)On Street($|\W)'
    ],
    'PARKING': [
        r'(^|\W)Parking($|\W)'
    ],
    '1_PLACE': [
        r'(^|\W)1 space($|\W)',
        r'(^|\W)1 Car($|\W)',
        r'^1$',
    ],
    '2_PLACE': [
        r'(^|\W)2 spaces($|\W)',
        r'(^|\W)2 Car($|\W)',
        r'^2$',
    ],
    '3_PLACE': [
        r'(^|\W)3 spaces($|\W)',
        r'^3$',
    ],
    'MORE_PLACES': [
        r'(^|\W)4 spaces($|\W)',
        r'(^|\W)5 spaces($|\W)',
        r'(^|\W)8 spaces($|\W)',
        r'(^|\W)7 spaces($|\W)',
        r'(^|\W)9 spaces($|\W)',
        r'(^|\W)10 spaces($|\W)',
        r'^4$',
        r'^5$',
        r'^6$',
        r'^8$',
    ],

}

def get_parking_feature(value, parking_feature):
    """
    Функция возвращает бинарное значение признака, извлекаемого из поля parking по имени

    Parameters:
    value (str): Значение поля parking
    pr_type_feature (str): Имя бинарного признака

    Returns:
    int: Значение бинарного признака
    """

    for re_pattern in parking_regexps[parking_feature]:
        if re.search(re_pattern, value, re.IGNORECASE) is not None:
            return 1
        
    return 0

def convert_to_ord_cat(val, min_val, max_val, n_cats):
    """
    Функция конвертирует числовое значение в порядковую категорию

    Parameters:
    val (number): Исходное значение
    min_val (number): Начало диапазона
    max_val (number): Конец диапазона
    n_cats (int): Число отрезко разбиения

    Returns:
    int: Номер отрезка внутри диапазона от min_val до max_val
    """

    step = (max_val - min_val) / n_cats
    
    cur_val = min_val
    cur_cat = 0
    
    while True:
        if cur_val >= val or cur_val >= max_val:
            break
        
        cur_val += step
        cur_cat += 1
        
    return cur_cat


def get_city_dict_key(state, city):
    """
    Функция возвращает значение ключа в кэш-словаре городов

    Parameters:
    state (str): Название штата
    city (str): Название города

    Returns:
    str: Значение ключа
    """

    return state + ', ' + city

def get_city_feature(state, city, dict_field, cities_dict):
    """
    Функция возвращает значение признака из кэш-словаря городов по имени признака

    Parameters:
    state (str): Название штата
    city (str): Название города
    dict_field (str): Имя признака
    cities_dict (dictionary): Словарь городов

    Returns:
    any: Значение признака
    """

    key = get_city_dict_key(state, city)
    
    if key not in cities_dict:
        return np.NaN
    
    city_item = cities_dict[key]
    
    if dict_field == 'type':
        return city_item['type']
    elif dict_field == 'importance':
         return city_item['importance']
    elif dict_field == 'boundingbox':
        return [float(x) for x in city_item['boundingbox']]
    elif dict_field == 'lat':
        return float(city_item['lat'])
    elif dict_field == 'lng':
        return float(city_item['lng'])
    else:
        return np.NaN

def get_city_sqr_by_boundingbox(boundingbox):
    """
    Функция возвращает значение псевдо площади по boundingbox

    Parameters:
    boundingbox (list): boundingbox

    Returns:
    float: Значение псевдо площади
    """

    return (max(boundingbox[0], boundingbox[1]) - min(boundingbox[0], boundingbox[1])) * (max(boundingbox[2], boundingbox[3]) - min(boundingbox[2], boundingbox[3]))

def get_address_dict_key(state, city, street):
    """
    Функция возвращает значение ключа в кэш-словаре адресов

    Parameters:
    state (str): Название штата
    city (str): Название города
    street (str): Адрес

    Returns:
    str: Значение ключа
    """

    return str(state) + ', ' + str(city) + ', ' + str(street)

def get_address_zip_dict_key(state, city, zipcode):
    """
    Функция возвращает значение ключа в кэш-словаре почтовых индексов

    Parameters:
    state (str): Название штата
    city (str): Название города
    zipcode (str): Почтовый индекс

    Returns:
    str: Значение ключа
    """

    return str(state) + ', ' + str(city) + ', ' + str(zipcode)

def get_address_location_info(address, api_key, print_error=False):
    """
    Функция возвращает структуру геоинформации по адресу, используя GoogleMaps

    Parameters:
    address (str): Адрес объекта
    api_key (str): API key GMaps
    print_error (bool): Флаг вывода ошибки в консоль

    Returns:
    dictionary: Структура геоинформации или None
    """

    try:
        resp_json_payload = None
        
        params = {
            'address': address,
            'key': api_key
        }
        response = requests.get('https://maps.googleapis.com/maps/api/geocode/json?'+urllib.parse.urlencode(params))

        resp_json_payload = response.json()
                
        loc_rec = {
            'location': resp_json_payload['results'][0]['geometry']['location'],
            'location_type': resp_json_payload['results'][0]['geometry']['location_type'],
            'types': resp_json_payload['results'][0]['types']
        }
        
        return loc_rec
    except Exception as ex:
        if print_error:
            print(ex)
            print(address)
            print(resp_json_payload)
        return None
    
def get_address_location_info_by_osm(address, print_error=False):
    """
    Функция возвращает структуру геоинформации по адресу, используя OpenStreetMAp

    Parameters:
    address (str): Адрес объекта
    print_error (bool): Флаг вывода ошибки в консоль

    Returns:
    dictionary: Структура геоинформации или None
    """

    try:
        location = osm_geolocator.geocode(address)
        loc_rec = {
            'location': {'lat': float(location.raw['lat']), 'lng': float(location.raw['lon'])},
            'location_type': location.raw['type'],
            'types': []
        }
        
        return loc_rec
    except Exception as ex:
        if print_error:
            print(ex)
            print(address)
            try:
                print(location)
            except:
                pass
        return None

def get_address_location_info_by_us_census(street, city, state, zipcode, print_error=False):
    """
    Функция возвращает структуру геоинформации по адресу, используя USCensus

    Parameters:
    street (str): Адрес
    state (str): Название штата
    city (str): Название города
    zipcode (str): Почтовый индекс
    print_error (bool): Флаг вывода ошибки в консоль

    Returns:
    dictionary: Структура геоинформации или None
    """

    try:
        comps_data=[
            {
                'street': street,
                'city': city,
                'state': state,
                'zipcode': zipcode
            }
        ]
        location = us_census_geocoder.geocode(components_data=comps_data)
        loc_rec = {
            'location': {'lat': float(location.coords[0][0]), 'lng': float(location.coords[0][1])},
            'location_type': '',
            'types': []
        }
        
        return loc_rec
    except Exception as ex:
        if print_error:
            print(ex)
            print(street, city, state, zipcode)
            try:
                print(location)
            except:
                pass
        return None

def get_zipcode_location_info_by_us_census(city, state, zipcode, print_error=False):
    """
    Функция возвращает структуру геоинформации по адресу (с точностью до почтового индекса), используя USCensus

    Parameters:
    city (str): Название города
    state (str): Название штата
    zipcode (str): Почтовый индекс
    print_error (bool): Флаг вывода ошибки в консоль

    Returns:
    dictionary: Структура геоинформации или None
    """

    try:
        comps_data=[
            {
                'city': city,
                'state': state,
                'zipcode': zipcode
            }
        ]
        location = us_census_geocoder.geocode(components_data=comps_data)
        loc_rec = {
            'location': {'lat': float(location.coords[0][0]), 'lng': float(location.coords[0][1])},
            'location_type': '',
            'types': []
        }
        
        return loc_rec
    except Exception as ex:
        if print_error:
            print(ex)
            print(city, state, zipcode)
            try:
                print(location)
            except:
                pass
        return None

def clear_street_from_house_number(street):
    street = re.sub('(^|\W)[#0-9]+($|\W)', r'\2', street).strip()
    street = re.sub('\(.*\)', '', street)
    street = re.sub('  ', ' ', street)
    
    return street


def get_center_distance(state, city, street, address_dict, city_lat, city_lng, city_bb, zipcode=None, address_by_zip_dict=None):
    """
    Функция возвращает значение псевдо расстояние (нормированное относительно габаритов города) до центра города

    Parameters:
    state (str): Название штата
    city (str): Название города
    street (str): Адрес
    address_dict (dictionary): Словарь с геоинформацией по адресам
    city_lat (float): Широта центра города
    city_lng (float): Долгото центра города
    city_bb (list): BoundingBox города
    zipcode (str): Почтовый индекс
    address_by_zip_dict (dictionary): Словарь с геоинформацией по почтовым индексам

    Returns:
    float: Значение псевдо расстояния
    """

    key = get_address_dict_key(state, city, street)
    if key not in address_dict:
        if zipcode is not None and address_by_zip_dict is not None:
            key = get_address_zip_dict_key(state, city, zipcode)
            if key not in address_by_zip_dict:
                return 0.5
        else:
            return 0.5
    
    if key in address_dict:
        lat = float(address_dict[key]['location']['lat'])
        lng = float(address_dict[key]['location']['lng'])
    else:
        lat = float(address_by_zip_dict[key]['location']['lat'])
        lng = float(address_by_zip_dict[key]['location']['lng'])
    
    size_lat = city_bb[1] - city_bb[0]
    size_lng = city_bb[3] - city_bb[2]
    
    result = ((lat - city_lat)**2 / size_lat**2 + (lng - city_lng)**2 / size_lng**2)**0.5
    if result > 1:
        result = 1
    
    return result

def get_hight_price_distance(state, city, street, address_dict, cities_clusters_dict, city_bb, zipcode=None, address_by_zip_dict=None):
    """
    Функция возвращает значение псевдо расстояние (нормированное относительно габаритов города) до кластера с дорогой недвижимостью

    Parameters:
    state (str): Название штата
    city (str): Название города
    street (str): Адрес
    address_dict (dictionary): Словарь с геоинформацией по адресам
    cities_clusters_dict (dictionary): Словарь с геоинформацией по ценовым кластерам в каждом городе
    city_bb (list): BoundingBox города
    zipcode (str): Почтовый индекс
    address_by_zip_dict (dictionary): Словарь с геоинформацией по почтовым индексам

    Returns:
    float: Значение псевдо расстояния
    """

    key = get_address_dict_key(state, city, street)
    city_key = get_city_dict_key(state, city)
    
    if city_key not in cities_clusters_dict:
        return 0.5
    
    if key not in address_dict:
        if zipcode is not None and address_by_zip_dict is not None:
            key = get_address_zip_dict_key(state, city, zipcode)
            if key not in address_by_zip_dict:
                return 0.5
        else:
            return 0.5

    if key in address_dict:
        lat = float(address_dict[key]['location']['lat'])
        lng = float(address_dict[key]['location']['lng'])
    else:
        lat = float(address_by_zip_dict[key]['location']['lat'])
        lng = float(address_by_zip_dict[key]['location']['lng'])
    
    city_lat = cities_clusters_dict[city_key]['hight']['lat']
    city_lng = cities_clusters_dict[city_key]['hight']['lng']
    
    size_lat = city_bb[1] - city_bb[0]
    size_lng = city_bb[3] - city_bb[2]
    
    result = ((lat - city_lat)**2 / size_lat**2 + (lng - city_lng)**2 / size_lng**2)**0.5
    if result > 1:
        result = 1
    
    return result

def get_low_price_distance(state, city, street, address_dict, cities_clusters_dict, city_bb, zipcode=None, address_by_zip_dict=None):
    """
    Функция возвращает значение псевдо расстояние (нормированное относительно габаритов города) до кластера с дешовой недвижимостью

    Parameters:
    state (str): Название штата
    city (str): Название города
    street (str): Адрес
    address_dict (dictionary): Словарь с геоинформацией по адресам
    cities_clusters_dict (dictionary): Словарь с геоинформацией по ценовым кластерам в каждом городе
    city_bb (list): BoundingBox города
    zipcode (str): Почтовый индекс
    address_by_zip_dict (dictionary): Словарь с геоинформацией по почтовым индексам

    Returns:
    float: Значение псевдо расстояния
    """

    key = get_address_dict_key(state, city, street)
    city_key = get_city_dict_key(state, city)
    
    if city_key not in cities_clusters_dict:
        return 0.5
    
    if key not in address_dict:
        if zipcode is not None and address_by_zip_dict is not None:
            key = get_address_zip_dict_key(state, city, zipcode)
            if key not in address_by_zip_dict:
                return 0.5
        else:
            return 0.5
    
    if key in address_dict:
        lat = float(address_dict[key]['location']['lat'])
        lng = float(address_dict[key]['location']['lng'])
    else:
        lat = float(address_by_zip_dict[key]['location']['lat'])
        lng = float(address_by_zip_dict[key]['location']['lng'])
    
    city_lat = cities_clusters_dict[city_key]['low']['lat']
    city_lng = cities_clusters_dict[city_key]['low']['lng']
    
    size_lat = city_bb[1] - city_bb[0]
    size_lng = city_bb[3] - city_bb[2]
    
    result = ((lat - city_lat)**2 / size_lat**2 + (lng - city_lng)**2 / size_lng**2)**0.5
    if result > 1:
        result = 1
    
    return result

def get_subset_mean_location(df, state, city, percentile, cities_dict, address_dict, address_by_zip_dict=None):
    """
    Функция возвращает структуру, описывающую ценовой кластер для данного города

    Parameters:
    df (DataFrame): датасет
    state (str): Название штата
    city (str): Название города
    percentile (float): перцентиль отбора в кластеры
    cities_dict (dictionary): Словарь с геоинформацией по городам
    address_dict (dictionary): Словарь с геоинформацией по адресам

    Returns:
    dictionary: Структура, описывающая ценовой кластер для данного города
    """

    city_key = get_city_dict_key(state, city)
    city_mask = (df['state'] == state) & (df['city'] == city)
    percentile_25 = df.loc[city_mask, 'target'].quantile(percentile/100)
    percentile_75 = df.loc[city_mask, 'target'].quantile(1 - percentile/100)
    
    low_mask = (df['state'] == state) & (df['city'] == city) & (df['target'] <= percentile_25)
    hight_mask = (df['state'] == state) & (df['city'] == city) & (df['target'] >= percentile_75)
    
    lats_low = []
    lngs_low = []
    lats_hi = []
    lngs_hi = []
    
    for _, rec in df[low_mask].iterrows():
        address_key = get_address_dict_key(state, city, rec['street'])
        if address_key in address_dict:
            lats_low.append(float(address_dict[address_key]['location']['lat']))
            lngs_low.append(float(address_dict[address_key]['location']['lng']))
        else:
            if address_by_zip_dict is not None:
                zip_key = get_address_zip_dict_key(state, city, rec['zipcode'])
                if zip_key in address_by_zip_dict:
                    lats_low.append(float(address_by_zip_dict[zip_key]['location']['lat']))
                    lngs_low.append(float(address_by_zip_dict[zip_key]['location']['lng']))
            

    for _, rec in df[hight_mask].iterrows():
        address_key = get_address_dict_key(state, city, rec['street'])
        if address_key in address_dict:
            lats_hi.append(float(address_dict[address_key]['location']['lat']))
            lngs_hi.append(float(address_dict[address_key]['location']['lng']))
        else:
            if address_by_zip_dict is not None:
                zip_key = get_address_zip_dict_key(state, city, rec['zipcode'])
                if zip_key in address_by_zip_dict:
                    lats_low.append(float(address_by_zip_dict[zip_key]['location']['lat']))
                    lngs_low.append(float(address_by_zip_dict[zip_key]['location']['lng']))
    
    if len(lats_low) == 0:
        lat_low = float(cities_dict[city_key]['lat'])
        lng_low = float(cities_dict[city_key]['lng'])
    else:
        lat_low = np.median(np.array(lats_low))
        lng_low = np.median(np.array(lngs_low))
    
    if len(lats_hi) == 0:
        lat_hi = float(cities_dict[city_key]['lat'])
        lng_hi = float(cities_dict[city_key]['lng'])
    else:
        lat_hi = np.median(np.array(lats_hi))
        lng_hi = np.median(np.array(lngs_hi))
    
    return {
        'low': {
            'lat': lat_low,
            'lng': lng_low
        },
        'hight': {
            'lat': lat_hi,
            'lng': lng_hi
        }
    }

##################################################################################################
#
# Группа функция, имплементирующая высокоуровневую обработку данных 
# перед обучением/предсказанием модели
# 
# (описание методики находится в соответствующих ноутбуках, например: /model/base_line.ipynb)
#
##################################################################################################

stored_default_values = None

facts = [
    {'col_name': 'fact_year_built', 'col_value': 'Year built'},
    {'col_name': 'fact_remodeled_year', 'col_value': 'Remodeled year'},
    {'col_name': 'fact_cooling', 'col_value': 'Cooling'},
    {'col_name': 'fact_heating', 'col_value': 'Heating'},
    {'col_name': 'fact_parking', 'col_value': 'Parking'}
]

city_by_zip_state = {
    '32686_FL': 'Reddick',
    '32668_FL': 'Morriston',
    '78045_TX': 'Laredo',
    '34474_FL': 'Ocala',
    '34432_FL': 'Dunnellon',
    '34741_FL': 'Kissimmee',
    '38732_MS': 'Cleveland',
    '34747_FL': 'Kissimmee',
    '34744_FL': 'Kissimmee',
    '33126_FL': 'Miami',
    '77032_TX': 'Houston'
}

base_year = 2023

top_cities_slice_size = 30

##################################################################################################

def found_city_by_zip_state(zipcode, state):
    key = zipcode+'_'+state
    return np.NaN if key not in city_by_zip_state else city_by_zip_state[key]

def read_default_values(default_values_file_name, force_read=False):
    global stored_default_values

    if force_read or stored_default_values is None:
        try:
            with open(default_values_file_name, 'rb') as f:
                stored_default_values = pickle.load(f)
        except:
            stored_default_values = {}            
            save_default_values(default_values_file_name)

def save_default_values(default_values_file_name):
    global stored_default_values
    
    if stored_default_values is None:
        stored_default_values = {}
        
    with open(default_values_file_name, 'wb') as f:
        pickle.dump(stored_default_values, f)
        
def get_default_values():
    global stored_default_values
    
    return stored_default_values
    

##################################################################################################

def clear_data_base_line(df, default_values_file_name, can_drop_rows=False, force_rebuild_cached_data=False):
    global stored_default_values
    
    read_default_values(default_values_file_name, force_read=True)
    
    df = drop_not_informative_columns(df, ['status', 'private pool', 'fireplace', 'mls-id', 'PrivatePool'])
    
    for fact in facts:
        df[fact['col_name']] = df['homeFacts'].apply(lambda x: get_subfact(x, fact['col_value']))
        
    df = df.drop('homeFacts', axis=1)
    
    df['schools_count'] = df['schools'].apply(lambda x: get_schools_count(x))
    df['schools_avg_rate'] = df['schools'].apply(lambda x: get_schools_avg_rate(x))
    df['schools_min_distance'] = df['schools'].apply(lambda x: get_schools_min_distance(x))
    df['schools_avg_distance'] = df['schools'].apply(lambda x: get_schools_avg_distance(x))
    df['schools_grades_list'] = df['schools'].apply(lambda x: get_schools_grades_list(x))

    df['schools_PK'] = df['schools_grades_list'].apply(lambda x: 1 if 'PK' in x else 0)
    df['schools_K'] = df['schools_grades_list'].apply(lambda x: 1 if 'K' in x else 0)
    df['schools_M'] = df['schools_grades_list'].apply(lambda x: 1 if 'M' in x else 0)
    df['schools_H'] = df['schools_grades_list'].apply(lambda x: 1 if 'H' in x else 0)

    df = df.drop(['schools_grades_list', 'schools'], axis=1)
    
    city_mask = df['city'].isna()
    df.loc[city_mask, 'city'] = df[city_mask].apply(lambda x: found_city_by_zip_state(x['zipcode'], x['state']), axis=1)
       
    if can_drop_rows:
        df = df.dropna(subset=['city'], axis=0)
    else:
        df['city'].fillna('--', inplace=True)
    
    if can_drop_rows:
        df = df.dropna(subset=['street'], axis=0)
    
    df['city'] = df['city'].apply(lambda x: str.lower(x))
    
    mask = ~df['sqft'].isna()
    df.loc[mask, 'sqft_fl'] = df[mask]['sqft'].apply(lambda x: convert_sqft_str_to_float(x))
    
    sqft_fill_value = df['sqft_fl'].median()
    df['sqft_fl'].fillna(sqft_fill_value, inplace=True)

    pr_type_fiil_value = df['propertyType'].mode()[0]
    df['propertyType'].fillna(pr_type_fiil_value, inplace=True)

    if can_drop_rows:
        stored_default_values['sqft_median'] = df[df['sqft_fl'] < 15000]['sqft_fl'].median()        
        save_default_values(default_values_file_name)

        df = df[df['sqft_fl'] < 15000].copy()
    else:
        mask = df['sqft_fl'] >= 15000
        df.loc[mask, 'sqft_fl'] = stored_default_values['sqft_median']
    
    df = df.drop('sqft', axis=1)

    df['has_mls_id'] = 0
    mask = ~df['MlsId'].isna()
    df.loc[mask, 'has_mls_id'] = 1

    df = df.drop('MlsId', axis=1)

    pr_type_features = get_pr_type_features_list()

    mask = ~df['propertyType'].isna()

    for pr_type_feature in pr_type_features:
        df[pr_type_feature] = 0
        df.loc[mask, pr_type_feature] = df[mask]['propertyType'].apply(lambda x: get_pr_type_feature(x, pr_type_feature))
       
    story_features = get_stories_features_list()

    mask = ~df['propertyType'].isna()

    for story_feature in story_features:
        df[story_feature] = 0
        df.loc[mask, story_feature] = df[mask]['propertyType'].apply(lambda x: get_stroty_feature_by_pr_type(x, story_feature))
       
    mask = ~df['stories'].isna()
    df.loc[mask, 'stories_int'] = df[mask]['stories'].apply(get_story_count_by_story)
    
    story_features = get_stories_features_list()
    st_count = 1

    mask = ~df['stories_int'].isna()
    for st_feature in story_features:
        if st_count < len(story_features):
            df.loc[mask, st_feature] = df[mask]['stories_int'].apply(lambda x: 1 if st_count == x else 0)
        else:
            df.loc[mask, st_feature] = df[mask]['stories_int'].apply(lambda x: 1 if st_count <= x else 0)

        st_count += 1
    
    df = df.drop(['propertyType', 'stories', 'stories_int'], axis=1)
    
    mask = ~df['beds'].isna()
    df.loc[mask, 'beds'] = df[mask]['beds'].apply(clear_beds_from_sqr)    

    mask = ~df['beds'].isna()
    df.loc[mask, 'beds_int'] = df[mask]['beds'].apply(convert_beds_str_to_int)
    
    beds_features = get_beds_features_list()
    beds_count = 1

    mask = ~df['beds_int'].isna()
    for bed_feature in beds_features:
        df[bed_feature] = 0
        if beds_count < len(beds_features):
            df.loc[mask, bed_feature] = df[mask]['beds_int'].apply(lambda x: 1 if beds_count == x else 0)
        else:
            df.loc[mask, bed_feature] = df[mask]['beds_int'].apply(lambda x: 1 if beds_count <= x else 0)

        beds_count += 1
    
    df = df.drop(['beds_int', 'beds'], axis=1)
    
    mask = ~df['baths'].isna()
    df.loc[mask, 'baths_int'] = df[mask]['baths'].apply(convert_baths_str_to_int)
    
    baths_features = get_bathrooms_features_list()
    baths_count = 1

    mask = ~df['baths_int'].isna()
    for bath_feature in baths_features:
        df[bath_feature] = 0
        if baths_count < len(baths_features):
            df.loc[mask, bath_feature] = df[mask]['baths_int'].apply(lambda x: 1 if baths_count == x else 0)
        else:
            df.loc[mask, bath_feature] = df[mask]['baths_int'].apply(lambda x: 1 if baths_count <= x else 0)

        baths_count += 1
    
    df = df.drop(['baths_int', 'baths'], axis=1)
    
    df['was_remodeled'] = 0

    mask = ~df['fact_remodeled_year'].isna()
    df.loc[mask, 'was_remodeled'] = df[mask]['fact_remodeled_year'].apply(lambda x: 1)
    
    df = df.drop('fact_remodeled_year', axis=1)

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            if 'years_distr_subset' not in stored_default_values:
                need_rebuil_data = True
            else:
                years_distr_subset = stored_default_values['years_distr_subset']
        else:
            years_distr_subset = stored_default_values['years_distr_subset']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        years_subset_size = get_nunuques_size(df, 'fact_year_built', percent=50, start_n=10, step=5)
        years_distr_subset = get_weighted_nuniques_list(df, 'fact_year_built', years_subset_size)
        
        stored_default_values['years_distr_subset'] = years_distr_subset       
        save_default_values(default_values_file_name)

    if force_rebuild_cached_data:
        df['fact_year_built'] = df['fact_year_built'].apply(lambda x: get_weighted_random_item(years_distr_subset) if pd.isnull(x) or x == 'No Data' else x)
        freq_year = df['fact_year_built'].mode()[0]
        stored_default_values['years_mode'] = freq_year
        save_default_values(default_values_file_name)
    else:
        freq_year = stored_default_values['years_mode']
        df['fact_year_built'] = df['fact_year_built'].apply(lambda x: freq_year if pd.isnull(x) or x == 'No Data' else x)
    
    df['fact_year_built'] = df['fact_year_built'].apply(lambda x: int(x))
    
    df['object_age'] = df['fact_year_built'].apply(lambda x: 0 if (base_year-x) < 0 else (base_year-x))
    
    df = df.drop('fact_year_built', axis=1)
    
    cooling_features = get_cooling_features_list()

    mask = ~df['fact_cooling'].isna()

    for cooling_feature in cooling_features:
        df[cooling_feature] = 0
        df.loc[mask, cooling_feature] = df[mask]['fact_cooling'].apply(lambda x: get_cooling_feature(x, cooling_feature))
        
    df = df.drop('fact_cooling', axis=1)
    
    heating_features = get_heating_features_list()

    mask = ~df['fact_heating'].isna()

    for heat_feature in heating_features:
        df[heat_feature] = 0
        df.loc[mask, heat_feature] = df[mask]['fact_heating'].apply(lambda x: get_heating_feature(x, heat_feature))
        
    df = df.drop('fact_heating', axis=1)
    
    parking_features = get_parking_features_list()

    mask = ~df['fact_parking'].isna()

    for parking_feature in parking_features:
        df[parking_feature] = 0
        df.loc[mask, parking_feature] = df[mask]['fact_parking'].apply(lambda x: get_parking_feature(x, parking_feature))
        
    df = df.drop('fact_parking', axis=1)
    
    return df

def encode_state_and_city(df, default_values_file_name, can_drop_rows=False, force_rebuild_cached_data=False):
    global stored_default_values
    
    read_default_values(default_values_file_name)

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            top_cities = stored_default_values['top_cities_list']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
    
    if need_rebuil_data:
        top_cities = set((df['city'].value_counts() / df.shape[0] * 100)[:top_cities_slice_size].index)
        
        stored_default_values['top_cities_list'] = top_cities
        save_default_values(default_values_file_name)
            
    df['city'] = df['city'].apply(lambda x: x if x in top_cities else 'other_city')

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['state_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['state'])
        bin_encoder = bin_encoder.fit(df['state'])
        
        stored_default_values['state_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['state'])
    df = pd.concat([df, type_bin], axis=1).drop('state', axis=1)

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['city_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['city'])
        bin_encoder = bin_encoder.fit(df['city'])

        stored_default_values['city_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['city'])
    df = pd.concat([df, type_bin], axis=1).drop('city', axis=1)
    
    df = df.drop(['street', 'zipcode'], axis=1)
    
    return df

##################################################################################################

cities_dict = None
address_dict = None
address_by_zip_dict = None
cities_clusters_dict = None

states_replace = [
    ['Fl', 'FL'],
    ['BA', 'FL']
]

cities_replaces = [
    ['cherryhillsvillage', 'Cherry Hills Village'],
    ['commercecity', 'Commerce City'],
    ['federalheights', 'Federal Heights'],
    ['bonita spgs', 'Bonita Springs'],
    ['doctor philips', 'Orlando'],
    ['ldhl', 'Lauderhill'],
    ['p c beach', 'Panama City Beach'],
    ['un-incorporated broward county', 'Fort Lauderdale'],
    ['unincorporated broward county', 'Fort Lauderdale'],
    ['atlaanta', 'Atlanta'],
    ['saranac vlg', 'Saranac'],
    ['uninc', 'Charlotte'],
    ['west ashville', 'Ashville'],
    ['city center', 'Las Vegas'],
    ['bellerose manor', 'Queens Village'],
    ['bellerose vlg', 'Bellerose Village'],
    ['jamaica est', 'Jamaica'],
    ['old mill basin', 'Brooklyn'],
    ['downtown pgh', 'Pittsburgh'],
    ['outside area (outside ca)', 'Nashville'],
    ['unicorp/memphis', 'Memphis'],
    ['botines', 'Laredo'],
    ['brookside vl', 'Brookside Village'],
    ['bville', 'Brownsville'],
    ['clear lk shrs', 'Clear Lake Shores'],
    ['hollywood pa', 'Hollywood Park'],
    ['la moca', 'Laredo'],
    ['longvi', 'Longview'],
    ['mc allen', 'Mcallen'],
    ['mc gregor', 'Mcgregor'],
    ['mc kinney', 'Mckinney'],
    ['romayor', 'Cleveland'],
    ['s.a.', 'San Antonio'],
    ['tarkington prairie', 'Cleveland'],
    ['belllingham', 'Bellingham'],
    ['china spring', np.NaN],
    ['other city - in the state of florida', np.NaN],
    ['other city not in the state of florida', np.NaN],
    ['other city value - out of area', np.NaN],
    ['other city value out of area', np.NaN],
    ['unincorporated dade county', np.NaN],
    ['foreign country', np.NaN],
    ['other', np.NaN],
    [' ', np.NaN],
    ['--', np.NaN]
]

##################################################################################################

def get_cities_dict(file_path, force_read=False, print_error=False):
    global cities_dict
    
    if cities_dict is None or force_read:
        try:
            with open(file_path, 'rb') as f:
                cities_dict = pickle.load(f)
        except Exception as ex:
            if print_error:
                print(ex)
            cities_dict = {}
    
    return cities_dict


def get_addresses_dict(file_path, force_read=False, print_error=False):
    global address_dict
    
    if address_dict is None or force_read:
        try:
            with open(file_path, 'rb') as f:
                address_dict = pickle.load(f)
        except Exception as ex:
            if print_error:
                print(ex)
            address_dict = {}
            
    return address_dict


def get_address_by_zipcode_dict(file_path, force_read=False, print_error=False):
    global address_by_zip_dict
    
    if address_by_zip_dict is None or force_read:
        try:
            with open(file_path, 'rb') as f:
                address_by_zip_dict = pickle.load(f)
        except Exception as ex:
            if print_error:
                print(ex)
            address_by_zip_dict = {}
            
    return address_by_zip_dict

def get_citiess_clusters_dict(file_path, force_read=False, print_error=False):
    global cities_clusters_dict
    
    if cities_clusters_dict is None or force_read:
        try:
            with open(file_path, 'rb') as f:
                cities_clusters_dict = pickle.load(f)
        except Exception as ex:
            if print_error:
                print(ex)
            cities_clusters_dict = {}
            
    return cities_clusters_dict

def city_if_exists(state, city, cities_dict):
    if pd.isna(city):
        return np.NaN

    if get_city_dict_key(state, city) in cities_dict:
        return city
    else:
        return np.NaN


def fix_incorrect_states_and_cities(df, default_values_file_name, cities_dict, can_drop_rows=False):
    global stored_default_values
    
    read_default_values(default_values_file_name)

    for s_repl in states_replace:
        mask = (df['state'] == s_repl[0])
        df.loc[mask, 'state'] = s_repl[1]

    for c_repl in cities_replaces:
        mask = (df['city'] == c_repl[0])
        df.loc[mask, 'city'] = np.NaN if pd.isna(c_repl[1]) else str.lower(c_repl[1])
    
    df['city'] = df.apply(lambda x: city_if_exists(x['state'], x['city'], cities_dict), axis=1)
    
    if can_drop_rows:
        df = df.dropna(subset=['city'], axis=0)
        
        top_state = df['state'].mode()[0]
        
        pop_cities = {}
        top_state_cities = df.groupby('state')['city'].agg(pd.Series.mode).reset_index()
        for _, rec in top_state_cities.iterrows():
            pop_cities[rec['state']] = rec['city']      
        
        stored_default_values['popular_state_name'] = top_state
        stored_default_values['popular_cities'] = pop_cities
        save_default_values(default_values_file_name)
    else:
        for indx, rec in df.iterrows():
            if not pd.isna(rec['city']):
                continue
            
            if rec['state'] not in stored_default_values['popular_cities']:
                df.loc[indx, 'state'] = stored_default_values['popular_state_name']
                df.loc[indx, 'city'] = stored_default_values['popular_cities'][stored_default_values['popular_state_name']]
            else:
                if pd.isna(rec['city']):
                    df.loc[indx, 'city'] = stored_default_values['popular_cities'][rec['state']]

    return df        


def add_city_features(df, default_values_file_name, cities_dict, address_dict, address_by_zip_dict, cities_clusters_dict, force_rebuild_cached_data=False):
    global stored_default_values
    
    read_default_values(default_values_file_name)

    cities_features = [
        {'dict_field': 'type', 'df_field': 'city_type'},
        {'dict_field': 'importance', 'df_field': 'city_importance'},
        {'dict_field': 'boundingbox', 'df_field': 'city_boundingbox'},
        {'dict_field': 'lat', 'df_field': 'city_lat'},
        {'dict_field': 'lng', 'df_field': 'city_lng'},
    ]

    for city_feature in cities_features:
        df[city_feature['df_field']] = df.apply(lambda x: get_city_feature(x['state'], x['city'], city_feature['dict_field'], cities_dict), axis=1)
        
    popular_cities_types = (df['city_type'].value_counts())[:10].index
    df['city_type'] = df['city_type'].apply(lambda x: x if x in popular_cities_types else 'other_type')
    
    df['center_dist'] = df.apply(lambda  x: get_center_distance(x['state'], x['city'], x['street'], address_dict, x['city_lat'], x['city_lng'], x['city_boundingbox'], x['zipcode'], address_by_zip_dict), axis=1)
    df['hp_dist'] = df.apply(lambda  x: get_hight_price_distance(x['state'], x['city'], x['street'], address_dict, cities_clusters_dict, x['city_boundingbox'], x['zipcode'], address_by_zip_dict), axis=1)    
    df['lp_dist'] = df.apply(lambda  x: get_low_price_distance(x['state'], x['city'], x['street'], address_dict, cities_clusters_dict, x['city_boundingbox'], x['zipcode'], address_by_zip_dict), axis=1)   
    
    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['city_type_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['city_type'])
        bin_encoder = bin_encoder.fit(df['city_type'])
        
        stored_default_values['city_type_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['city_type'])
    df = pd.concat([df, type_bin], axis=1).drop('city_type', axis=1)
       
    if force_rebuild_cached_data:
        min_val = df['city_importance'].min()
        max_val = df['city_importance'].max()
        
        stored_default_values['city_importance_min'] = min_val
        stored_default_values['city_importance_max'] = max_val
        save_default_values(default_values_file_name)
    else:
        min_val = stored_default_values['city_importance_min']
        max_val = stored_default_values['city_importance_max']
        
    n_cats = 15
    df['city_importance_cat'] = 0
    df['city_importance_cat'] = df['city_importance'].apply(lambda x: convert_to_ord_cat(x, min_val, max_val, n_cats))

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['city_importance_cat_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['city_importance_cat'])
        bin_encoder = bin_encoder.fit(df['city_importance_cat'])
        
        stored_default_values['city_importance_cat_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['city_importance_cat'])
    df = pd.concat([df, type_bin], axis=1).drop(['city_importance', 'city_importance_cat'], axis=1)
   
    df['city_sqr'] = df['city_boundingbox'].apply(get_city_sqr_by_boundingbox)
    
    if force_rebuild_cached_data:
        min_val = df['city_sqr'].min()
        max_val = df['city_sqr'].mean()
        
        stored_default_values['city_sqr_min'] = min_val
        stored_default_values['city_sqr_max'] = max_val
        save_default_values(default_values_file_name)
    else:
        min_val = stored_default_values['city_sqr_min']
        max_val = stored_default_values['city_sqr_max']

    n_cats = 15
    df['city_sqr_cat'] = 0
    df['city_sqr_cat'] = df['city_sqr'].apply(lambda x: convert_to_ord_cat(x, min_val, max_val, n_cats))
    
    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['city_sqr_cat_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['city_sqr_cat'])
        bin_encoder = bin_encoder.fit(df['city_sqr_cat'])
        
        stored_default_values['city_sqr_cat_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['city_sqr_cat'])
    df = pd.concat([df, type_bin], axis=1).drop(['city_sqr', 'city_sqr_cat'], axis=1)

    df = df.drop('city_boundingbox', axis=1)
    
    if force_rebuild_cached_data:
        min_val = df['city_lat'].min()
        max_val = df['city_lat'].max()
        
        stored_default_values['city_lat_min'] = min_val
        stored_default_values['city_lat_max'] = max_val
        save_default_values(default_values_file_name)
    else:
        min_val = stored_default_values['city_lat_min']
        max_val = stored_default_values['city_lat_max']

    n_cats = 31
    df['city_lat_cat'] = 0
    df['city_lat_cat'] = df['city_lat'].apply(lambda x: convert_to_ord_cat(x, min_val, max_val, n_cats))
    
    if force_rebuild_cached_data:
        min_val = df['city_lng'].min()
        max_val = df['city_lng'].max()
        
        stored_default_values['city_lng_min'] = min_val
        stored_default_values['city_lng_max'] = max_val
        save_default_values(default_values_file_name)
    else:
        min_val = stored_default_values['city_lng_min']
        max_val = stored_default_values['city_lng_max']

    n_cats = 31
    df['city_lng_cat'] = 0
    df['city_lng_cat'] = df['city_lng'].apply(lambda x: convert_to_ord_cat(x, min_val, max_val, n_cats))
    
    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['city_lat_cat_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['city_lat_cat'])
        bin_encoder = bin_encoder.fit(df['city_lat_cat'])
        
        stored_default_values['city_lat_cat_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['city_lat_cat'])
    df = pd.concat([df, type_bin], axis=1).drop(['city_lat', 'city_lat_cat'], axis=1)

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['city_lng_cat_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['city_lng_cat'])
        bin_encoder = bin_encoder.fit(df['city_lng_cat'])
        
        stored_default_values['city_lng_cat_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['city_lng_cat'])
    df = pd.concat([df, type_bin], axis=1).drop(['city_lng', 'city_lng_cat'], axis=1)   
    
    return df


def add_population_features(df, default_values_file_name, uscities_df_path, can_drop_rows=False, force_rebuild_cached_data=False):
    global stored_default_values
    
    read_default_values(default_values_file_name)

    df_us_pop = pd.read_csv(uscities_df_path)
    
    df_us_pop['city'] = df_us_pop['city'].apply(lambda x: str.lower(x))
    df_us_pop = df_us_pop[['state_id', 'city', 'population', 'density']]
    df_us_pop.drop_duplicates(inplace=True)
    
    df_us_pop = df_us_pop.groupby(['state_id', 'city']).agg({'population': 'median', 'density': 'median'}).reset_index()
    
    df_pop = pd.merge(
        df,
        df_us_pop,
        left_on=['state', 'city'],
        right_on=['state_id', 'city'],
        how='left'
    ).drop(['state_id'], axis=1)
    
    df = df_pop.copy()
    
    if force_rebuild_cached_data:
        state_medians = df[~df['population'].isna()].groupby('state').agg({'population': 'median', 'density': 'median'})
        stored_default_values['pop_state_medians'] = state_medians
        save_default_values(default_values_file_name)
    else:
        state_medians = stored_default_values['pop_state_medians']
    
    mask = df['population'].isna()
    df.loc[mask, 'population'] = df[mask]['state'].apply(lambda x: np.NaN if x not in state_medians.index else state_medians.loc[x, 'population'])
    
    mask = df['density'].isna()
    df.loc[mask, 'density'] = df[mask]['state'].apply(lambda x: np.NaN if x not in state_medians.index else state_medians.loc[x, 'density'])
    
    if can_drop_rows:
        mask = ~df['population'].isna()
        
        p_median = df[mask]['population'].median()
        d_median = df[mask]['density'].median()
        stored_default_values['p_median'] = p_median
        stored_default_values['d_median'] = d_median
        save_default_values(default_values_file_name)
        
        df = df[mask].copy()
    else:
        p_median = stored_default_values['p_median']
        d_median = stored_default_values['d_median']

        df['population'].fillna(p_median, inplace=True)
        df['density'].fillna(p_median, inplace=True)
    
    if force_rebuild_cached_data:
        min_val = df['population'].min()
        max_val = df['population'].max()
        
        stored_default_values['population_min'] = min_val
        stored_default_values['population_max'] = max_val
        save_default_values(default_values_file_name)
    else:
        min_val = stored_default_values['population_min']
        max_val = stored_default_values['population_max']

    n_cats = 15
    df['population_cat'] = 0
    df['population_cat'] = df['population'].apply(lambda x: convert_to_ord_cat(x, min_val, max_val, n_cats))
    
    if force_rebuild_cached_data:
        min_val = df['density'].min()
        max_val = df['density'].max()
        
        stored_default_values['density_min'] = min_val
        stored_default_values['density_max'] = max_val
        save_default_values(default_values_file_name)
    else:
        min_val = stored_default_values['density_min']
        max_val = stored_default_values['density_max']

    n_cats = 15
    df['density_cat'] = 0
    df['density_cat'] = df['density'].apply(lambda x: convert_to_ord_cat(x, min_val, max_val, n_cats))
    
    df = df.drop(['population', 'density'], axis=1)
    
    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['population_cat_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['population_cat'])
        bin_encoder = bin_encoder.fit(df['population_cat'])
        
        stored_default_values['population_cat_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['population_cat'])
    df = pd.concat([df, type_bin], axis=1).drop('population_cat', axis=1)

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            bin_encoder = stored_default_values['density_cat_binenc']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        bin_encoder = ce.BinaryEncoder(cols=['density_cat'])
        bin_encoder = bin_encoder.fit(df['density_cat'])
        
        stored_default_values['density_cat_binenc'] = bin_encoder
        save_default_values(default_values_file_name)

    type_bin = bin_encoder.transform(df['density_cat'])
    df = pd.concat([df, type_bin], axis=1).drop('density_cat', axis=1)
    
    return df

##################################################################################################

city_descr_cats_features = [
    'city_type_0',
    'city_type_1',
    'city_type_2',
    'city_type_3',
    'city_importance_cat_0',
    'city_importance_cat_1',
    'city_importance_cat_2',
    'city_importance_cat_3',
    'city_importance_cat_4',
    'city_sqr_cat_0',
    'city_sqr_cat_1',
    'city_sqr_cat_2',
    'city_sqr_cat_3',
    'city_sqr_cat_4',
    'city_lat_cat_0',
    'city_lat_cat_1',
    'city_lat_cat_2',
    'city_lat_cat_3',
    'city_lat_cat_4',
    'city_lng_cat_0',
    'city_lng_cat_1',
    'city_lng_cat_2',
    'city_lng_cat_3',
    'city_0',
    'city_1',
    'city_2',
    'city_3',
    'city_4'
]

city_population_cats_features = [
    'population_cat_0',
    'population_cat_1',
    'population_cat_2',
    'population_cat_3',
    'density_cat_0',
    'density_cat_1',
    'density_cat_2',
    'density_cat_3'
]

PCA_SHRINK_RATE_1 = 0.75
PCA_SHRINK_RATE_2 = 0.90

num_cols = [
    'schools_count',
    'schools_avg_rate',
    'schools_min_distance',
    'schools_avg_distance',
    'sqft_fl',
    'object_age'
]

##################################################################################################

def final_tune_pca_and_scale(df, default_values_file_name, force_rebuild_cached_data=False):
    global stored_default_values
    
    read_default_values(default_values_file_name)

    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            pca_city_descr = stored_default_values['pca_city_descr']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        pca_city_descr = PCA(
            n_components=int(round(len(city_descr_cats_features)*PCA_SHRINK_RATE_1, 0)),
            random_state=42
        )
        pca_city_descr = pca_city_descr.fit(df[city_descr_cats_features].values)
        
        stored_default_values['pca_city_descr'] = pca_city_descr
        save_default_values(default_values_file_name)
    
    new_features = pca_city_descr.transform(df[city_descr_cats_features].values)
    new_features_names = ['cdcf_'+str(x+1) for x in range(new_features.shape[1])]
    df = df.drop(city_descr_cats_features, axis=1)
    df[new_features_names] = new_features
    
    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            pca_cpop = stored_default_values['pca_cpop']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        pca_cpop = PCA(
            n_components=int(round(len(city_population_cats_features)*PCA_SHRINK_RATE_2, 0)),
            random_state=42
        )

        pca_cpop = pca_cpop.fit(df[city_population_cats_features].values)

        stored_default_values['pca_cpop'] = pca_cpop
        save_default_values(default_values_file_name)      
    
    new_features = pca_cpop.transform(df[city_population_cats_features].values)
    new_features_names = ['cp_'+str(x+1) for x in range(new_features.shape[1])]
    df = df.drop(city_population_cats_features, axis=1)
    df[new_features_names] = new_features


    need_rebuil_data = False
    try:
        if force_rebuild_cached_data:
            need_rebuil_data = True
        else:
            scaler = stored_default_values['std_scaler']
            need_rebuil_data = False
    except Exception as ex:
        need_rebuil_data = True
        
    if need_rebuil_data:
        scaler = StandardScaler()
        scaler.fit(df[num_cols])

        stored_default_values['std_scaler'] = scaler
        save_default_values(default_values_file_name)      

    df[num_cols] = scaler.transform(df[num_cols])
    
    return df

def get_nan_replacer():
    return '***NaN***'

def prepare_for_json(df):
    return df.fillna(get_nan_replacer()).values.tolist()

def convert_to_data_frame(data):
    cols_names = [
        'status', 'private pool', 'propertyType', 'street', 'baths', 'homeFacts', 'fireplace', 'city', 'schools', 
        'sqft', 'zipcode', 'beds', 'state', 'stories', 'mls-id', 'PrivatePool', 'MlsId'
    ]
    
    result_df = pd.DataFrame(data, columns=cols_names).astype('str')
    
    replacer = get_nan_replacer()
    
    for col in cols_names:
        result_df[col] = result_df[col].apply(lambda x: np.NaN if x == replacer else x)
    
    return result_df

def get_default_model_name():
    return 'model_abr'

def get_current_prediction_model(path, model_name=None):
    if model_name is None:
        model_name = get_default_model_name()
        
    with open(path + '/' + model_name + '.pkl', 'rb') as f:
        model = pickle.load(f)
        
    return model


