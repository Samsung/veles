"""
type_data: данные х и у в одном файле. 0 в разных
type_input: (1 данные х формат 1-таблица 2-звук файлы
             3-спарс матрица 4-изображения)
file_input: адрес файла с данными
size: размер выборки (необязательный)
count_parametrs: количество параметров
type_output: (данные y формат 0-вектор(label) 1-таблица
              2-директория 3-спарс матрица)
file_output: адрес файла c y
count_output: количество классов
use_clone: клонировать будем?
"""

config_data2 = {'name_data': 'wine',
                'comment': '',
                'count_data': 1,
                'use_all_in_one_struct': 1,
                'data_sets': {'data0': {'number': 1,
                                        'type_data': 0,
                                        'type_input': 1,
                                        'file_input': 'wine/wine.csv',
                                        'size': 178,
                                        'count_parametrs': 13,
                                        'type_output': 1,
                                        'file_output':
                                        'wine/wine_y_labels.csv',
                                        'count_output': 3,
                                        'const_class_y ': 0,
                                        'use_clone': 0,
                                        'clone': {'use clone_const': 1,
                                                  'clone_const': 1,
                                                  'clones_proc:': [0.1, 0.2]},
                                        'use_weight': 0,
                                        'weight': {'weight_const': 1,
                                                   'use_file_weight': 1,
                                                   'type_model_weight': 1,
                                                   'weight_name_file':
                                                   'xor/xor_weight_sample.csv',
                                                   'use_weight_koef_const': 1,
                                                   'weight_koef_const': 1,
                                                   'weight_koef:': [1.5, 0.5]}}
                              }}
