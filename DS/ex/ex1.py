import os

PATH = 'C:/Users/jun/Downloads/python_basic_1.5/2.QnA/source/41-1.txt'

# #방식 1
# def read_txt_file(path):
#     value_list = []
#     with open(path, 'r') as f:
#         lines = f.readlines()
#
#     for line in lines:
#         country, value = line.rstrip().split(',')
#         if country.lower().startswith('c'):
#             value_list.append(int(value))
#             print(country)
#
#     return sum(value_list)
#
# print(read_txt_file(PATH))

# 방식 2
import csv
import os

files = os.listdir('C:/Users/jun/Downloads/python_basic_1.5/2.QnA/source/42-1')

# def read_txt_file2():
#
#     png_list = []
#     py_list = []
#
#     for f in os.listdir('C:/Users/jun/Downloads/python_basic_1.5/2.QnA/source/42-1'):
#         print(type(os.path.splitext(f)))
#         ext = f.split('.')[-1]
#
#         if ext == 'png':
#             png_list.append(f)
#
#         if ext == 'py':
#             py_list.append(f)
#
#     print(png_list, py_list)
#
#     print('PNG file info: ', png_list, "count : ", len(png_list))
#     print('PNG file info: ', py_list, "count : ", len(py_list))
#
# print(read_txt_file2())
# import glob
#
# png_list2 = glob.glob1('C:/Users/jun/Downloads/python_basic_1.5/2.QnA/source/42-1', '*.png')
# py_list2 = glob.glob1('C:/Users/jun/Downloads/python_basic_1.5/2.QnA/source/42-1', '*.py')
#
# print(png_list2, ' count : ', len(png_list2))
# print(py_list2, 'count : ', len(py_list2))


import os
import glob

path = 'C:/Users/jun/Downloads/python_basic_1.5/2.QnA/source/43-1'


def ext_count(path):
    png_list = []
    py_list = []
    txt_list = []
    for path, folder, file in os.walk(path):
        for ext in file:
            f_ext = ext.split('.')[-1]

            if f_ext == 'txt':
                txt_list.append(ext)
            elif f_ext == 'py':
                py_list.append(ext)
            elif f_ext == 'png':
                png_list.append(ext)

    print(txt_list, f'count : {len(txt_list)}')
    print(py_list, f'count : {len(py_list)}')
    print(png_list, f'count : {len(png_list)}')

print(ext_count(path))


def ext_count2(path):
    png_list = glob.glob(path, recursive=True)

#     png_list.append(glob.glob1(path, '*.png'))
#     txt_list.append(glob.glob1(path, '*.txt'))
# print(png_list, len(png_list))
# print(txt_list, len(png_list))
