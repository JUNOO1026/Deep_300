import os
import glob


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


class JunFile:
    def __init__(self, path):
        self.path = path

    # 디렉토리 내에 파일 갯수 셀 때 사용
    def ext_count(self):
        png_list = []
        py_list = []
        txt_list = []
        for path, folder, file in os.walk(self.path):
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

1