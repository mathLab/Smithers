'''
Utilities to perform changes inside xml files.
'''

from __future__ import print_function
from os import listdir, path
import re


WIDTH_NEW = 800
HEIGHT_NEW = 600

DIMLINE_MASK = r'<(?P<type1>width|height)>(?P<size>\d+)</(?P<type2>width|height)>'
BBLINE_MASK = r'<(?P<type1>xmin|xmax|ymin|ymax)>(?P<size>\d+)</(?P<type2>xmin|xmax|ymin|ymax)>'
NAMELINE_MASK = r'<(?P<type1>filename)>(?P<size>\S+)</(?P<type2>filename)>'
PATHLINE_MASK = r'<(?P<type1>path)>(?P<size>.+)</(?P<type2>path)>'
#regular expression

def resize_file(file_lines):
    '''
    Function performing the requested changes on the xml file, like changing
    th coordinates x and y of the boxes and the height and width accordingly.

    :param list file_lines: list containing the lines of the file under
        consideration.
    '''
    new_lines = []
    for line in file_lines:
        match = re.search(DIMLINE_MASK, line) or re.search(BBLINE_MASK, line)
        print(match)
        if match is not None:
            size = match.group('size')
            type1 = match.group('type1')
            type2 = match.group('type2')
            print(size)
            print(type1)
            print(type2)
            if type1 != type2:
                raise ValueError('Malformed line: {}'.format(line))

            if type1.startswith('f'):
                print('f')
            if type1.startswith('x'):
                size = int(size)
                new_size = int(round(size * WIDTH_NEW / width_old))
                new_line = '\t\t\t<{}>{}</{}>\n'.format(type1, new_size, type1)
            elif type1.startswith('y'):
                size = int(size)
                new_size = int(round(size * HEIGHT_NEW / height_old))
                new_line = '\t\t\t<{}>{}</{}>\n'.format(type1, new_size, type1)
            elif type1.startswith('w'):
                size = int(size)
                width_old = size
                new_size = int(WIDTH_NEW)
                new_line = '\t\t<{}>{}</{}>\n'.format(type1, new_size, type1)
            elif type1.startswith('h'):
                size = int(size)
                height_old = size
                new_size = int(HEIGHT_NEW)
                new_line = '\t\t<{}>{}</{}>\n'.format(type1, new_size, type1)
            else:
                raise ValueError('Unknown type: {}'.format(type1))
            #new_line = '\t\t\t<{}>{}</{}>\n'.format(type1, new_size, type1)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    return ''.join(new_lines)


def change_xml(nome_file):
    '''
    Function that chnages an xml file.

    :param str nome_file: path where the xml files are
        contained (if it is a directory) or path of an xml file,
    '''
    if len(nome_file) < 1:
        raise ValueError('No file submitted')

    if path.isdir(nome_file):
    # the argument is a directory
        files = listdir(nome_file)
        for file in files:
            file_path = path.join(nome_file, file)
            _, file_ext = path.splitext(file)
            if file_ext.lower() == '.xml':
                with open(file_path, 'r') as f:
                    rows = f.readlines()

                new_file = resize_file(rows)
                with open(file_path, 'w') as f:
                    f.write(new_file)
    else:
        # otherwise i have a file (hopefully)
        with open(nome_file, 'r') as f:
            rows = f.readlines()

        new_file = resize_file(rows)
        with open(nome_file, 'w') as f:
            f.write(new_file)

#insert name of the xml file or directory that contains them
xml_file = 'voc_dir/VOC_cow/Annotations'
change_xml(xml_file)
