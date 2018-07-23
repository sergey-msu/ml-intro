import os
import numpy as np


class PATH:
    CURRENT   = os.path.dirname(os.path.realpath(__file__))
    MATERIALS = os.path.join(CURRENT, '..\materials')
    STORE     = os.path.join(CURRENT, '..\store')

    @staticmethod
    def MATERIALS_FILE(fn, dir=None):
        fpath = fn if dir is None else os.path.join(dir, fn)
        return os.path.join(PATH.MATERIALS, fpath)

    @staticmethod
    def STORE_FILE(fn, dir=None):
        fpath = fn if dir is None else os.path.join(dir, fn)
        return os.path.join(PATH.STORE, fpath)


class PRINT:
    @staticmethod
    def HEADER(text, len=80, begin_line=True, end_line=True):
        template = '{:*^'+str(len)+'}'
        if begin_line: template = '\n'+template
        if end_line:   template = template + '\n'
        print('\n{0}{1}{0}\n'.format('*'*len, template.format(' '+text+' ')))

