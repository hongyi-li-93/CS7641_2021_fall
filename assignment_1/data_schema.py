#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:28:38 2021

@author: lihongyi
"""
import os


Data_Sets_Names = ['abalone']


class DataType:
    @property
    def is_categorical(self):
        return False
    
    @property
    def support(self):
        return []


class Numerical(DataType):
    pass


class Categorical(DataType):
    def __init__(self, support):
        self._support = [c for c in support]
        
    @property
    def is_categorical(self):
        return True
    
    @property
    def support(self):
        return [c for c in self._support]


class DataSetSchema:
    def __init__(self, data_set_name, column_names, column_types=None):
        self._data_set_name = data_set_name
        self._column_names = [c for c in column_names]
        self._column_type_map = {}
        if column_types is not None:
            self._column_type_map.update(column_types)
    
    @property
    def data_set_name(self):
        return self._data_set_name
    
    @property
    def column_names(self):
        return [c for c in self._column_names]
    
    def column_type(self, col_name):
        assert col_name in self._column_names
        if col_name in self._column_type_map:
            return self._column_type_map[col_name]
        else:
            return Numerical()
    
    def update_column_type(self, col_name, col_type):
        assert col_name in self._column_names
        self._column_type_map[col_name] = col_type
    

Data_Set_Schemas = {
    'abalone': DataSetSchema(
        'abalone', 
        ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings'], 
        {'Sex': ['M', 'F', 'I'], 'Rings': [i+1 for i in range(29)]}),
    }
    
        
def get_data_set(data_set_name):
    return os.path.join('data', data_set_name, f'{data_set_name}.data')


