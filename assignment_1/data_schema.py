#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:28:38 2021

@author: lihongyi
"""
import os
import pandas as pd
from dataclasses import dataclass


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
        assert len(support) > 1
        self._support = [c for c in support]
        
    @property
    def is_categorical(self):
        return True
    
    @property
    def support(self):
        return [c for c in self._support]


class DataSetSchema:
    def __init__(self, schema_name, column_names, column_types=None, derived_column_map=None):
        self._schema_name = schema_name
        self._column_names = [c for c in column_names]
        self._column_type_map = {}
        if column_types is not None:
            self._column_type_map.update(column_types)
        self._derived_column_map = {}
        if derived_column_map is not None:
            self._derived_column_map.update(derived_column_map)
    
    @property
    def schema_name(self):
        return self._schema_name
    
    @property
    def input_column_names(self):
        return [c for c in self._column_names]
    
    @property
    def column_names(self):
        return [c for c in self._column_names] + list(self._derived_column_map)
    
    def column_type(self, col_name):
        assert col_name in self.column_names
        if col_name in self._column_type_map:
            return self._column_type_map[col_name]
        else:
            return Numerical()
    
    def derive_columns(self, df):
        for col, map_c in self._derived_column_map.items():
            if col not in df.columns:
                df[col] = map_c(df)
        return df
    
    def add_derived_column(self, col_name, map_func):
        assert col_name not in self.column_names
        self._derived_column_map[col_name] = map_func
    
    def update_column_type(self, col_name, col_type):
        assert col_name in self.column_names
        self._column_type_map[col_name] = col_type


@dataclass
class MLDataSpec:
    dependent_column: str
    train_slice: slice
    test_slice: slice


class MLDateSet:
    def __init__(self, df_data, schema, dependent_column):
        self._schema = schema
        self._dependent_column = dependent_column
        self._df_data = df_data.reset_index().sort_index()
        self.decompose_categorical_columns()
    
    def decompose_categorical_columns(self):
        for col in self.schema.column_names:
            if col == self._dependent_column:
                continue    
            if not isinstance(self.schema.column_type(col), Categorical):
                continue
            for c in self.schema.column_type(col).support[1:]:
                new_col_name = self.expanded_categorical_column_name(col, c)
                assert new_col_name not in self._df_data.columns
                self._df_data[new_col_name] = self._df_data[col].map(lambda x: 1 if x == c else 0)
    
    @property
    def expanded_independent_columns(self):
        cols = []
        for col in self.schema.column_names:
            if col == self._dependent_column:
                continue
            if not isinstance(self.schema.column_type(col), Categorical):
                cols.append(col)
            else:
                for c in self.schema.column_type(col).support[1:]: 
                    cols.append(self.expanded_categorical_column_name(col, c))
        return cols
    
    @property
    def schema(self):
        return self._schema
    
    @property
    def seperator(self):
        return ']\['
    
    def expanded_categorical_column_name(self, col, category):
        return col + self.seperator + str(category)
    
    @property
    def independent_matrix(self):
        return self._df_data[self.expanded_independent_columns].to_numpy()
    
    @property
    def dependent_vector(self):
        return self._df_data[self._dependent_column].to_numpy()


Data_Sets_Names = ['abalone']


Data_Set_Schemas = {
    'abalone': DataSetSchema(
        'abalone', 
        ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings', ], 
        {'Sex': Categorical(['M', 'F', 'I']), 
         'Rings': Categorical([i+1 for i in range(29)]), 
         'Rings_cat': Categorical([1, 2, 3]), }, 
        {'Rings_cat': lambda df: df['Rings'].map(lambda r: 1 if r <= 8 else 2 if r <= 10 else 3)}, ),
    }


Data_Set_Specs = {
    'abalone': MLDataSpec('Rings_cat', slice(0, 3133), slice(3133, 4177))
    }


def get_data_set_loc(data_set_name):
    return os.path.join('data', data_set_name, f'{data_set_name}.data')


def get_train_test_ml_set(data_set_name):
    schema = Data_Set_Schemas[data_set_name]
    data_raw = pd.read_csv(get_data_set_loc(data_set_name), header=None)
    data_raw.columns = schema.input_column_names
    data_raw = schema.derive_columns(data_raw)
    
    specs = Data_Set_Specs[data_set_name]
    train_set = MLDateSet(data_raw[specs.train_slice], schema, specs.dependent_column)
    test_set = MLDateSet(data_raw[specs.test_slice], schema, specs.dependent_column)
    return train_set, test_set        
    