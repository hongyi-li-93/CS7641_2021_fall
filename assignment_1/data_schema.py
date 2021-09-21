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
    suffix: str
    header: int
    sep: str
    rand_order: bool


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


def validate_df(data_raw, schema):
    for c in schema.column_names:
        if schema.column_type(c).is_categorical:
            diff = set(data_raw[c].unique().tolist()).difference(schema.column_type(c).support)
            assert len(diff) == 0
        else:
            data_raw[c].astype(float)


Data_Sets_Names = ['abalone', 'bank-additional']


Data_Set_Schemas = {
    'abalone': DataSetSchema(
        'abalone', 
        ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings', ], 
        {'Sex': Categorical(['M', 'F', 'I']), 
         'Rings': Categorical([i+1 for i in range(29)]), 
         'Rings_cat': Categorical([1, 2, 3]), }, 
        {'Rings_cat': lambda df: df['Rings'].map(lambda r: 1 if r <= 8 else 2 if r <= 10 else 3)}, ),
    'bank-additional': DataSetSchema(
        'bank-additional', 
        ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 
         'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'], 
        {'job': Categorical(["admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown"]), 
         'marital': Categorical(["divorced","married","single","unknown"]), 
         'education': Categorical(["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"]), 
         'default': Categorical(["no","yes","unknown"]),
         'housing': Categorical(["no","yes","unknown"]),
         'loan': Categorical(["no","yes","unknown"]),
         'contact': Categorical(["cellular","telephone"]),
         'month': Categorical(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
         'day_of_week': Categorical(["mon","tue","wed","thu","fri"]),
         'poutcome': Categorical(["failure","nonexistent","success"]),
         'y': Categorical(["yes","no"]),
         'y_cat': Categorical([1, 0]), }, 
        {'y_cat': lambda df: df['y'].map(lambda r: 1 if r == 'yes' else 0)}, ), 
    }


Data_Set_Specs = {
    'abalone': MLDataSpec('Rings_cat', slice(0, 3133), slice(3133, 4177), '.data', None, ',', False),
    'bank-additional': MLDataSpec('y_cat', slice(0, 30891), slice(30891, 41188), '-full.csv', 0, ';', True),
    }


def get_data_set_loc(data_set_name, suffix):
    return os.path.join('data', data_set_name, f'{data_set_name}{suffix}')


def get_train_test_ml_set(data_set_name, seed=111):
    assert data_set_name in Data_Sets_Names
    specs = Data_Set_Specs[data_set_name]
    
    schema = Data_Set_Schemas[data_set_name]
    data_raw = pd.read_csv(get_data_set_loc(data_set_name, specs.suffix), header=specs.header, sep=specs.sep)
    if specs.rand_order:
        data_raw = data_raw.sample(frac=1, replace=False, random_state=seed).reset_index(drop=True)
    data_raw.columns = schema.input_column_names
    data_raw = schema.derive_columns(data_raw)
    validate_df(data_raw, schema)
    
    train_set = MLDateSet(data_raw[specs.train_slice], schema, specs.dependent_column)
    test_set = MLDateSet(data_raw[specs.test_slice], schema, specs.dependent_column)
    return train_set, test_set        
    