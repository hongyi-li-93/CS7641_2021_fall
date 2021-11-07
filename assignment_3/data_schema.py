#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 19:28:38 2021

@author: lihongyi
"""
import os
from typing import List
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class DataType:
    @property
    def is_categorical(self):
        return False
    
    @property
    def support(self):
        return []
    
    @property
    def is_description(self):
        return False


class Numerical(DataType):
    pass


class Text(DataType):
    @property
    def is_description(self):
        return True


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
    independent_columns: List[str]
    dependent_column: str
    train_slice: slice
    test_slice: slice
    suffix: str
    header: int
    sep: str
    rand_order: bool


class MLDateSet:
    def __init__(self, df_data, schema, independent_columns, dependent_column):
        self._schema = schema
        self._independent_columns = independent_columns
        self._dependent_column = dependent_column
        self._df_data = df_data.reset_index().sort_index()
        self.decompose_categorical_columns()
    
    def decompose_categorical_columns(self):
        for col in self._independent_columns:
            if not self.schema.column_type(col).is_categorical:
                continue
            for c in self.schema.column_type(col).support[1:]:
                new_col_name = self.expanded_categorical_column_name(col, c)
                assert new_col_name not in self._df_data.columns
                self._df_data[new_col_name] = self._df_data[col].map(lambda x: 1 if x == c else 0)
    
    @property
    def expanded_independent_columns(self):
        cols = []
        for col in self._independent_columns:
            if not self.schema.column_type(col).is_categorical:
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
        elif schema.column_type(c).is_description:
            continue
        else:
            data_raw[c].astype(float)


Data_Sets_Names = ['abalone', 'google_review_ratings', 'iris']


Data_Set_Schemas = {
    'abalone': DataSetSchema(
        'abalone', 
        ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings', ], 
        {'Sex': Categorical(['M', 'F', 'I']), 
         'Rings': Categorical([i+1 for i in range(29)]), 
         'Rings_cat': Categorical([1, 2, 3]), }, 
        {'Rings_cat': lambda df: df['Rings'].map(lambda r: 1 if r <= 8 else 2 if r <= 10 else 3)}, ),
    'google_review_ratings': DataSetSchema(
        'google_review_ratings',
        ['user_id',  'churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 
         'zoo', 'restaurats', 'pubs/bars', 'local services', 'burger/pizza shops', 'hotels/other lodgigs', 
         'juice bars', 'art galleries', 'dace clubs', 'swimmig pools', 'gyms', 'bakeries', 
         'beauty & spas', 'cafes', 'view poits', 'moumets', 'gardes', ],
        {'user_id': Text(), },
        ),
    'iris': DataSetSchema(
        'iris', 
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class', ], 
        {'class': Categorical(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']), }, 
        {'class_int': lambda df: df['class'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})},
        ),
    }


Data_Set_Specs = {
    'abalone': MLDataSpec(
        ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight'], 
        'Rings_cat', slice(0, 3133), slice(3133, 4177), '.data', None, ',', False),
    'google_review_ratings': MLDataSpec(
        ['churches', 'resorts', 'beaches', 'parks', 'theatres', 'museums', 'malls', 
         'zoo', 'restaurats', 'pubs/bars', 'local services', 'burger/pizza shops', 'hotels/other lodgigs', 
         'juice bars', 'art galleries', 'dace clubs', 'swimmig pools', 'gyms', 'bakeries', 
         'beauty & spas', 'cafes', 'view poits', 'moumets', 'gardes', ], 
        'user_id', slice(0, 5456), slice(0, 0), '.csv', 0, ',', True),
    'iris': MLDataSpec(
        ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], 
        'class_int', slice(0, 3133), slice(3133, 4177), '.data', None, ',', False),
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
    
    train_set = MLDateSet(data_raw[specs.train_slice], schema, specs.independent_columns, specs.dependent_column)
    test_set = MLDateSet(data_raw[specs.test_slice], schema, specs.independent_columns, specs.dependent_column)
    return train_set, test_set    


def get_scaler(independent_matrix):
    scaler = StandardScaler()
    scaler.fit(independent_matrix) 
    return scaler    


def plot_cluster(ml_set, lables, x, y, clusters=None, name=None):
    all_clusters = np.unique(lables).tolist()
    if clusters is None:
        clusters = all_clusters
    
    where = np.where(np.isin(lables, clusters))
    data_2d = ml_set.independent_matrix[where][:, [x, y]]
    new_map = {c: i+1 for i, c in enumerate(clusters)}
    new_map_v = np.vectorize(lambda c: new_map[c])
    new_lab = new_map_v(lables[where])
    
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=new_lab, s=new_lab, alpha = 0.6)
    plt.xlabel(ml_set.expanded_independent_columns[x])
    plt.ylabel(ml_set.expanded_independent_columns[y])
    plt.title(f'showing {len(clusters)} out of {len(all_clusters)} clusters')
    if name is None:
        plt.show()
    else:
        plt.savefig(f'{name}_2d_clusters.png')
    plt.close()
    