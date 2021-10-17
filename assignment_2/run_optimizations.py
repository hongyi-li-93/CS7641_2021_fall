#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 20:00:32 2021

@author: lihongyi
"""

import rhc
import genetic
import sim_anneal
import mimic


if __name__ == '__main__':
    rhc.run()
    genetic.run()
    sim_anneal.run()
    mimic.run()
    
    rhc.nn()
    sim_anneal.nn()
    genetic.nn()
