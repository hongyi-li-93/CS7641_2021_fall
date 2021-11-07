#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 11:45:35 2021

@author: lihongyi
"""

import unittest
import numpy as np
import gap_score


class TestGapScore(unittest.TestCase):
    def setUp(self):
        self.lables = np.array([1,1,1,2,2,2])
        self.samples = np.array([
            [1,3],
            [2,4],
            [5,3],
            [2,6],
            [5,2],
            [6,8],
            ])
        self.sample_l2 = np.array([s.dot(s.T) for s in self.samples])
        self.w_s = [np.exp(.1), np.exp(.2), np.exp(.3)]
    
    def test_l2(self):
        l2_t = gap_score.get_sample_l2(self.samples)
        np.testing.assert_allclose(l2_t, self.sample_l2)

    def test_d(self):
        where, _ = gap_score.get_where_claster(self.lables, 1)
        d_t = gap_score.get_d_cluster(where, self.samples, self.sample_l2)
        
        d_v = 0
        for i in range(3):
            s_i = self.samples[i]
            for j in range(3):
                s_j = self.samples[j]
                d = s_i - s_j
                d_l2 = d.dot(d.T)
                d_v += d_l2
        np.testing.assert_almost_equal(d_t, d_v)
    
    def test_w(self):
        w_t = gap_score.get_w(self.lables, self.samples)
        
        w_v = 0
        for i in range(3):
            s_i = self.samples[i]
            for j in range(3):
                s_j = self.samples[j]
                d = s_i - s_j
                d_l2 = d.dot(d.T)
                w_v += d_l2 / 6
        for i in range(3, 6):
            s_i = self.samples[i]
            for j in range(3, 6):
                s_j = self.samples[j]
                d = s_i - s_j
                d_l2 = d.dot(d.T)
                w_v += d_l2 / 6
        np.testing.assert_almost_equal(w_t, w_v)
    
    def test_reference(self):
        references = gap_score.get_reference_samples(self.samples, 3)
        
        self.assertEqual(len(references), 3)
        for ref in references:
            self.assertEqual(ref.shape, (6, 2))
            self.assertGreaterEqual(ref[:,0].min(), 1)
            self.assertLessEqual(ref[:, 0].max(), 6)
            self.assertGreaterEqual(ref[:, 1].min(), 2)
            self.assertLessEqual(ref[:, 1].max(), 8)
    
    def test_l_s(self):
        l_t, s_t = gap_score.get_l_s(self.w_s)
        
        l_v = 0.2
        np.testing.assert_almost_equal(l_t, l_v)
        
        sd_v = np.sqrt(.02/3)
        s_v = sd_v * np.sqrt(4/3)
        np.testing.assert_almost_equal(s_t, s_v)


if __name__ == '__main__':
    unittest.main()

                

    