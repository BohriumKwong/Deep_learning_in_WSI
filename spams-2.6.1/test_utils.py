from __future__ import absolute_import, division, print_function

import sys
import spams
import time
import numpy as np
import scipy.sparse as ssp

myfloat = np.float64

def set_float32():
    global myfloat
    myfloat = np.float32

def test1(txt,func,*args):
    tic = time.time()
    res  = func(*args)
    tac = time.time()
    print("  Time (%s) : %.3fs" %(txt,(tac - tic)))
    return res

def Xtest1(txt,expr,locs):
    tic = time.time()
    res  = eval(expr,globals(),locs)
    tac = time.time()
    print("  Time (%s) : %.3fs" %(txt,(tac - tic)))
    return res

def Xtest(s1,s2,locs):
    y1 = Xtest1('numpy',s1,locs)
    y2 = Xtest1('spams',s2,locs)
    return  abs(y2 - y1).max()
