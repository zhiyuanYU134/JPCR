import torch
import math
import numpy as np
import jittor as jt
from jittor import nn
from jittor.contrib import concat 

def index_select(data, index, dim):
    output=jt.index_select(data,dim,index.view(-1))

    if index.ndim > 1:
        output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
        output = output.view(*output_shape)
    return output
