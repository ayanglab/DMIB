#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import os, glob, json
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
import random
import SimpleITK as sitk
import numpy as np
import pandas as pd

import datetime


def resample_image_to_350_350_350(itk_image, is_label=False):

    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [350, 350, 350]
    out_spacing = [original_spacing[0] / (out_size[0] / original_size[0]),
                   original_spacing[1] / (out_size[1] / original_size[1]),
                   original_spacing[2] / (out_size[2] / original_size[2])]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)
