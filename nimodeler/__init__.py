# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:26:42 2021

@author: Ame
"""
import inspect
from os.path import split
from pathlib import Path

import numpy as np

from .nimodeler import *


__version__ = "0.1.0"
__pkg_dir__ = Path(inspect.getfile(inspect.currentframe())).parent

if split(__pkg_dir__)[-1] == "":
    __git_dir__ = str(Path(split(__pkg_dir__)[0]).parent)
else:
    __git_dir__ = str(split(__pkg_dir__)[0])
__pkg_dir__ = str(__pkg_dir__)
