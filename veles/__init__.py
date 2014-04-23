# -*- coding: utf-8 -*-
'''
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
'''


from email.utils import parsedate_tz, mktime_tz
from warnings import warn


__project__ = "Veles Machine Learning Platform"
__version__ = "0.3.0"
__license__ = "Samsung Proprietary License"
__copyright__ = "© 2013 Samsung Electronics Co., Ltd."
__authors__ = ["Gennady Kuznetsov", "Vadim Markovtsev", "Alexey Kazantsev",
               "Lyubov Podoynitsina", "Denis Seresov", "Dmitry Senin",
               "Alexey Golovizin", "Egor Bulychev", "Ernesto Sanches"]

try:
    __git__ = "$Commit$"
    __date__ = mktime_tz(parsedate_tz("$Date$"))
except Exception as ex:
    warn("Cannot expand variables generated by Git, setting them to None")
    __git__ = None
    __date__ = None

from veles.logger import Logger
from veles.units import Unit, OpenCLUnit
from veles.workflows import Workflow, OpenCLWorkflow
