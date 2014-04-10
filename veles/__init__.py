# -*- coding: utf-8 -*-
'''
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
'''


from email.utils import parsedate_tz, mktime_tz


__git__ = "$Commit$"
__date__ = mktime_tz(parsedate_tz("$Date$"))
__version__ = "0.2.0"
__license__ = "Samsung Proprietary License"
__copyright__ = "© 2013 Samsung Electronics Co., Ltd."


from veles.logger import Logger
from veles.units import Unit, OpenCLUnit
from veles.workflows import Workflow, OpenCLWorkflow
