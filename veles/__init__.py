'''
Created on May 21, 2013

@author: Markovtsev Vadim <v.markovtsev@samsung.com>
'''

from email.utils import parsedate_tz, mktime_tz


__git__ = "$Commit$"
__date__ = mktime_tz(parsedate_tz("$Date$"))
__version__ = "2.0.0"


from veles.units import Unit, OpenCLUnit
from veles.workflows import Workflow, OpenCLWorkflow
from veles.launcher import Launcher
