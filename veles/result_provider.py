# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Jun 12, 2015

Interface and base class to provide flexible workflow run results output.

███████████████████████████████████████████████████████████████████████████████

Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.

███████████████████████████████████████████████████████████████████████████████
"""


from zope.interface import Interface


class IResultProvider(Interface):
    """
    Specifies the contract for supplying the measurable outcome of any model's
    execution. It must be implemented by units which measure the reportable
    metrics.
    """

    def get_metric_values():
        """
        :return The measurable results of model's execution, e.g., accuracy,
        number of errors, RMSE, etc. Technically, they are a dictionary of
        metric names (as returned by get_metric_names()) and achieved values.
        """

    def get_metric_names():
        """
        :return The names of metrics returned by get_metric_values(). A set.
        """
