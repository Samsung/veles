#!/usr/bin/python3.3 -O
"""
Created on Mart 21, 2014

Example of Wine config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


from veles.config import root, Config


root.decision = Config()  # not necessary for execution (it will do it in real
# time any way) but good for Eclipse editor

# optional parameters

root.common.update = {"plotters_disabled": True}

root.update = {"decision": {"fail_iterations": 250,
                            "snapshot_prefix": "wine_relu"},
               "global_alpha": 0.75,
               "global_lambda": 0,
               "layers": [10, 3],
               "loader": {"minibatch_maxsize": 1000000},
               "path_for_load_data":
               "/data/veles/Veles2/veles/samples/wine/wine.data"}
