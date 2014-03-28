#!/usr/bin/python3.3 -O
"""
Created on Mart 26, 2014

Example of Mnist config.

@author: Podoynitsina Lyubov <lyubov.p@samsung.com>
"""


from veles.config import root, Config

root.all2all = Config()  # not necessary for execution (it will do it in real
root.decision = Config()  # time any way) but good for Eclipse editor
root.loader = Config()

# optional parameters

root.labels = {"blues": 0,
               "country": 1,
               "jazz": 2,
               "pop": 3,
               "rock": 4,
               "classical": 5,
               "disco": 6,
               "hiphop": 7,
               "metal": 8,
               "reggae": 9}

root.norm_add = {'Rolloff': (-4194.1299697454906),
                 'Centroid': (-2029.2262731600895),
                 'ZeroCrossings': (-55.22063408843276),
                 'Flux': (-0.91969949785961735),
                 'Energy': (-10533446.715802385)}
root.norm_mul = {'Rolloff': 0.00016505214530598153,
                 'Centroid': 0.00014461928085116515,
                 'ZeroCrossings': 0.0025266602711760356,
                 'Flux': 0.066174680046850856,
                 'Energy': 3.2792848460441024e-09}

root.update = {"colors": ["blue",
                          "pink",
                          "green",
                          "brown",
                          "gold",
                          "white",
                          "red",
                          "black",
                          "gray",
                          "orange"],
               "features": ["Energy", "Centroid", "Flux", "Rolloff",
                            "ZeroCrossings"],
               "file_name": "/data/veles/music/22.flac",
               "graphics": 1,
               "limit": 2000000000,
               "path_for_features": "/data/veles/music/features.xml",
               "plotter_window_name": "",
               "shift_size": 10,
               "snapshot_forward":
               "/data/veles/music/GTZAN/gtzan_1000_500_10_28.88pt_Wb.pickle",
               "title_fontsize": 23,
               "window_size": 100
               }
