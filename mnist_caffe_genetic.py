import os
import time
import mnist_genetic as mg


class CaffeChromo(mg.MnistChromo):
    def fit(self, *args):
        folder_num = 0 if len(args) == 0 else args[0]
        self.create_config(folder_num)
        os.system("scripts/velescli.py -v error -s -d 1:0 "
                  "veles/znicz/tests/research/mnist.py "+
                  str(folder_num)+"/new_mnist_config.py "
                  "-r veles/znicz/tests/research/seed:1024:int32")
        min_error = 100.0
        for _root, _dir, files in os.walk(str(folder_num)):
            for file in files:
                if "pt.4.pickle" in file:
                    print(file)
                    error = float(file[len("mnist_caffe_"):
                                       len(file)-len("pt.4.pickle")])
                    os.system("rm "+str(folder_num)+"/"+file)
                    if error < min_error:
                        min_error = error
                elif file == "mnist_caffe_current.4.pickle":
                    os.system("rm "+str(folder_num)+"/"+file)
        if min_error < 0.1:
            l = 0
        return 100-min_error

    def create_config(self, folder_num):
        try:
            os.mkdir("%s" % str(folder_num))
        except:
            pass
        config = open("../Veles/"+str(folder_num)+"/new_mnist_config.py", "w")
        config.write("""#!/usr/bin/python3.3 -O\n
\"\"\"
Created on Mart 21, 2014

Example of Mnist config.

Copyright (c) 2013 Samsung Electronics Co., Ltd.
\"\"\"


from veles.config import root


# optional parameters
root.common.snapshot_dir = \"%s\"
root.update = {"learning_rate_adjust": {"do": True}, # True False
               "decision": {"max_epochs": 10},
               "snapshotter": {"prefix": "mnist_caffe"},
               "loader": {"minibatch_size": %d},
               "weights_plotter": {"limit": 64},
               "mnist": {#"learning_rate": 0.01, "gradient_moment": 0.9, # 0-1
                         #"weights_decay": 0.0005,
                         "layers":
                         [{"type": "conv", # conv conv_relu conv_str
                           "n_kernels": %d,
                           "kx": 5, "ky": 5,
                           "sliding": (1, 1),
                           "learning_rate": %f,
                           "learning_rate_bias": %f,
                           "gradient_moment": %f,
                           "gradient_moment_bias": %f,
                           "weights_filling": "uniform", # "gaussian"
                           "weights_stddev": %f,
                           "bias_filling": "constant", # "uniform", "gaussian"
                           "bias_stddev": %f,
                           "weights_decay": %f,
                           "weights_decay_bias": %f
                           },
                          {"type": "max_pooling",# abs_pooling
                           "kx": 2, "ky": 2,
                           "sliding": (2, 2)},


                          {"type": "conv",
                           "n_kernels": %d,
                           "kx": 5, "ky": 5,
                           "sliding": (1, 1),
                           "learning_rate": %f,
                           "learning_rate_bias": %f,
                           "gradient_moment": %f,
                           "gradient_moment_bias": %f,
                           "weights_filling": "uniform",
                           "weights_stddev": %f,
                           "bias_filling": "constant",
                           "bias_stddev": %f,
                           "weights_decay": %f,
                           "weights_decay_bias": %f
                           },
                          {"type": "max_pooling",
                           "kx": 2, "ky": 2, "sliding": (2, 2)},

                          {"type": "all2all_relu",
                           "output_shape": %d, # 10 - 1000
                           "learning_rate": %f,
                           "learning_rate_bias": %f,
                           "gradient_moment": %f,
                           "gradient_moment_bias": %f,
                           "weights_filling": "uniform",
                           "weights_stddev": %f,
                           "bias_filling": "constant",
                           "bias_stddev": %f,
                           "weights_decay": %f,
                           "weights_decay_bias": %f},
                          {"type": "softmax",
                           "output_shape": 10,
                           "learning_rate": %f,
                           "learning_rate_bias": %f,
                           "gradient_moment": %f,
                           "gradient_moment_bias": %f,
                           "weights_filling": "uniform",
                           "weights_stddev": %f,
                           "bias_filling": "constant",
                           "bias_stddev": %f,
                           "weights_decay": %f,
                           "weights_decay_bias": %f}]}}
""" % (folder_num,
       self.numeric[0], self.numeric[1], self.numeric[2], self.numeric[3],
       self.numeric[4], self.numeric[5], self.numeric[6], self.numeric[7],
       self.numeric[8], self.numeric[9], self.numeric[10], self.numeric[11],
       self.numeric[12], self.numeric[13], self.numeric[14], self.numeric[15],
       self.numeric[16], self.numeric[17], self.numeric[18], self.numeric[19],
       self.numeric[20], self.numeric[21], self.numeric[22], self.numeric[23],
       self.numeric[24], self.numeric[25], self.numeric[26], self.numeric[27],
       self.numeric[28], self.numeric[29], self.numeric[30], self.numeric[31],
       self.numeric[32], self.numeric[33], self.numeric[34], self.numeric[35]))

class CaffeGenetic(mg.MnistGenetic):
    def new_chromo(self, n, min_x, max_x, accuracy, codes,
                   binary=None, numeric=None, *args):
        chromo = CaffeChromo(n, min_x, max_x, accuracy, codes,
                             binary, numeric, *args)
        return chromo

if __name__ == "__main__":
    CaffeGenetic().evolution()
