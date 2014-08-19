import os
import sys
import pickle
from veles.znicz.tests.research import genetic as gen
from veles.config import root
import math
from multiprocessing import Pool


class MnistChromo(gen.Chromosome):
    def fit(self, *args):
        folder_num = 0 if len(args) == 0 else args[0]
        self.create_config(folder_num)
        os.system("scripts/velescli.py --no-logo -v error -d 1:0 -s " +
                  "-r ../Veles/seed.rnd " +
                  "veles/znicz/samples/mnist.py " +
                  str(args[0])+"/new_mnist_config.py")
        errfile = open(str(args[0]) + "/errors.txt", "r")
        error = 0
        while True:
            line = errfile.readline()
            if line == "":
                break
            else:
                error = float(line)
#         os.system("rm errors.txt")
        return 100-error

    def create_config(self, folder_num):
        try:
            os.mkdir("%s" % str(folder_num))
        except:
            pass
        template = open("../Veles/veles/znicz/samples/config.py", "r")
        config = open("../Veles/"+str(folder_num)+"/new_mnist_config.py", "w")
        s = template.readline()
        while s != "":
            if "mnist_dir = " in s:
                s = "mnist_dir = \"../Veles/veles/znicz/samples/MNIST\""
            config.write(s)
            s = template.readline()
        config.write("root.common.snapshot_dir = \"%s\"\n\n" % str(folder_num))
        config.write("root.update = "
                     "{\"all2all\": {\"weights_stddev\": %.3f},\n"
                     % self.numeric[0])
        config.write("               \"decision\":"
                     " {\"fail_iterations\": 100,\n")
        config.write("                            \"store_samples_mse\":"
                     " True,\n")
        config.write("                            \"max_epochs\": 70},\n")
        config.write("               \"loader\": {\"minibatch_size\": %d},\n"
                     % self.numeric[1])
        config.write("               \"snapshotter\":"
                     " {\"prefix\": \"mnist\"},\n")
        config.write("               \"mnist\": {\"learning_rate\": %.3f,\n"
                     % self.numeric[2])
        config.write("                         \"weights_decay\": %.3f,\n"
                     % self.numeric[3])
        config.write("                         \"layers\": [%d, 10],\n"
                     % self.numeric[4])
        config.write("                         \"data_paths\":"
                     " {\"test_images\": "
                     "test_image_dir,\n")
        config.write("                                        \"test_label\": "
                     "test_label_dir,\n")
        config.write("                                        \"train_images\""
                     ": train_image_dir,\n")
        config.write("                                        \"train_label\":"
                     " train_label_dir}}}\n")
        config.close()


def add_mnist_chromo(i, genetic):
    if i < root.population.chromosomes:
        chromo = MnistChromo(root.optimization.dimensions,
                             root.optimization.min_x,
                             root.optimization.max_x,
                             1/root.optimization.accuracy,
                             genetic.codes, None, None, i)
        genetic.chromosomes.append(chromo)
        genetic.population_fitness += chromo.fitness
        print("chromosome #%d fitness = %.2f" % (i, chromo.fitness))


class MnistGenetic(gen.Genetic):
    def __init__(self):
        gen.set_config(sys.argv[1])
        self.outfile = open("out.txt", "w")
        max_abs_x = 0
        for i in range(root.optimization.dimensions):
            if math.fabs(root.optimization.min_x[i]) > max_abs_x:
                max_abs_x = math.fabs(root.optimization.min_x[i])
            if math.fabs(root.optimization.max_x[i]) > max_abs_x:
                max_abs_x = math.fabs(root.optimization.max_x[i])
        max_coded_int = int(max_abs_x/root.optimization.accuracy)
        # Length of code of one int number
        self.dl = int(math.ceil(math.log2(max_coded_int)))
        self.codes = []
        if root.optimization.code == "gray":
            self.codes = gen.gray(self.dl)
        # +1 symbol 1/0 for positive/negative
        self.dl += 1

        if os.path.exists("population.p"):
            self.chromosomes = pickle.load(open("population.p", "rb"))
        else:
            self.chromosomes = []

        i = len(self.chromosomes)
        p = Pool(3)
        while i < root.population.chromosomes:
            (chromo1,
             chromo2,
             chromo3) = p.starmap(MnistChromo,
                                  [(root.optimization.dimensions,
                                    root.optimization.min_x,
                                    root.optimization.max_x,
                                    1/root.optimization.accuracy,
                                    self.codes, None, None, 10),
                                   (root.optimization.dimensions,
                                    root.optimization.min_x,
                                    root.optimization.max_x,
                                    1/root.optimization.accuracy,
                                    self.codes, None, None, 11),
                                   (root.optimization.dimensions,
                                    root.optimization.min_x,
                                    root.optimization.max_x,
                                    1/root.optimization.accuracy,
                                    self.codes, None, None, 12)])
            self.add(chromo1)
            self.add(chromo2)
            self.add(chromo3)
            print("chromosomes #%d and #%d and #%d fitness = "
                  "%.2f and %.2f and %.2f" %
                  (i, i+1, i+2,
                   chromo1.fitness, chromo2.fitness, chromo3.fitness))
            i += 3
        pickle.dump(self.chromosomes, open("population.p", "wb"))

        self.population_fitness = 0
        for chromo in self.chromosomes:
                    self.population_fitness += chromo.fitness

    def new_chromo(self, n, min_x, max_x, accuracy, codes,
                   binary=None, numeric=None, *args):
        chromo = MnistChromo(n, min_x, max_x, accuracy,
                             codes, binary, numeric, *args)
        return chromo


def run():
    pop = MnistGenetic()
    pop.evolution()
    print(pop.chromosomes[0].numeric)
    print(pop.chromosomes[0].fitness)


if __name__ == "__main__":
    sys.exit(run())
