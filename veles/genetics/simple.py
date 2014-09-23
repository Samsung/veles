"""
Created on July 17, 2014

Classes for genetic optimizations.

In general:
Gene - contains parameter to optimize.
Chromosome - contains set of genes.
Individual - contains set of chromosomes.
Population - contains set of individuals.

For now:
Chromosome - parameters to optimize.
Population - list of chromosomes.

Copyright (c) 2014 Samsung Electronics Co., Ltd.
"""


import numpy
import pickle

from veles.config import root
from veles.distributable import Pickleable
from veles.external.progressbar import ProgressBar


def schwefel(values):
    """Schwefel's (Sine Root) Function.

    May be used for visualization of the population fitness.
    """
    if not len(values):
        return 0
    return 1.0 / (
        418.9829 * len(values) - numpy.sum(numpy.multiply(
            values, numpy.sin(numpy.sqrt(numpy.fabs(values))))))


def gray(code_length):
    """Recursive constructing of Gray codes list.

    TODO(a.kazantsev): examine for correctness and possible optimizations.
    """
    if code_length == 2:
        return ["00", "01", "11", "10"]
    else:
        codes = gray(code_length - 1)
        codes_size = len(codes)
        for i in range(codes_size):
            codes.append("1" + codes[codes_size - i - 1])
            codes[codes_size - i - 1] = "0" + codes[codes_size - i - 1]
        return codes


def bin_to_num(binaries, delimeter, accuracy, codes):
    """Convert gray codes of chromosomes to arrays of floats.

    TODO(a.kazantsev): examine for correctness and possible optimizations.
    """
    num = ([], [])
    delimiter1 = 0
    delimiter2 = delimeter
    chromo_length = len(binaries[0])
    binaries_num = len(binaries)
    while delimiter1 < chromo_length:
        for i in range(binaries_num):
            cut = binaries[i][delimiter1:delimiter2]
            # Gray codes to dec numbers
            num[i].append(codes.index(cut[1:]) * accuracy
                          * (-1 if cut[0] == '0' else 1))
        delimiter1 = delimiter2
        delimiter2 += delimeter
    return num


def num_to_bin(numbers, accuracy, codes):
    """Convert float numbers to gray codes.

    TODO(a.kazantsev): examine for correctness and possible optimizations.
    """
    binary = ""
    for i in range(len(numbers)):
        if numbers[i] > 0:
            binary += "1"
        else:
            binary += "0"
        binary += codes[int(numpy.fabs(numbers[i] / accuracy))]
    return binary


class Chromosome(Pickleable):
    """Chromosome (for now it is the same as individual).

    Abstract methods:
        evaluate

    Attributes:
        size: current number of genes.
        binary: binary representation of genes as string with "0"s and "1"s.
        numeric: list of numeric genes.
    """
    def __init__(self, size, minvles, maxvles, accuracy, codes,
                 binary=None, numeric=None):
        """Constructs the chromosome and computes it's fitness.

        Parameters:
            size: number of genes or 0.
            minvles: list of minimum values for genes.
            maxvles: list of maximum values for genes.
            accuracy: floating point approximation accuracy.
            codes: gray codes if any.
            binary: binary representation of genes.
            numeric: list of numeric genes.
        """
        super(Chromosome, self).__init__()

        self.optimization_choice = "betw"
        self.optimization_code = "float"

        self.minvles = minvles
        self.maxvles = maxvles
        if size:
            self.size = size
            self.binary = ""
            self.numeric = []
            for j in range(size):
                if self.optimization_choice == "or":
                    rand = numpy.random.choice([minvles[j], maxvles[j]])
                    self.numeric.append(rand)
                elif type(minvles[j]) == float or type(maxvles[j]) == float:
                    rand = numpy.random.randint(int(minvles[j] * accuracy),
                                                int(maxvles[j] * accuracy) + 1)
                    self.numeric.append(rand / accuracy)
                else:
                    rand = numpy.random.randint(minvles[j], maxvles[j] + 1)
                    self.numeric.append(rand)
                    rand = int(rand * accuracy)
                if self.optimization_code == "gray":
                    if rand > 0:
                        self.binary += "1" + codes[rand]
                    else:
                        self.binary += "0" + codes[rand]
        else:
            self.numeric = numeric
            self.numeric_correct()
            self.binary = binary
            self.size = len(numeric)

        self.fitness = None

    def init_unpickled(self):
        super(Chromosome, self).init_unpickled()
        self.mut_map_ = {
            "binary_point": self.mutation_binary_point,
            "gaussian": self.mutation_gaussian,
            "uniform": self.mutation_uniform,
            "altering": self.mutation_altering}

    def copy(self):
        return pickle.loads(pickle.dumps(self))

    def evaluate(self):
        """Computes fitness

        Should assign self.fitness.
        """
        raise NotImplementedError()

    def numeric_correct(self):
        for pos in range(len(self.numeric)):
            maxvle = self.maxvles[pos]
            minvle = self.minvles[pos]
            diff = maxvle - minvle
            while self.numeric[pos] < minvle or self.numeric[pos] > maxvle:
                self.fitness = None
                if self.numeric[pos] < minvle:
                    self.numeric[pos] += diff
                else:
                    self.numeric[pos] -= diff

    def mutate(self, mutnme, n_points, probability):
        return self.mut_map_[mutnme](n_points, probability)

    def mutation_binary_point(self, n_points, probability):
        """Changes 0 => 1 and 1 => 0.
        """
        mutant = ""
        for _ in range(n_points):
            self.fitness = None
            pos = numpy.random.randint(1, len(self.binary) - 1)
            p_m = numpy.random.rand()
            if p_m < probability:
                if self.binary[pos] == "0":
                    mutant = self.binary[:pos] + "1" + self.binary[pos + 1:]
                else:
                    mutant = self.binary[:pos] + "0" + self.binary[pos + 1:]
            else:
                mutant = self.binary
        self.binary = mutant

    def mutation_altering(self, n_points, probability):
        """Changes positions of two floats.
        """
        if self.optimization_code == "gray":
            mutant = ""
            for _ in range(n_points):
                self.fitness = None
                pos1 = numpy.random.randint(len(self.binary))
                pos2 = numpy.random.randint(len(self.binary))
                p_m = numpy.random.rand()
                if p_m < probability:
                    if pos1 < pos2:
                        mutant = (self.binary[:pos1] + self.binary[pos1] +
                                  self.binary[pos1 + 1:pos2] +
                                  self.binary[pos2] + self.binary[pos2 + 1:])
                    else:
                        mutant = (self.binary[:pos2] + self.binary[pos2] +
                                  self.binary[pos2 + 1:pos1] +
                                  self.binary[pos1] + self.binary[pos1 + 1:])
                else:
                    mutant = self.binary
            self.binary = mutant
        else:
            for _ in range(n_points):
                self.fitness = None
                pos1 = numpy.random.randint(len(self.numeric))
                pos2 = numpy.random.randint(len(self.numeric))
                p_m = numpy.random.rand()
                if p_m < probability:
                    temp = self.numeric[pos1]
                    self.numeric[pos1] = self.numeric[pos2]
                    self.numeric[pos2] = temp

    def mutation_gaussian(self, n_points, probability):
        """Adds random gaussian number.
        """
        minvles = self.minvles
        maxvles = self.maxvles
        mut_pool = [i for i in range(len(self.numeric))]
        for _ in range(n_points):
            self.fitness = None
            pos = numpy.random.choice(mut_pool)
            if self.optimization_choice == "or":
                self.numeric[pos] = numpy.random.choice(
                    [minvles[pos], maxvles[pos]])
            else:
                isint = (type(self.numeric[pos]) == int)
                diff = maxvles[pos] - minvles[pos]
                max_prob = minvles[pos] + diff / 2
                gauss = numpy.random.normal(max_prob, numpy.sqrt(diff / 6))
                p_m = numpy.random.rand()
                if p_m < probability:
                    if numpy.random.random() < 0.5:
                        self.numeric[pos] -= gauss
                    else:
                        self.numeric[pos] += gauss
                    # Bringing numeric[pos] to its limits
                    while (self.numeric[pos] < minvles[pos] or
                           self.numeric[pos] > maxvles[pos]):
                        if self.numeric[pos] < minvles[pos]:
                            self.numeric[pos] += diff
                        else:
                            self.numeric[pos] -= diff
                    if isint:
                        self.numeric[pos] = int(self.numeric[pos])
                mut_pool.remove(pos)
                if not len(mut_pool):
                    break

    def mutation_uniform(self, n_points, probability):
        """Replaces float number with another random number.
        """
        minvles = self.minvles
        maxvles = self.maxvles
        mut_pool = list(range(len(self.numeric)))
        for _ in range(n_points):
            self.fitness = None
            pos = numpy.random.choice(mut_pool)
            if self.optimization_choice == "or":
                self.numeric[pos] = numpy.random.choice(
                    [minvles[pos], maxvles[pos]])
            else:
                isint = (type(self.numeric[pos]) == int)
                p_m = numpy.random.rand()
                if p_m < probability:
                    rand = numpy.random.uniform(minvles[pos], maxvles[pos])
                    if isint:
                        rand = int(rand)
                    self.numeric[pos] = rand
                mut_pool.remove(pos)
                if len(mut_pool) == 0:
                    break


class Population(Pickleable):
    """Base class for population.

    Abstract methods:
        new_chromo
    """
    def __init__(self, optimization_size,
                 optimization_minvles, optimization_maxvles,
                 optimization_accuracy=0.00001):
        super(Population, self).__init__()

        self.optimization_choice = "betw"
        self.optimization_code = "float"

        self.optimization_size = optimization_size
        self.optimization_minvles = optimization_minvles
        self.optimization_maxvles = optimization_maxvles
        self.optimization_accuracy = optimization_accuracy

        self.chromosomes = []

        self.population_size = 50

        self.fitness = None
        self.average_fit = None
        self.best_fit = None
        self.worst_fit = None
        self.median_fit = None

        self.roulette_select_size = 0.75
        self.random_select_size = 0.5
        self.tournament_size = 0.5
        self.tournament_select_size = 0.1

        self.crossing_pointed_crossings = 0.2
        self.crossing_pointed_points = 0.08
        self.crossing_pointed_probability = 1.0

        self.crossing_uniform_crossings = 0.15
        self.crossing_uniform_probability = 0.9

        self.crossing_arithmetic_crossings = 0.15
        self.crossing_arithmetic_probability = 0.9

        self.crossing_geometric_crossings = 0.2
        self.crossing_geometric_probability = 0.9

        self.delimeter = None
        self.codes = None

        self.crossings = [self.crossing_uniform,
                          self.crossing_arithmetic,
                          self.crossing_geometric]

        self.mutations = {
            "binary_point": {"use": False,
                             "chromosomes": 0.2,
                             "points": 0.06,
                             "probability": 0.35},
            "gaussian": {"use": True,
                         "chromosomes": 0.35,
                         "points": 0.05,
                         "probability": 0.7},
            "uniform": {"use": True,
                        "chromosomes": 0.35,
                        "points": 0.05,
                        "probability": 0.7},
            "altering": {"use": False,
                         "chromosomes": 0.1,
                         "points": None,
                         "probability": 0.35}}

        self.generation = 0

        self.prev_state_fnme = None

    def new_chromo(self, size, minvles, maxvles, accuracy, codes,
                   binary=None, numeric=None):
        raise NotImplementedError()

    def __repr__(self):
        return "%s with %d chromosomes" % (
            super(Population, self).__repr__(), len(self))

    def __getitem__(self, key):
        """Returns the Chromosome by its index.
        """
        return self.chromosomes[key]

    def __iter__(self):
        """Returns the iterator for chromosomes.
        """
        return iter(self.chromosomes)

    def __len__(self):
        """Returns the number of chromosomes.
        """
        return len(self.chromosomes)

    def add(self, chromo):
        assert isinstance(chromo, Chromosome)
        self.chromosomes.append(chromo)

    def sort(self):
        """Sorts the population be fitness and
        truncates up to maximum population size.
        """
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        del self.chromosomes[self.population_size:]

    def evaluate(self):
        """Sequential evaluation.
        """
        for i, u in enumerate(self):
            if u.fitness is None:
                self.info("Will evaluate chromosome number %d (%.2f%%)",
                          i, 100.0 * i / len(self))
                u.evaluate()

    def _evaluate(self):
        self.evaluate()  # evaluate population
        self.sort()  # kill excessive worst ones
        self.fitness = sum(u.fitness for u in self)

    def selection(self):
        """Current selection procedure.
        """
        return self.selection_roulette()

    def selection_roulette(self):
        """Selection for crossing with roulette.
        """
        bound = []
        sum_v = 0.0
        for u in self:
            sum_v += u.fitness / self.fitness
            bound.append(100.0 * sum_v)
        bound[-1] = 100.0
        parents = []
        for _ in range(int(len(self) * self.roulette_select_size)):
            rand = 100.0 * numpy.random.rand()
            j = 0
            while rand > bound[j]:
                j += 1
            parents.append(self[j])
        return parents

    def selection_random(self):
        """Random selection for crossing.
        """
        parents = []
        for _ in range(int(len(self) * self.random_select_size)):
            rand = numpy.random.randint(len(self))
            parents.append(self[rand])
        return parents

    def selection_tournament(self):
        """Tournament selection for crossing.
        """
        tournament_pool = []
        for _ in range(int(len(self) * self.tournament_size)):
            rand = numpy.random.randint(len(self))
            j = 0
            while (j < len(tournament_pool) and
                    tournament_pool[j].fitness < self[rand].fitness):
                j += 1
            tournament_pool.insert(j, self[rand])
        return tournament_pool[:int(len(self) * self.tournament_select_size)]

    def crossing_pointed(self, parents):
        """Genetic operator.
        """
        for _cross_num in range(int(len(self) *
                                    self.crossing_pointed_crossings)):
            rand1 = numpy.random.randint(len(parents))
            parent1 = parents[rand1].binary
            rand2 = numpy.random.randint(len(parents))
            parent2 = parents[rand2].binary
            cross_points = [0, ]
            l = 0
            for _cross_point in range(int(len(self) *
                                          self.crossing_pointed_points)):
                while l in cross_points:
                    l = numpy.random.randint(1, len(parent1) - 1)
                j = 0
                while j < len(cross_points) and cross_points[j] < l:
                    j += 1
                cross_points.insert(j, l)
            cross_points.append(len(parent1))
            cross1 = ""
            cross2 = ""
            i = 1
            while i <= self.crossing_pointed_points + 1:
                if i % 2 == 0:
                    cross1 += parent1[cross_points[i - 1]:cross_points[i]]
                    cross2 += parent2[cross_points[i - 1]:cross_points[i]]
                else:
                    cross1 += parent2[cross_points[i - 1]:cross_points[i]]
                    cross2 += parent1[cross_points[i - 1]:cross_points[i]]
                i += 1
            (num1, num2) = bin_to_num([cross1, cross2], self.dl,
                                      self.optimization_accuracy, self.codes)
            chromo_son1 = self.new_chromo(0, self.minvles, self.maxvles,
                                          1.0 / self.optimization_accuracy,
                                          self.codes, cross1, num1)
            chromo_son2 = self.new_chromo(0, self.minvles, self.maxvles,
                                          1.0 / self.optimization_accuracy,
                                          self.codes, cross2, num2)
            chromo_son1.size = len(chromo_son1.numeric)
            chromo_son2.size = len(chromo_son2.numeric)
            self.add(chromo_son1)
            self.add(chromo_son2)

    def crossing_uniform(self, parents):
        for _cross_num in range(int(len(self) *
                                    self.crossing_uniform_crossings)):
            if self.optimization_code == "gray":
                rand1 = numpy.random.randint(len(parents))
                parent1 = parents[rand1].binary
                rand2 = numpy.random.randint(len(parents))
                parent2 = parents[rand2].binary
                cross = ""
                for i in range(len(parent1)):
                    rand = numpy.random.uniform(0, 2)
                    if rand < 1:
                        cross += parent1[i]
                    else:
                        cross += parent2[i]
                numeric = bin_to_num([cross], self.dl,
                                     self.optimization_accuracy, self.codes)[0]
                chromo_son = self.new_chromo(0, self.minvles, self.maxvles,
                                             1.0 / self.optimization_accuracy,
                                             self.codes, cross, numeric)
            else:
                rand1 = numpy.random.randint(len(parents))
                parent1 = parents[rand1].numeric
                rand2 = numpy.random.randint(len(parents))
                parent2 = parents[rand2].numeric
                cross = []
                for i in range(len(parent1)):
                    rand = numpy.random.uniform(0, 2)
                    if rand < 1:
                        cross.append(parent1[i])
                    else:
                        cross.append(parent2[i])
                chromo_son = self.new_chromo(0, self.optimization_minvles,
                                             self.optimization_maxvles,
                                             1.0 / self.optimization_accuracy,
                                             self.codes, None, cross)
            self.add(chromo_son)

    def crossing_arithmetic(self, parents):
        """Arithmetical crossingover.
        """
        for _cross_num in range(int(len(self) *
                                    self.crossing_arithmetic_crossings)):
            rand1 = numpy.random.randint(0, len(parents))
            parent1 = parents[rand1].numeric
            rand2 = numpy.random.randint(0, len(parents))
            parent2 = parents[rand2].numeric
            cross1 = []
            cross2 = []
            for i in range(len(parent1)):
                a = numpy.random.random()
                if self.optimization_choice == "or":
                    if a > 0.5:
                        cross1.append(parent1[i])
                        cross2.append(parent2[i])
                    else:
                        cross1.append(parent2[i])
                        cross2.append(parent1[i])
                elif type(parent1[i]) == int:
                    k = int(a * parent1[i] + (1 - a) * parent2[i])
                    cross1.append(k)
                    cross2.append(parent1[i] + parent2[i] - k)
                else:
                    cross1.append(a * parent1[i] + (1 - a) * parent2[i])
                    cross2.append((1 - a) * parent1[i] + a * parent2[i])
            if self.optimization_code == "gray":
                (bin1, bin2) = (num_to_bin(cross1, self.optimization_accuracy,
                                           self.codes),
                                num_to_bin(cross2, self.optimization_accuracy,
                                           self.codes))
            else:
                (bin1, bin2) = ("", "")
            chromo1 = self.new_chromo(0, self.optimization_minvles,
                                      self.optimization_maxvles,
                                      1.0 / self.optimization_accuracy,
                                      self.codes, bin1, cross1)
            chromo2 = self.new_chromo(0, self.optimization_minvles,
                                      self.optimization_maxvles,
                                      1.0 / self.optimization_accuracy,
                                      self.codes, bin2, cross2)
            self.add(chromo1)
            self.add(chromo2)

    def crossing_geometric(self, parents):
        """Geometrical crossingover.
        """
        for _cross_num in range(int(len(self) *
                                    self.crossing_geometric_crossings)):
            cross = []
            rand1 = numpy.random.randint(len(parents))
            parent1 = parents[rand1].numeric
            rand2 = numpy.random.randint(len(parents))
            parent2 = parents[rand2].numeric
            for i in range(len(parent1)):
                if self.optimization_choice == "or":
                    if numpy.random.random() > 0.5:
                        cross.append(parent1[i])
                    else:
                        cross.append(parent2[i])
                else:
                    # correct1 is used to invert [-x1; -x2] to [x2; x1]
                    correct1 = -1 if self.optimization_maxvles[i] < 0 else 1
                    # correct2 is used to alter [-x1; x2] to [0; x2+x1]
                    if self.optimization_minvles[i] > 0 or correct1 == -1:
                        correct2 = 0
                    else:
                        correct2 = -self.optimization_minvles[i]
                    a = numpy.random.rand()
                    gene = (correct1 * (numpy.power(
                        correct1 * parent1[i] + correct2, a) * numpy.power(
                        correct1 * parent2[i] + correct2, (1 - a)) - correct2))
                    if type(parent1[i]) == int:
                        gene = int(gene)
                    cross.append(gene)
            binary = ""
            if self.optimization_code == "gray":
                binary = num_to_bin(cross, self.optimization_accuracy,
                                    self.codes)
            chromo_son = self.new_chromo(0, self.optimization_minvles,
                                         self.optimization_maxvles,
                                         1.0 / self.optimization_accuracy,
                                         self.codes, binary, cross)
            self.add(chromo_son)

    def compute_gray_codes(self):
        max_abs_x = 0
        for i in range(self.optimization_size):
            if numpy.fabs(self.optimization_minvles[i]) > max_abs_x:
                max_abs_x = numpy.fabs(self.optimization_minvles[i])
            if numpy.fabs(self.optimization_maxvles[i]) > max_abs_x:
                max_abs_x = numpy.fabs(self.optimization_maxvles[i])
        max_coded_int = int(max_abs_x / self.optimization_accuracy)
        # Length of code of one int number
        self.delimeter = int(numpy.log2(max_coded_int))
        self.codes = gray(self.delimeter)
        # +1 symbol 1/0 for positive/negative
        self.delimeter += 1

    def do_evolution_step(self):
        """Evolves the population (one step).
        """
        fin = self.get_pickle_fin()
        if fin is not None:
            self.chromosomes = pickle.load(fin)
            fin.close()

        self.info("Creating chromosomes...")
        for _ in ProgressBar(term_width=20)(range(self.population_size -
                                                  len(self.chromosomes))):
            chromo = self.new_chromo(self.optimization_size,
                                     self.optimization_minvles,
                                     self.optimization_maxvles,
                                     1.0 / self.optimization_accuracy,
                                     self.codes)
            self.add(chromo)

        if self.optimization_code == "gray":
            self.compute_gray_codes()

        self._evaluate()

        self.average_fit = self.fitness / self.population_size
        self.best_fit = self.chromosomes[0].fitness
        self.worst_fit = self.chromosomes[-1].fitness
        self.median_fit = self.chromosomes[self.population_size >> 1].fitness

        this_population_size = len(self)

        self.info("Breeding")
        # Choose parents for breeding
        chromos = self.selection()
        # Breed, children will be appended
        for cross in self.crossings:
            cross(chromos)

        self.info("Mutating")
        # Mutate parents
        for mutnme, mutparams in self.mutations.items():
            if not mutparams["use"]:
                continue
            mut_pool = list(range(this_population_size))
            for _i_mut in range(int(this_population_size *
                                    mutparams["chromosomes"])):
                if not len(mut_pool):
                    break
                rand = numpy.random.choice(mut_pool)
                mutating = self.chromosomes[rand].copy()
                mutating.mutate(mutnme, int(this_population_size *
                                            mutparams["points"]),
                                mutparams["probability"])
                if self.optimization_code == "gray":
                    mutating.numeric = bin_to_num(
                        [mutating.binary], self.delimeter,
                        self.optimization_accuracy, self.codes)[0]
                self.add(mutating)
                mut_pool.remove(rand)

        fout = self.get_pickle_fout()
        if fout is not None:
            pickle.dump(self.chromosomes, fout,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def get_pickle_fin(self):
        if self.prev_state_fnme is None:
            return None
        try:
            return open(self.prev_state_fnme, "rb")
        except OSError:
            return None

    def get_pickle_fout(self):
        try:
            self.prev_state_fnme = "%s/chromosomes_%d_%.2f.pickle" % (
                root.common.snapshot_dir, self.generation, self.best_fit)
            return open(self.prev_state_fnme, "wb")
        except OSError:
            return None

    def evolve(self):
        """Evolve until completion.
        """
        try:
            while True:
                self.do_evolution_step()
                if self.on_after_evolution_step():
                    break
                self.generation += 1
        except KeyboardInterrupt:
            self.error("Evolution was interrupted")

    def on_after_evolution_step(self):
        """Called after an evolution step.

        Returns:
            True to stop evolution.
        """
        self.log_statistics()
        return False

    def log_statistics(self):
        self.info("Epochs completed %d: fitness: best=%.2f total=%.2f "
                  "average=%.2f median=%.2f worst=%.2f", self.generation + 1,
                  self.best_fit, self.fitness,
                  self.average_fit, self.median_fit, self.worst_fit)
