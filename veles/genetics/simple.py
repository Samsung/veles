"""
  _   _ _____ _     _____ _____
 | | | |  ___| |   |  ___/  ___|
 | | | | |__ | |   | |__ \ `--.
 | | | |  __|| |   |  __| `--. \
 \ \_/ / |___| |___| |___/\__/ /
  \___/\____/\_____|____/\____/

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


import numpy
import pickle
from twisted.internet import reactor

from veles.config import root
from veles.distributable import Pickleable
from veles.external.progressbar import ProgressBar
import veles.prng as prng


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


class InlineObject(object):
    pass


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
                 binary=None, numeric=None, rand=prng.get()):
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

        self.rand = rand

        self.optimization = InlineObject()
        self.optimization.choice = "betw"
        self.optimization.code = "float"

        self.minvles = minvles
        self.maxvles = maxvles
        if size:
            self.size = size
            self.binary = ""
            self.numeric = []
            for j in range(size):
                if self.optimization.choice == "or":
                    rand = self.rand.choice([minvles[j], maxvles[j]])
                    self.numeric.append(rand)
                elif type(minvles[j]) == float or type(maxvles[j]) == float:
                    rand = self.rand.randint(int(minvles[j] * accuracy),
                                             int(maxvles[j] * accuracy) + 1)
                    self.numeric.append(rand / accuracy)
                else:
                    rand = self.rand.randint(minvles[j], maxvles[j] + 1)
                    self.numeric.append(rand)
                    rand = int(rand * accuracy)
                if self.optimization.code == "gray":
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

    @property
    def valid(self):
        return True

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
            pos = self.rand.randint(1, len(self.binary) - 1)
            p_m = self.rand.rand()
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
        if self.optimization.code == "gray":
            mutant = ""
            for _ in range(n_points):
                self.fitness = None
                pos1 = self.rand.randint(len(self.binary))
                pos2 = self.rand.randint(len(self.binary))
                p_m = self.rand.rand()
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
                pos1 = self.rand.randint(len(self.numeric))
                pos2 = self.rand.randint(len(self.numeric))
                p_m = self.rand.rand()
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
            pos = self.rand.choice(mut_pool)
            if self.optimization.choice == "or":
                self.numeric[pos] = self.rand.choice(
                    [minvles[pos], maxvles[pos]])
            else:
                isint = (type(self.numeric[pos]) == int)
                diff = maxvles[pos] - minvles[pos]
                max_prob = minvles[pos] + diff / 2
                gauss = self.rand.normal(max_prob, numpy.sqrt(diff / 6))
                p_m = self.rand.rand()
                if p_m < probability:
                    if self.rand.random() < 0.5:
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
            pos = self.rand.choice(mut_pool)
            if self.optimization.choice == "or":
                self.numeric[pos] = self.rand.choice(
                    [minvles[pos], maxvles[pos]])
            else:
                isint = (type(self.numeric[pos]) == int)
                p_m = self.rand.rand()
                if p_m < probability:
                    rand = self.rand.uniform(minvles[pos], maxvles[pos])
                    if isint:
                        rand = int(rand)
                    self.numeric[pos] = rand
                mut_pool.remove(pos)
                if len(mut_pool) == 0:
                    break


class Population(Pickleable):
    """Base class for a species population.
    """

    def __init__(self, chromosome_class, optimization_size,
                 min_values, max_values, population_size, accuracy=0.00001,
                 rand=prng.get()):
        super(Population, self).__init__()

        self.rand = rand

        self.chromosome_class = chromosome_class

        self.optimization = InlineObject()
        self.optimization.choice = "betw"
        self.optimization.code = "float"
        self.optimization.size = optimization_size
        self.optimization.min_values = min_values
        self.optimization.max_values = max_values
        self.optimization.accuracy = accuracy

        self.chromosomes = []

        self.population_size = population_size

        self.fitness = None
        self.average_fit = None
        self.best_fit = None
        self.worst_fit = None
        self.median_fit = None
        self.prev_fitness = -1.0e30
        self.prev_average_fit = -1.0e30
        self.prev_best_fit = -1.0e30
        self.prev_worst_fit = -1.0e30
        self.prev_median_fit = -1.0e30

        self.roulette_select_size = 0.75
        self.random_select_size = 0.5
        self.tournament_size = 0.5
        self.tournament_select_size = 0.1

        self.crossing = InlineObject()
        self.crossing.pointed_crossings = 0.2
        self.crossing.pointed_points = 0.08
        self.crossing.pointed_probability = 1.0

        self.crossing.uniform_crossings = 0.15
        self.crossing.uniform_probability = 0.9

        self.crossing.arithmetic_crossings = 0.15
        self.crossing.arithmetic_probability = 0.9

        self.crossing.geometric_crossings = 0.2
        self.crossing.geometric_probability = 0.9

        self.crossing.pipeline = [
            self.cross_uniform, self.cross_arithmetic,
            self.cross_geometric]

        self.delimeter = None
        self.codes = None

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
        self.crossing_attempts = 10

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

    def new(self, size, minvles, maxvles, accuracy, codes, binary=None,
            numeric=None):
        population = self
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        kwargs["rand"] = self.rand
        return self.chromosome_class(**kwargs)

    def add(self, chromo):
        assert isinstance(chromo, Chromosome)
        self.chromosomes.append(chromo)

    def sort(self):
        """Sorts the population be fitness and
        truncates up to maximum population size.
        """
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        del self.chromosomes[self.population_size:]

    def evaluate(self, callback):
        """Sequential evaluation.
        """
        for i, u in enumerate(self):
            if u.fitness is None:
                self.info("Will evaluate chromosome number %d (%.2f%%)",
                          i, 100.0 * i / len(self))
                u.evaluate()
        callback()

    def select(self):
        """Current selection procedure.
        """
        return self.select_roulette()

    def select_roulette(self):
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
            rand = 100.0 * self.rand.rand()
            j = 0
            while rand > bound[j]:
                j += 1
            parents.append(self[j])
        return parents

    def select_random(self):
        """Random select for crossing.
        """
        parents = []
        for _ in range(int(len(self) * self.random_select_size)):
            rand = self.rand.randint(len(self))
            parents.append(self[rand])
        return parents

    def select_tournament(self):
        """Tournament select for crossing.
        """
        tournament_pool = []
        for _ in range(int(len(self) * self.tournament_size)):
            rand = self.rand.randint(len(self))
            j = 0
            while (j < len(tournament_pool) and
                    tournament_pool[j].fitness < self[rand].fitness):
                j += 1
            tournament_pool.insert(j, self[rand])
        return tournament_pool[:int(len(self) * self.tournament_select_size)]

    def _cross_with_attempts(self, parents, crossings, f_attempt):
        for _ in range(int(len(self) * crossings)):
            for i in range(self.crossing_attempts):
                sons = f_attempt(parents)
                if any(not son.valid for son in sons):
                    self.warning("Invalid crossing result detected, "
                                 "will retry (attempt number %d)", i + 1)
                    continue
                break
            else:
                self.warning("Unsuccessfull crossing, but will still use "
                             "the result of the last attempt")
            for son in sons:
                self.add(son)

    def cross_pointed(self, parents):
        """Genetic operator.
        """
        self._cross_with_attempts(parents, self.crossing.pointed_crossings,
                                  self._cross_pointed_attempt)

    def _cross_pointed_attempt(self, parents):
        rand1 = self.rand.randint(len(parents))
        parent1 = parents[rand1].binary
        rand2 = self.rand.randint(len(parents))
        parent2 = parents[rand2].binary
        cross_points = [0, ]
        l = 0
        for _ in range(int(len(self) * self.crossing.pointed_points)):
            while l in cross_points:
                l = self.rand.randint(1, len(parent1) - 1)
            j = 0
            while j < len(cross_points) and cross_points[j] < l:
                j += 1
            cross_points.insert(j, l)
        cross_points.append(len(parent1))
        cross1 = ""
        cross2 = ""
        i = 1
        while i <= self.crossing.pointed_points + 1:
            if i % 2 == 0:
                cross1 += parent1[cross_points[i - 1]:cross_points[i]]
                cross2 += parent2[cross_points[i - 1]:cross_points[i]]
            else:
                cross1 += parent2[cross_points[i - 1]:cross_points[i]]
                cross2 += parent1[cross_points[i - 1]:cross_points[i]]
            i += 1
        (num1, num2) = bin_to_num([cross1, cross2], self.dl,
                                  self.optimization.accuracy, self.codes)
        chromo_son1 = self.new(
            0, self.minvles, self.maxvles,
            1.0 / self.optimization.accuracy, self.codes, cross1, num1)
        chromo_son2 = self.new(
            0, self.minvles, self.maxvles,
            1.0 / self.optimization.accuracy, self.codes, cross2, num2)
        chromo_son1.size = len(chromo_son1.numeric)
        chromo_son2.size = len(chromo_son2.numeric)
        return chromo_son1, chromo_son2

    def cross_uniform(self, parents):
        self._cross_with_attempts(parents, self.crossing.uniform_crossings,
                                  self._cross_uniform_attempt)

    def _cross_uniform_attempt(self, parents):
        if self.optimization.code == "gray":
            rand1 = self.rand.randint(len(parents))
            parent1 = parents[rand1].binary
            rand2 = self.rand.randint(len(parents))
            parent2 = parents[rand2].binary
            cross = ""
            for i in range(len(parent1)):
                rand = self.rand.uniform(0, 2)
                if rand < 1:
                    cross += parent1[i]
                else:
                    cross += parent2[i]
            numeric = bin_to_num([cross], self.dl,
                                 self.optimization.accuracy, self.codes)[0]
            chromo_son = self.new(
                0, self.minvles, self.maxvles,
                1.0 / self.optimization.accuracy, self.codes, cross,
                numeric)
        else:
            rand1 = self.rand.randint(len(parents))
            parent1 = parents[rand1].numeric
            rand2 = self.rand.randint(len(parents))
            parent2 = parents[rand2].numeric
            cross = []
            for i in range(len(parent1)):
                rand = self.rand.uniform(0, 2)
                if rand < 1:
                    cross.append(parent1[i])
                else:
                    cross.append(parent2[i])
            chromo_son = self.new(
                0, self.optimization.min_values,
                self.optimization.max_values,
                1.0 / self.optimization.accuracy, self.codes, None, cross)
        return (chromo_son,)

    def cross_arithmetic(self, parents):
        """Arithmetical crossingover.
        """
        self._cross_with_attempts(parents, self.crossing.arithmetic_crossings,
                                  self._cross_arithmetic_attempt)

    def _cross_arithmetic_attempt(self, parents):
        rand1 = self.rand.randint(0, len(parents))
        parent1 = parents[rand1].numeric
        rand2 = self.rand.randint(0, len(parents))
        parent2 = parents[rand2].numeric
        cross1 = []
        cross2 = []
        for i in range(len(parent1)):
            a = self.rand.random()
            if self.optimization.choice == "or":
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
        if self.optimization.code == "gray":
            (bin1, bin2) = (num_to_bin(cross1, self.optimization.accuracy,
                                       self.codes),
                            num_to_bin(cross2, self.optimization.accuracy,
                                       self.codes))
        else:
            (bin1, bin2) = ("", "")
        chromo1 = self.new(0, self.optimization.min_values,
                           self.optimization.max_values,
                           1.0 / self.optimization.accuracy,
                           self.codes, bin1, cross1)
        chromo2 = self.new(0, self.optimization.min_values,
                           self.optimization.max_values,
                           1.0 / self.optimization.accuracy,
                           self.codes, bin2, cross2)
        return (chromo1, chromo2)

    def cross_geometric(self, parents):
        """Geometrical crossingover.
        """
        self._cross_with_attempts(parents, self.crossing.geometric_crossings,
                                  self._cross_geometric_attempt)

    def _cross_geometric_attempt(self, parents):
        cross = []
        rand1 = self.rand.randint(len(parents))
        parent1 = parents[rand1].numeric
        rand2 = self.rand.randint(len(parents))
        parent2 = parents[rand2].numeric
        for i in range(len(parent1)):
            if self.optimization.choice == "or":
                if self.rand.random() > 0.5:
                    cross.append(parent1[i])
                else:
                    cross.append(parent2[i])
            else:
                # correct1 is used to invert [-x1; -x2] to [x2; x1]
                correct1 = -1 if self.optimization.max_values[i] < 0 else 1
                # correct2 is used to alter [-x1; x2] to [0; x2+x1]
                if self.optimization.min_values[i] > 0 or correct1 == -1:
                    correct2 = 0
                else:
                    correct2 = -self.optimization.min_values[i]
                a = self.rand.rand()
                gene = (correct1 * (numpy.power(
                    correct1 * parent1[i] + correct2, a) * numpy.power(
                    correct1 * parent2[i] + correct2, (1 - a)) - correct2))
                if type(parent1[i]) == int:
                    gene = int(gene)
                cross.append(gene)
        binary = ""
        if self.optimization.code == "gray":
            binary = num_to_bin(cross, self.optimization.accuracy,
                                self.codes)
        chromo_son = self.new(0, self.optimization.min_values,
                              self.optimization.max_values,
                              1.0 / self.optimization.accuracy,
                              self.codes, binary, cross)
        return (chromo_son,)

    def compute_gray_codes(self):
        max_abs_x = 0
        for i in range(self.optimization.size):
            if numpy.fabs(self.optimization.min_values[i]) > max_abs_x:
                max_abs_x = numpy.fabs(self.optimization.min_values[i])
            if numpy.fabs(self.optimization.max_values[i]) > max_abs_x:
                max_abs_x = numpy.fabs(self.optimization.max_values[i])
        max_coded_int = int(max_abs_x / self.optimization.accuracy)
        # Length of code of one int number
        self.delimeter = int(numpy.log2(max_coded_int))
        self.codes = gray(self.delimeter)
        # +1 symbol 1/0 for positive/negative
        self.delimeter += 1

    def do_evolution_step(self, callback):
        """Evolves the population (one step) asynchronously.

        callback will be called after the step is finished.
        """
        fin = self.get_pickle_fin()
        if fin is not None:
            self.chromosomes = pickle.load(fin)
            fin.close()

        chromo_count = self.population_size - len(self.chromosomes)
        if chromo_count > 0:
            self.info("Creating %d chromosomes...", chromo_count)
            for _ in ProgressBar(term_width=20)(range(chromo_count)):
                chromo = self.new(self.optimization.size,
                                  self.optimization.min_values,
                                  self.optimization.max_values,
                                  1.0 / self.optimization.accuracy,
                                  self.codes)
                self.add(chromo)

        if self.optimization.code == "gray":
            self.compute_gray_codes()

        def continue_callback():
            self.sort()  # kill excessive worst ones
            self.fitness = sum(u.fitness for u in self)

            self.average_fit = self.fitness / self.population_size
            self.best_fit = self.chromosomes[0].fitness
            self.worst_fit = self.chromosomes[-1].fitness
            self.median_fit = \
                self.chromosomes[self.population_size // 2].fitness

            this_population_size = len(self)

            self.info("Breeding")
            # Choose parents for breeding
            chromos = self.select()
            # Breed, children will be appended
            for cross in self.crossing.pipeline:
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
                    rand = self.rand.choice(mut_pool)
                    mutating = self.chromosomes[rand].copy()
                    mutating.mutate(mutnme, int(this_population_size *
                                                mutparams["points"]),
                                    mutparams["probability"])
                    if self.optimization.code == "gray":
                        mutating.numeric = bin_to_num(
                            [mutating.binary], self.delimeter,
                            self.optimization.accuracy, self.codes)[0]
                    self.add(mutating)
                    mut_pool.remove(rand)

            fout = self.get_pickle_fout()
            if fout is not None:
                pickle.dump(self.chromosomes, fout,
                            protocol=pickle.HIGHEST_PROTOCOL)
            callback()

        self.evaluate(continue_callback)

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

    def _evolve(self):
        """Evolve until the completion.
        """
        repeat = [False]  # matters only for standalone evolution

        def after_evolution_step():
            if not self.on_after_evolution_step():
                if reactor.running:
                    reactor.callLater(0, self._evolve)
                else:
                    repeat[0] = True
            self.generation += 1

        try:
            self.do_evolution_step(after_evolution_step)
        except KeyboardInterrupt:
            self.error("Evolution was interrupted")

        return repeat[0]

    def evolve(self):
        """This method can be overriden.
        """
        while self._evolve():
            # Evolve until convergence in standalone mode
            pass
        # On master, self._evolve will schedule population evaluation
        # and leave this function immediately.

    def on_after_evolution_step(self):
        """Called after an evolution step.

        Returns:
            True to stop evolution.
        """
        self.log_statistics()
        # Conservative stop condition
        if (self.prev_fitness >= self.fitness and
                self.prev_average_fit >= self.average_fit and
                self.prev_best_fit >= self.best_fit and
                self.prev_worst_fit >= self.worst_fit and
                self.prev_median_fit >= self.median_fit):
            self.info("No population improvement detected, "
                      "will stop evolution")
            return True
        self.prev_fitness = self.fitness
        self.prev_average_fit = self.average_fit
        self.prev_best_fit = self.best_fit
        self.prev_worst_fit = self.worst_fit
        self.prev_median_fit = self.median_fit
        return False

    def log_statistics(self):
        self.info("Generations completed %d: fitness: best=%.2f total=%.2f "
                  "average=%.2f median=%.2f worst=%.2f", self.generation + 1,
                  self.best_fit, self.fitness,
                  self.average_fit, self.median_fit, self.worst_fit)
