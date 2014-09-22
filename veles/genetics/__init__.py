"""
Created on Sep 8, 2014

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


from veles.genetics.simple import Chromosome, Population, schwefel
from veles.genetics.config import (Tune,
                                   ConfigChromosome,
                                   ConfigPopulation,
                                   fix_config)
