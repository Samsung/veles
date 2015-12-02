#!/usr/bin/env python3
# encoding: utf-8


from jinja2 import Environment, FileSystemLoader
import os
import sys

import veles
from veles.loader.base import UserLoaderRegistry
from veles.logger import Logger
from veles.plotter import Plotter
from veles.workflow import Workflow
from veles.unit_registry import UnitRegistry
from veles.units import Unit
from veles.znicz.activation import Activation
from veles.znicz.nn_units import ForwardBase, GradientDescentBase,\
    MatchingObject


docs_source_dir = os.path.join(os.path.dirname(__file__), "source")
result_file_name_base = "manualrst_veles_units_kwargs"


class UnitsKeywordArgumentsGenerator(Logger):
    @staticmethod
    def get_condition_loader_with_labels(unit):
        return (isinstance(unit, UserLoaderRegistry)
                and str(unit).find("MSE") < 0)

    @staticmethod
    def get_condition_loader_with_targets(unit):
        return (isinstance(unit, UserLoaderRegistry)
                and str(unit).find("MSE") > 0)

    @staticmethod
    def get_condition_forwards(unit):
        return (isinstance(unit, MatchingObject) and
                issubclass(unit, ForwardBase) and
                not issubclass(unit, Activation))

    @staticmethod
    def get_condition_backwards(unit):
        return (isinstance(unit, MatchingObject) and
                issubclass(unit, GradientDescentBase) and
                not issubclass(unit, Activation))

    @staticmethod
    def get_condition_forward_activations(unit):
        return (isinstance(unit, MatchingObject) and
                issubclass(unit, ForwardBase) and
                issubclass(unit, Activation))

    @staticmethod
    def get_condition_backward_activations(unit):
        return (isinstance(unit, MatchingObject) and
                issubclass(unit, GradientDescentBase) and
                issubclass(unit, Activation))

    @staticmethod
    def get_condition_plotters(unit):
        return issubclass(unit, Plotter)

    @staticmethod
    def get_condition_kohonen(unit):
        return str(unit).find("kohonen") > 0

    @staticmethod
    def get_condition_rbm(unit):
        return str(unit).find("rbm") > 0

    @staticmethod
    def get_condition_working_units(unit):
        return (str(unit).find("rbm") < 0 and str(unit).find("kohonen") < 0 and
                not issubclass(unit, Plotter) and
                not issubclass(unit, Activation) and
                not issubclass(unit, GradientDescentBase) and
                not issubclass(unit, ForwardBase) and
                not isinstance(unit, UserLoaderRegistry))

    @staticmethod
    def get_condition_base_units(unit):
        return not issubclass(unit, Workflow)

    def run(self, output_path):
        def usorted(units):
            return sorted(units, key=lambda u: str(u))

        Environment(trim_blocks=True, lstrip_blocks=True,
                    loader=FileSystemLoader(docs_source_dir)) \
            .get_template(result_file_name_base + ".jrst") \
            .stream(origin=self, root_kwattrs=Unit.KWATTRS,
                    all_units=usorted(veles.__units__),
                    hidden_units=usorted(UnitRegistry.hidden_units)) \
            .dump(output_path, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(UnitsKeywordArgumentsGenerator().run(sys.argv[1]))
