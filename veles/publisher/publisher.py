# -*- coding: utf-8 -*-
"""
.. invisible:
     _   _ _____ _     _____ _____
    | | | |  ___| |   |  ___/  ___|
    | | | | |__ | |   | |__ \ `--.
    | | | |  __|| |   |  __| `--. \
    \ \_/ / |___| |___| |___/\__/ /
     \___/\____/\_____|____/\____/

Created on Apr 23, 2015

Publishing the experiment results to various media: wiki, file system (HTML,
PDF), IPython Notebook, etc.

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


import json
import logging
import os
import platform
from six import BytesIO, StringIO
from time import time
from zope.interface import implementer

from veles.config import root
from veles.distributable import TriviallyDistributable, IDistributable
from veles.loader import Loader
from veles.plotter import Plotter
from veles.publisher.registry import PublishingBackendRegistry
from veles.units import Unit, IUnit, nothing


@implementer(IUnit, IDistributable)
class Publisher(Unit, TriviallyDistributable):
    def __init__(self, workflow, **kwargs):
        super(Publisher, self).__init__(workflow, **kwargs)
        self._backends = dict(kwargs["backends"])
        self._backend_instances = {}
        self._templates = dict(kwargs.get("templates", {}))
        for key, val in self.templates.items():
            if os.path.exists(val):
                with open(val, "r") as fin:
                    val = fin.read()
            self.templates[key] = val
        self._include_plots = bool(kwargs.get("plots", True))
        self._matplotlib_backend = kwargs.get("matplotlib_backend", "cairo")
        self._matplotlib_packages_ = {}
        self._savefig_kwargs = dict(kwargs.get(
            "savefig_kwargs", {"transparent": True}))
        self._workflow_graph_kwargs = dict(kwargs.get(
            "workflow_graph_kwargs", {"with_data_links": True}))
        # Launcher sets self.workflow_graphs thanks to this flag
        self.wants_workflow_graph = True
        # Instance of veles.loader.Loader
        self._loader_unit = kwargs.get("loader")

    @property
    def backends(self):
        return self._backends

    @property
    def templates(self):
        return self._templates

    @property
    def include_plots(self):
        return self._include_plots

    @include_plots.setter
    def include_plots(self, value):
        if not isinstance(value, bool):
            raise TypeError("include_plots must be boolean")
        self._include_plots = value

    @property
    def matplotlib_backend(self):
        return self._matplotlib_backend

    @matplotlib_backend.setter
    def matplotlib_backend(self, value):
        if not isinstance(value, str):
            raise TypeError("matplotlib_backend must be a string")
        self._matplotlib_backend = value

    @property
    def savefig_kwargs(self):
        return self._savefig_kwargs

    @property
    def workflow_graph_kwargs(self):
        return self._workflow_graph_kwargs

    @property
    def loader_unit(self):
        return self._loader_unit

    @loader_unit.setter
    def loader_unit(self, value):
        if value is None:
            self._loader_unit = None
            return
        if not isinstance(value, Loader):
            raise TypeError(
                "loader_unit must be an instance of %s (got %s)" %
                (Loader, type(value)))
        self._loader_unit = value

    def initialize(self, **kwargs):
        try:
            import matplotlib
            matplotlib.use(self.matplotlib_backend)
            self._matplotlib_packages_ = Plotter.import_matplotlib()
        except ImportError:
            self.include_plots = False
            self.warning("Failed to import matplotlib: there will be no plots "
                         "in the published documents.")
        for backend_class, backend_kwargs in self.backends.items():
            self.debug("Creating %s...", backend_class)
            self._backend_instances[backend_class] = \
                PublishingBackendRegistry.backends[backend_class](
                    self.templates.get(backend_class), **backend_kwargs)

    def run(self):
        info = self.init_info()
        self.add_info(info)
        if self.logger.isEnabledFor(logging.DEBUG):
            self._debug_info(info)
        self.info("Publishing the results...")
        for backend_class, backend_kwargs in self.backends.items():
            self.debug("Rendering %s...", backend_class)
            self._backend_instances[backend_class].render(info)

    def init_info(self):
        self.info("Gathering the results...")
        info = {
            "plots": self._gather_plots() if self._include_plots else {},
            "workflow_graph": self.workflow_graphs,
            "name": self.workflow.name,
            "description": self.workflow.__doc__,
            "id": self.launcher.id,
            "python": "%s %s" % (platform.python_implementation(),
                                 platform.python_version()),
            "pid": os.getpid(),
            "logid": self.launcher.log_id,
            "config_root": root,
            "workflow_file": self.launcher.workflow_file,
            "config_file": self.launcher.config_file,
            "unit_run_times_by_class":
            dict(self.workflow.get_unit_run_time_stats()),
            "unit_run_times_by_name":
            dict(self.workflow.get_unit_run_time_stats(by_name=True))
        }
        sio = StringIO()
        root.print_(file=sio)
        info["config_text"] = sio.getvalue()
        workflow_dir = os.path.dirname(self.launcher.workflow_file)
        manifest_file = os.path.join(workflow_dir, "manifest.json")
        if os.access(manifest_file, os.R_OK):
            with open(manifest_file, "r") as fin:
                manifest = json.load(fin)
            image_path = os.path.join(workflow_dir, manifest["image"])
            if not os.access(image_path, os.R_OK):
                self.warning("Could not read %s", image_path)
                info["image"] = None
            else:
                with open(image_path, "rb") as fin:
                    info["image"] = {"name": manifest["image"],
                                     "data": fin.read()}
        else:
            info["image"] = None
        mins, secs = divmod(time() - self.launcher.start_time, 60)
        hours, mins = divmod(mins, 60)
        days, hours = divmod(hours, 24)
        info.update({"days": days, "hours": hours, "mins": mins, "secs": secs})
        if self.loader_unit is not None:
            unit = self.loader_unit
            if unit.has_labels:
                info.update({"labels": tuple(unit.labels_mapping),
                             "label_stats": (dict(unit.test_diff_labels),
                                             dict(unit.valid_diff_labels),
                                             dict(unit.train_diff_labels))})
            info.update({"class_lengths": tuple(unit.class_lengths),
                         "total_samples": unit.total_samples,
                         "epochs": unit.epoch_number,
                         "normalization": unit.normalization_type,
                         "normalization_parameters":
                         unit.normalization_parameters})
        return info

    def add_info(self, info):
        pass

    def _gather_plots(self):
        plots = {}
        for unit in self.workflow.units_in_dependency_order:
            if not isinstance(unit, Plotter):
                continue
            self.debug("Rendering \"%s\"...", unit.name)
            unit.set_matplotlib(self._matplotlib_packages_)
            unit.show_figure = nothing
            unit.fill()
            figure = unit.redraw()
            if not getattr(unit, "redraw_plot", True) or figure is None:
                continue
            plots[unit.name] = formats = {}
            for fmt in "png", "svg":
                rendered = BytesIO()
                figure.savefig(rendered, format=fmt, **self._savefig_kwargs)
                formats[fmt] = rendered.getvalue()
        return plots

    def _debug_info(self, info):
        plots_copy = info["plots"]
        del info["plots"]
        info["plots"] = {}
        for key in plots_copy:
            info["plots"][key] = {"png": "<data>", "svg": "<data>"}
        graph_copy = info["workflow_graph"]
        del info["workflow_graph"]
        info["workflow_graph"] = {"png": "<data>", "svg": "<data>"}
        image_copy = info["image"]
        if image_copy is not None:
            info["image"] = {"name": image_copy["name"], "data": "<data>"}
        self.debug("Info: %s", info)
        info["plots"] = plots_copy
        info["workflow_graph"] = graph_copy
        info["image"] = image_copy
