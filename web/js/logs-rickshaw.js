Rickshaw.Graph.Legend.prototype.initialize = function(args) {
    this.graph = args.graph;
    this.naturalOrder = args.naturalOrder;

    this.element = args.element;
    var jq_list = $(this.element).addClass(this.className).find("#rickshaw-legend");
    this.list = jq_list.get(0);
    if (this.list === undefined || args.force === true) {
        this.list = document.createElement('ul');
        this.list.id = "rickshaw-legend";
        this.element.appendChild(this.list);
        this.render();
        // Remove the "hidden" line, which is always comes last
        $(this.lines[this.lines.length - 1].element).css("visibility", "hidden");
        this.lines.length = this.lines.length - 1
        $(this.element).addClass(this.className).find("#rickshaw-legend")
            .data("Rickshaw.Graph.Legend", this);
        for (var index in this.lines) {
          var line = this.lines[index];
          line.series = [line.series]
        }
        this.graph_indices = {};
        this.graph_indices[this.graph.element.id] = 0;
        this.graphs = [this.graph]
    } else {
        var legend = jq_list.data("Rickshaw.Graph.Legend");
        this.lines = legend.lines;
        this.graph_indices = legend.graph_indices;
        this.graphs = legend.graphs;
        var series = this.graph.series.map( function(s) { return s } );
        if (!this.naturalOrder) {
			series = series.reverse();
		}
        for (var index in this.lines) {
          var line = this.lines[index];
          line.series.push(series[index]);
        }
        this.graph_indices[this.graph.element.id] = Object.keys(this.graph_indices).length;
        this.graphs.push(this.graph);
    }

    // we could bind this.render.bind(this) here
    // but triggering the re-render would lose the added
    // behavior of the series toggle
    this.graph.onUpdate( function() {} );
};

Rickshaw.Graph.Behavior.Series.Toggle = function(args) {

	this.legend = args.legend;
    this.graph = this.legend.graph;

	var self = this;
    var graph_index = self.legend.graph_indices[this.graph.element.id];

	this.addAnchor = function(line) {
    var anchor = $(line.element).find("#rickshaw-legend-toggle").get(0);
    if (anchor === undefined) {
        anchor = document.createElement('a');
        anchor.innerHTML = '&#10004;';
        anchor.classList.add('action');
        anchor.id = "rickshaw-legend-toggle";
        line.element.insertBefore(anchor, line.element.firstChild);
    }

		anchor.addEventListener( 'click', (function(e) {
			if (line.series[graph_index].disabled) {
				line.series[graph_index].enable();
				line.element.classList.remove('disabled');
			} else {
				if (this.graph.series.filter(function(s) { return !s.disabled }).length <= 1) return;
				line.series[graph_index].disable();
				line.element.classList.add('disabled');
			}

			self.graph.update();

		}).bind(this));

    var label = line.element.getElementsByTagName('span')[0];

    label.onclick = function(e) {

            var disableAllOtherLines = line.series[0].disabled;
            if ( ! disableAllOtherLines ) {
                    for ( var i = 0; i < self.legend.lines.length; i++ ) {
                            var l = self.legend.lines[i];
                            if ( line.series === l.series ) {
                                    // noop
                            } else if ( l.series[0].disabled ) {
                                    // noop
                            } else {
                                    disableAllOtherLines = true;
                                    break;
                            }
                    }
            }

            // show all or none
            if ( disableAllOtherLines ) {

                    // these must happen first or else we try ( and probably fail ) to make a no line graph
                    for (var index in line.series) {
                      line.series[index].enable();
                    }
                    line.element.classList.remove('disabled');

                    self.legend.lines.forEach(function(l){
                            if ( line.series === l.series ) {
                                    // noop
                            } else {
                               for (var index in l.series) {
                                 l.series[index].disable();
                               }
                               l.element.classList.add('disabled');
                            }
                    });

            } else {

                    self.legend.lines.forEach(function(l){
                      for (var index in l.series) {
                        l.series[index].enable();
                      }
                      l.element.classList.remove('disabled');
                    });

            }
            for (index in self.legend.graphs) {
              graphs[index].update();
            }
    };
	};

	if (this.legend) {

		var $ = jQuery;
		if (typeof $ != 'undefined' && $(this.legend.list).sortable) {

			$(this.legend.list).sortable( {
				start: function(event, ui) {
					ui.item.bind('no.onclick',
						function(event) {
							event.preventDefault();
						}
					);
				},
				stop: function(event, ui) {
					setTimeout(function(){
						ui.item.unbind('no.onclick');
					}, 250);
				}
			});
		}

		this.legend.lines.forEach( function(l) {
			self.addAnchor(l);
		} );
	}

	this._addBehavior = function() {

		this.graph.series.forEach( function(s) {

			s.disable = function() {

				if (self.graph.series.length <= 1) {
					throw('only one series left');
				}

				s.disabled = true;
			};

			s.enable = function() {
				s.disabled = false;
			};
		} );
	};

	this._addBehavior();

	this.updateBehaviour = function () { this._addBehavior() };

};

Rickshaw.Graph.Behavior.Series.Highlight = function(args) {

	this.legend = args.legend;
    this.graph = this.legend.graph;

	var self = this;

	var colorSafe = {};
	var activeLine = null;
    var graph_index = self.legend.graph_indices[this.graph.element.id];

	var disabledColor = args.disabledColor || function(seriesColor) {
		return d3.interpolateRgb(seriesColor, d3.rgb('#d8d8d8'))(0.8).toString();
	};

    this.addHighlightEvents = function (l) {

		l.element.addEventListener( 'mouseover', function(e) {

			if (activeLine) return;
			else activeLine = l;

			self.legend.lines.forEach( function(line) {

				if (l === line) {

					// if we're not in a stacked renderer bring active line to the top
					if (self.graph.renderer.unstack && (line.series[graph_index].renderer ? line.series[graph_index].renderer.unstack : true)) {

						var seriesIndex = self.graph.series.indexOf(line.series[graph_index]);
						line.originalIndex = seriesIndex;

						var series = self.graph.series.splice(seriesIndex, 1)[0];
						self.graph.series.push(series);
					}
					return;
				}

				colorSafe[line.series[graph_index].name] = colorSafe[line.series[graph_index].name] || line.series[graph_index].color;
				line.series[graph_index].color = disabledColor(line.series[graph_index].color);

			} );

			self.graph.update();

		}, false );

		l.element.addEventListener( 'mouseout', function(e) {
			if (!activeLine) return;
			else activeLine = null;
			self.legend.lines.forEach( function(line) {

				// return reordered series to its original place
				if (l === line && line.hasOwnProperty('originalIndex')) {

					var series = self.graph.series.pop();
					self.graph.series.splice(line.originalIndex, 0, series);
					delete line.originalIndex;
				}

				if (colorSafe[line.series[graph_index].name]) {
					line.series[graph_index].color = colorSafe[line.series[graph_index].name];
				}
			} );

			self.graph.update();

		}, false );
	};
    if (this.legend) {
		this.legend.lines.forEach( function(l) {
			self.addHighlightEvents(l);
		} );
	}

};

Rickshaw.Graph.Behavior.Series.Order = function(args) {
	this.legend = args.legend;

	var self = this;

	if (typeof window.jQuery == 'undefined') {
		throw "couldn't find jQuery at window.jQuery";
	}

	if (typeof window.jQuery.ui == 'undefined') {
		throw "couldn't find jQuery UI at window.jQuery.ui";
	}

	$(function() {
		$(self.legend.list).sortable( {
			containment: 'parent',
			tolerance: 'pointer',
			update: function( event, ui ) {
                for (var index in self.legend.graphs) {
                    var series = [];
                    $(self.legend.list).find('li').each( function(index, item) {
                        if (!item.series) return;
                        series.push(item.series);
                    } );

                    var graph = self.legend.graphs[index];
                    for (var i = graph.series.length - 1; i >= 0; i--) {
                        graph.series[i] = series.shift();
                    }
                    graph.update();
                }
			}
		} );
		$(self.legend.list).disableSelection();
	});

	//hack to make jquery-ui sortable behave
	this.legend.graph.onUpdate( function() {
		var h = window.getComputedStyle(self.legend.element).height;
		self.legend.element.style.height = h;
	} );
};
