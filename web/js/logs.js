var graphs = [];
/*
render_events = function() {
  var instances = new Set();
  var names = new Set();
  var master = null;
  for (var index in events) {
    var event = events[index];
    instances.add(event.instance);
    names.add(event.name);
    smallest_time = Math.min(smallest_time, event.time);
    biggest_time = Math.max(biggest_time, event.time);
  }
  elapsed_time = biggest_time - smallest_time;
  time_scale = elapsed_time;
  names.forEach(function(name) {
    event_names_mapping[name] = event_names.push(name) - 1;
  });
  instances.forEach(function(instance) {
    if (instance != "master") {
      slaves_mapping[instance] = slaves.push(instance) - 1;
      $("select.slave-setter").append($.parseHTML("<option>" + instance + "</option>"));
    }
  });
  if (slaves.length > 0) {
    $("select.slave-setter").children().first().attr("selected", "selected");
    $("select.slave-setter").selectmenu("refresh").selectmenu("widget").removeAttr("style");
  }

  var palette = new Rickshaw.Color.Palette( { scheme: 'classic9' } );
  var graph_series = [[], []];
  var colors = event_names.map(function() { return palette.color(); });
  for (var index = 0; index < 2; index++) {
    for (var event in event_names) {
      graph_series[index].push({
          color: colors[event],
          data: seriesData[index][seriesData[index].push([]) - 1],
          name: event_names[event]
      });
    }
  }
  render_graphs(graph_series);
  update_nodes(0, elapsed_time);
  var scale_min = Math.log(0.001);
  var scale_max = Math.log(elapsed_time);
  $("#timescale").slider({
      min: scale_min,
      max: scale_max,
      step: (scale_max - scale_min) / 20}).on("slide", function(event, ui) {
         ui.value = Math.min(Math.max(ui.value, scale_min), scale_max);
         var new_time_scale = Math.exp(ui.value);
         if (new_time_scale == time_scale) {
           return;
         }
         var preview = $("#preview");
         var pos = preview.slider("values");
         var middle = (pos[0] + pos[1]) / 2;
         var length = (pos[1] - pos[0]);
         var ratio = new_time_scale / time_scale;
         var min = preview.slider("option", "min");
         var max = preview.slider("option", "max");
         var offset = middle * (1 - ratio);
         min = min * ratio + offset;
         max = max * ratio + offset;
         if (pos[1] > max) {
           pos[1] = max;
         }
         if (pos[0] < min) {
           pos[0] = min;
         }
         preview.slider({values: pos, min: min, max: max,
                         step: (max - min) / 100});
         time_scale = new_time_scale;
     });
}

function update_nodes(min, max) {
  update_node("master", min, max);
  update_node(slaves[selected_slave], min, max);
}

function update_node(node, min, max) {
  var node_index = node === "master"? 0 : 1;
  var series = seriesData[node_index];
  for (var index in series) {
    series[index].length = 0;
  }
  series[0].push({x: min, y: 0});
  for (var index in events) {
    var event = events[index];
    if (event.instance != node) {
      continue;
    }
    if (event.type === "single") {
      continue;
    }
    var time = event.time - smallest_time;
    if (time < min || time > max) {
      continue;
    }
    var data = series[event_names_mapping[event.name]];
    if (event.type === "begin") {
      data.push({x: time, y: 0});
    }
    var value = 1;
    if (event.height != undefined) {
      value = event.height;
    }
    meta = $.extend({}, event);
    delete meta._id;
    delete meta.time;
    delete meta.name;
    delete meta.session;
    delete meta.instance;
    delete meta.height;
    data.push({x: time, y: value, event: JSON.stringify(meta)});
    if (event.type === "end") {
      data.push({x: time, y: 0});
    }
  }
  series[0].push({x: max, y: 0});
  graphs[node_index].update();
}

$(document).ready(function() {
  if (events != null) {
    render_events();
  }
});


Rickshaw.Graph.HoverDetail.prototype.formatter = function(series, x, y, formattedX, formattedY, d) {
  return series.name + (d.value.event? ("  " + d.value.event) : "");
};


Rickshaw.Graph.RangeSlider = function(args) {
    var element = this.element = args.element;
    var graph = this.graph = args.graph;

    $(function() {
        $(element).slider( {
            range: true,
            min: 0,
            max: elapsed_time,
            values: [0, elapsed_time],
            slide: function(event, ui) {
                update_nodes(ui.values[0], ui.values[1]);
            }
        } );
    } );
};


function render_graphs(graph_series) {
  var graph_width = 600;
  var graph_height = 390;

  for (var index = 0; index < 2; index++) {
    graphs.push(new Rickshaw.Graph( {
      element: document.getElementById(index === 0? "chart-master" : "chart-slave"),
      width: graph_width,
      height: graph_height,
      renderer: 'area',
      stroke: true,
      preserve: true,
      stack: false,
      min: 0,
      max: 1.5,
      interpolation: "linear",
      series: graph_series[index]
    } ));
    graphs[index].render();
  }

  var preview = new Rickshaw.Graph.RangeSlider( {
      graph: graphs,
      element: document.getElementById('preview'),
  } );

  var hoverDetails = [];
  var legends = [];
  var shelvings = [];
  var orders = [];
  var highlighters = [];
  var ticksTreatment = 'glow';
  var xAxes = [];
  var yAxes = [];
  var controls = [];

  for (var index in graphs) {
      hoverDetails.push(new Rickshaw.Graph.HoverDetail( {
          graph: graphs[index],
          xFormatter: function(x) {
              return new Date((x + smallest_time) * 1000).toString();
          }
      } ));
      legends.push(new Rickshaw.Graph.Legend( {
          graph: graphs[index],
          element: document.getElementById('legend')
      } ));
      shelvings.push(new Rickshaw.Graph.Behavior.Series.Toggle( {
          legend: legends[index]
      } ));
      orders.push(new Rickshaw.Graph.Behavior.Series.Order( {
          legend: legends[index]
      } ));
      highlighters.push(new Rickshaw.Graph.Behavior.Series.Highlight( {
          legend: legends[index]
      } ));
      xAxes.push(new Rickshaw.Graph.Axis.Time( {
          graph: graphs[index],
          ticksTreatment: ticksTreatment,
          timeFixture: new Rickshaw.Fixtures.Time.Local()
      } ));
      xAxes[index].render();
      yAxes.push(new Rickshaw.Graph.Axis.Y( {
          graph: graphs[index],
          tickFormat: Rickshaw.Fixtures.Number.formatKMBT,
          ticksTreatment: ticksTreatment
      } ));
      yAxes[index].render();
  }
}

function logs_to_html(logs, level) {
  var html = "";

  for (var index in logs) {
    var record = logs[index];
  }

  return $.parseHTML(html);
}
*/
