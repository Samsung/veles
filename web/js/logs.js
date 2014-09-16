var graphs = [];
var graph_width = 600;
var graph_height = 380;
var table_rows = [[], []];
var animating_scroll = false;
var pending_scroll_offset = -1;

function formatDate(unix) {
  var date = new Date(unix * 1000);
  var month = date.getMonth();
  if (month < 10) {
    month = "0" + month;
  }
  var hours = date.getHours();
  if (hours < 10) {
    hours = "0" + hours;
  }
  var minutes = date.getMinutes();
  if (minutes < 10) {
    minutes = "0" + minutes;
  }
  var seconds = date.getSeconds();
  if (seconds < 10) {
    seconds = "0" + seconds;
  }
  var millisecs = date.getMilliseconds();
  if (millisecs < 10) {
    millisecs = "0" + millisecs;
  }
  if (millisecs < 100) {
    millisecs = "0" + millisecs;
  }
  return "" + date.getDate() + "." + month + " " +
      hours + ":" + minutes + ":" + seconds + "." + millisecs;
}

function setupUI() {
  if (ui_is_setup) {
    return;
  }
  ui_is_setup = true;
  $("#chart-loading-master").hide();
  $("#chart-loading-slave").hide();
  nodes.forEach(function(node) {
    if (node.id != "master") {
      $("select.slave-setter").append($.parseHTML("<option>" + node.id + "</option>"));
    }
  });
  if (nodes.length > 1) {
    $("select.slave-setter").children().first().attr("selected", "selected");
    $("select.slave-setter").selectmenu("refresh").selectmenu("widget").removeAttr("style");
  }
  $("#timescale").show();
  $("#preview").show();
  $("#slave-setter-horizontal-centering").show();
  $("#log-level").selectmenu({
    select: function(event, ui) {
      fetchLogs(ui.item.label);
    }
  });
  var palette = new Rickshaw.Color.Palette( { scheme: 'classic9' } );
  var graph_series = [[], []];
  var colors = event_names.map(function() { return palette.color(); });
  for (var index = 0; index < 2; index++) {
    for (var event in event_names) {
      graph_series[index].push({
          color: colors[event],
          data: series_data[index][event],
          name: event_names[event]
      });
    }
  }
  setupGraphs(graph_series);
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
      current_min_time = pos[0];
      current_max_time = pos[1];
      preview.slider({values: pos, min: min, max: max,
                     step: (max - min) / 100});
      time_scale = new_time_scale;
  });
  $("#time-interval").text(formatDate(current_min_time) + " - " + formatDate(current_max_time));
  setupLogTables();
}

function setupLogTables() {
  var tablesorter_options = {theme: 'jui', widthFixed: true, headers: {
      0: { sorter: "digit" },
      1: { sorter: "text" },
      2: { sorter: "text" },
      3: { sorter: false }
  }, widgets: ['uitheme', 'zebra', 'columns', 'stickyHeaders'],
  cancelSelection: true,
  widgetOptions: {
      zebra: [
          "ui-widget-content even",
          "ui-state-default odd"],
      stickyHeaders_attachTo : $("#logs-master"),

  },
  sortList: [[0, 0]],
  textExtraction: {
        0: function (node) {
            return $(node).attr("created");
        }
  }};
  $("#logs-contents-master").tablesorter(tablesorter_options).trigger('pagerComplete.tsSticky', null);
  tablesorter_options.widgetOptions.stickyHeaders_attachTo = $("#logs-slave");
  $("#logs-contents-slave").tablesorter(tablesorter_options).trigger('pagerComplete.tsSticky', null);
  $(".logs-contents").css("visibility", "visible");
  enumerateMasterTableRows();
  $("#logs-master").scroll(function(event) {
    if (!$("#sync-logs").is(":checked")) {
      return;
    }
    var offset = $("#logs-master").scrollTop();
    var rows = table_rows[0];
    var left = 0;
    var right = rows.length;
    while (left < right) {
      var middle = (left + right) >> 1;
      var val = rows[middle];
      if (val > offset) {
        right = middle;
      } else {
        left = (left == middle)? right : middle;
      }
    }
    offset = $($("#logs-contents-master > tbody > tr")[left]).find("td").attr("created");
    rows = table_rows[1];
    left = 0;
    right = rows.length;
    while (left < right) {
      var middle = (left + right) >> 1;
      var val = rows[middle];
      if (val > offset) {
        right = middle;
      } else {
        left = (left == middle)? right : middle;
      }
    }
    if (left > 0) {
      offset = $($("#logs-contents-slave > tbody > tr")[left - 1]).position().top - $("thead").height();
      if (offset < 0) {
        offset = 0;
      }
    } else {
      offset = 0;
    }
    scrollSlaveLogsDiv(offset);
  });
}

function scrollSlaveLogsDiv(offset) {
  var pos = $("#logs-slave").scrollTop();
  if (pos != offset) {
    if (!animating_scroll) {
      animating_scroll = true;
      $("#logs-slave").animate({scrollTop: offset}, 500, function() {
        animating_scroll = false;
        if (pending_scroll_offset >= 0) {
          offset = pending_scroll_offset;
          pending_scroll_offset = -1;
          scrollSlaveLogsDiv(offset);
        }
      });
    } else {
      pending_scroll_offset = offset;
    }
  }
}

function setupGraphs(graph_series) {
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


renderEvents = function() {
  setupUI();
  for (var index in graphs) {
    var graph = graphs[index];
    series_data[0][0][0] = {x: current_min_time - smallest_time, y: 0};
    series_data[0][0][1] = {x: current_max_time - smallest_time, y: 0};
    series_data[1][0][0] = {x: current_min_time - smallest_time, y: 0};
    series_data[1][0][1] = {x: current_max_time - smallest_time, y: 0};
    graph.window.xMin = current_min_time - smallest_time;
    graph.window.xMax = current_max_time - smallest_time;
    graph.update();
  }
}

$(document).ready(function() {
  if (!fetching_events[0] && !fetching_events[1] && nodes.length > 0) {
    renderEvents();
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
              current_min_time = ui.values[0] + smallest_time;
              current_max_time = ui.values[1] + smallest_time;
              fetchEvents();
            }
        } );
    } );
};


function makeVerbatimText(msg) {
  return "<pre>" + msg.replace(/</g, "&lt;").replace(/>/g, "&gt;") + "</pre>";
}


function enumerateMasterTableRows() {
  var table = $("#logs-contents-master");
  table_rows[0].length = 0;
  $("#logs-contents-master > tbody > tr").each(function() {
    table_rows[0].push($(this).position().top);
  });
}

function enumerateSlaveRows(logs) {
  table_rows[1].length = 0;
  for (var index in logs) {
    var record = logs[index];
    table_rows[1].push(record.created);
  }
}

function logsToHtml(logs) {
  var html = [];

  for (var index in logs) {
    var record = logs[index];
    var tr = "<tr class=\"log-record-" + record.levelname.toLowerCase() + "\">";
    tr += "<td created=\"" + record.created + "\">" + formatDate(record.created).replace(" ", "&nbsp;");
    tr += "</td><td>" + record.pathname + ":" + record.lineno + "</td><td>";
    tr += record.name + "</td><td>" + makeVerbatimText(record.message) + "</td></tr>\n";
    html.push(tr);
  }

  return $.parseHTML(html.join());
}

function renderLogs(logs, instance) {
  var html = logsToHtml(logs);
  var node = (instance == "master")? "master" : "slave";
  $("#logs-loading-" + node).hide();
  $("#logs-contents-" + node + " > tbody").empty().append(html);
  var table = $("#logs-contents-" + node);
  if (ui_is_setup) {
    table.trigger("update")
      .trigger("sorton", table.first().config.sortList)
      .trigger("appendCache")
      .trigger("applyWidgets");
    if (node == "master") {
      enumerateMasterTableRows();
    }
  }
  if (node == "slave") {
    enumerateSlaveRows(logs);
  }
}
