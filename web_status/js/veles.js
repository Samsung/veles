/*!
 * VELES web status interaction.
 * Copyright 2013 Samsung Electronics
 * Licensed under Samsung Proprietary License.
 */

function renderGraphviz(desc) {
  var result;
  try {
    var raw_result = Viz(desc, "svg", "circo");
    result = $.parseXML(raw_result);
  } catch(e) {
    result = e;
  }
  return result;
}

var updating = false;
var active_workflow_id = null;
var detailed_workflow_id = null;
var listed_workflows = null;
var svg_cache = {};

function updateUI() {
  if (updating) {
    return;
  }
  updating = true;
  console.log("Update started");
  var msg = {
    request: "workflows",
    args: ["name", "master", "slaves", "time", "user", "graph", "description",
           "plots", "custom_plots", "log_id", "log_addr"]
  };
  $.ajax({
    url: "service",
    type: "POST",
    data: JSON.stringify(msg),
    contentType: "application/json; charset=utf-8",
      async: true,
      success: function(ret) {
        console.log("Received response");
        listed_workflows = ret;
        if (!ret) {
          updating = false;
          console.log("Server returned an empty response " + ret);
          return;
        }
        var workflows = Object.keys(ret).map(function(key) {
          return { "key": key, "value": ret[key] };
        });
        if (workflows.length == 0) {
          updating = false;
          return;
        }
        workflows.sort(function(a, b) {
          return a.value.name > b.value.name;
        });
        if (active_workflow_id == null || !(active_workflow_id in ret)) {
          active_workflow_id = workflows[0].key;
        }
        var items = '';
        workflows.forEach(function(pair) {
          var workflow = pair.value;
          if (typeof svg_cache[pair.key] === "undefined") {
            svg_cache[pair.key] =
              $(renderGraphviz(workflow.graph)).find("svg");
          }
          listed_workflows[pair.key].svg = svg_cache[pair.key];
          var svg = listed_workflows[pair.key].svg.clone();
          svg.attr("class", "media-object pull-left");
          svg.attr("width", "100").attr("height", 100);
          items += '<li class="list-group-item media list-item-media';
          if (active_workflow_id == pair.key) {
            items += " active";
          }
          items += '" id="';
          items += pair.key;
          items += '">\n';
          items += svg.wrap('<div>').parent().html();
          items += '<div class="media-body graceful-overflow">\n';
          items += '<h4 class="list-group-item-heading"><a href="#" ';
          items += 'onclick="activateListItem(\'';
          items += pair.key;
          items += '\')">';
          items += workflow.name;
          items += '</a></h4>\n';
          items += '<a class="view-plots" href="';
          items += workflow.plots;
          items += '" target="_blank">plots</a>';
          items += '<span class="view-plots">&nbsp;</span>\n';
          items += '<a class="view-plots" href=log_viewer.html?id="';
          items += workflow.log_id;
          items += '"target="_blank">logs</a><br/>\n';
          items += '<span class="list-group-item-text">Master: ';
          items += '<a href="#"><strong>';
          items += workflow.master;
          items += '</strong></a><br/>\n';
          items += '<span class="badge pull-right">';
          var online_slaves = 0;
          for (var slave in workflow.slaves) {
            if (workflow.slaves[slave].state != 'Offline') {
              online_slaves++;
            }
          }
          items += online_slaves;
          items += '</span>\n';
          items += 'Slaves: ';
          slaves = Object.keys(workflow.slaves).map(function(key) {
            return { "key": key, "value": workflow.slaves[key] };
          });
          slaves.sort(function(a, b) {
            return a.value.host > b.value.host;
          });
          listed_workflows[pair.key].slaves = slaves;
          slaves.forEach(function(slave) {
            items += '<a href="#"><strong>';
            items += slave.value.host;
            items += '</strong></a>, ';
          });
          if (slaves.length > 0) {
            items = items.substring(0, items.length - 2);
          }
          items += '<br/>\n';
          items += 'Time running: <strong>';
          items += workflow.time;
          items += '</strong><br/>\n';
          items += 'Started by: <i class="glyphicon glyphicon-user">';
          items += '<a href="#"><strong>';
          items += workflow.user;
          items += '</strong></a></i></span>\n';
          items += '</div>\n';
          items += '</li>\n';
        });
        free_svgs = [];
        for (var key in svg_cache) {
          if (typeof listed_workflows[key] === "undefined") {
            free_svgs.push(key);
          }
        }
        for (var key in free_svgs) {
          delete svg_cache[key];
        }
        objs = $.parseHTML(items);
        $("#list-loading-indicator").remove();
        $("#workflow-list").empty().append(objs);
        console.log("Finished update");
        setTimeout(activateListItem, 0, active_workflow_id);
        updating = false;
      }
  });
}

function activateListItem(item_id) {
  if (active_workflow_id != item_id) {
    console.log("Switching items in the list");
    $("#" + active_workflow_id).removeClass("active");
    active_workflow_id = item_id;
    $("#" + item_id).addClass("active");
  }
  var full_update = true;
  if (detailed_workflow_id == item_id) {
    full_update = false;
  } else {
    detailed_workflow_id = item_id;
  }
  var workflow = listed_workflows[item_id];
  var details = "";
  if (full_update) {
    details += '<div class="detailed-description">\n';
    details += '<div class="panel panel-borderless">\n';
    details += '<div class="panel-heading details-panel-heading">';
    details += 'Actions</div>\n';
    details += '<div class="btn-group btn-group-justified">\n';
    details += '<div class="btn-group">\n';
    details += '<button type="button" class="btn btn-default" ';
    details += 'onclick="showPlots(\'';
    details += item_id;
    details += '\')">View plots';
    details += '</button>\n';
    details += '</div>\n';
    details += '<div class="btn-group">\n';
    details += '<button type="button" class="btn btn-default" ';
    details += 'onclick="showLogs(\'';
    details += item_id;
    details += '\')">View logs';
    details += '</button>\n';
    details += '</div>\n';
    details += '<div class="btn-group">\n';
    details += '<button type="button" class="btn btn-default">Suspend';
    details += '</button>\n';
    details += '</div>\n';
    details += '<div class="btn-group">\n';
    details += '<button type="button" class="btn btn-danger">Cancel</button>\n';
    details += '</div>\n';
    details += '</div>\n';
    details += '</div>\n';
    details += '<div class="panel panel-borderless">\n';
    details += '<div class="panel-heading details-panel-heading">Slaves';
    details += '</div>\n';
    details += '<div class="panel panel-default panel-margin-zero" ';
    details += 'id="slaves-table">\n';
  }
  details += '<table class="table table-condensed">\n';
  details += '<thead>\n';
  details += '<tr>\n';
  details += '<th>ID</th>\n';
  details += '<th>Host</th>\n';
  details += '<th class="center-cell">Power</th>\n';
  details += '<th class="center-cell">Status</th>\n';
  details += '<th class="center-cell">Actions</th>\n';
  details += '</tr>\n';
  details += '</thead>\n';
  details += '<tbody>\n';
  var max_power = 1;
  workflow.slaves.forEach(function(slave_pair) {
    var power = slave_pair.value.power;
    if (max_power < power) {
      max_power = power;
    }
  });
  var online_slaves = 0;
  workflow.slaves.forEach(function(slave_pair) {
    var slave = slave_pair.value;
    details += '<tr class="';
    switch (slave.state) {
      case "Working":
        details += "success";
        break;
      case "Waiting":
        details += "warning";
        break;
      case "Offline":
        details += "danger";
        break;
      default:
        break;
    }
    details += '">\n';
    details += '<td><div class="slave-id graceful-overflow">';
    details += slave_pair.key;
    details += '</div></td>\n';
    details += '<td><div class="slave-host graceful-overflow"><a href="#">';
    details += slave.host;
    details += '</a>';
    if (slave.host == workflow.master) {
      details += ' <span class="glyphicon glyphicon-flag"></span>';
    }
    details += '</div></td>\n';
    details += '<td class="power">';
    details += '<div class="progress"><div class="progress-bar ';
    var pwr = slave.power / max_power;
    if (pwr >= 0.7) {
      details += 'progress-bar-success';
    } else if (pwr >= 0.4) {
      details += 'progress-bar-warning';
    } else {
      details += 'progress-bar-danger';
    }
    details += '" role="progressbar" aria-valuenow="';
    details += slave.power.toFixed(0);
    details += '" aria-valuemin="0" aria-valuemax="';
    details += max_power.toFixed(0);
    details += '" style="width: ';
    var spp = slave.power * 100 / max_power;
    details += spp.toFixed(0);
    details += '%;">';
    if (pwr >= 0.5) {
      details += slave.power.toFixed(0);
    }
    details += '</div>';
    if (pwr < 0.5) {
        details += '<div class="progress-bar progress-overflow" ';
        details += 'role="progressbar" style="width: ';
        details += (100 - spp).toFixed(0);
        details += '%;">'
        details += slave.power.toFixed(0);
        details += '</div>';
    }
    details += '</div>';
    details += '</td>\n';
    details += '<td class="center-cell">';
    details += slave.state;
    details += '</td>\n';
    details += '<td class="center-cell">\n';
    if (slave.state != 'Offline') {
      online_slaves++;
      details += '<a href="#"><span class="glyphicon glyphicon-pause"></span></a>\n';
      details += '<a href="#"><span class="glyphicon glyphicon-remove"></span></a>\n';
      details += '<a href="#"><span class="glyphicon glyphicon-info-sign"></span></a>\n';
    }
    details += '</td>\n';
    details += '</tr>\n';
  });
  details += '</tbody>\n';
  details += '</table>\n';
  if (full_update) {
    details += '</div>\n';
    details += '</div>\n';
    details += '</div>\n';
    details += '<div class="detailed-text"><h3 class="media-heading">';
    details += workflow.name;
    details += '</h3>\n';
    details += workflow.description;
    details += 'This workflow is managed by ';
    details += '<i class="glyphicon glyphicon-user"><a href="#"">';
    details += workflow.user;
    details += '</a></i> on <a href="#">';
    details += workflow.master;
    details += "</a> and has ";
    details += online_slaves;
    details += ' nodes (';
    details += Object.keys(workflow.slaves).length - online_slaves;
    details += ' offline).<br/>';
    details += "Log ID: <strong>" + workflow.log_id + "</strong><br/>";
    details += 'ØMQ endpoints for custom plots:<br/><strong>';
    details += workflow.custom_plots;
    details += '</strong><br/><br/>';
    var svg = workflow.svg.clone();
    svg.attr("id", "workflow-svg");
    details += svg.wrap('<div>').parent().html();
    details += '</div>\n';
  }
  objs = $.parseHTML(details);
  $("#details-loading-indicator").remove();
  if (full_update) {
    $('#workflow-details').empty().append(objs);
  } else {
    $('#slaves-table').empty().append(objs);
  }
}

function showPlots(item_id) {
  var workflow = listed_workflows[item_id];
  var win = window.open(workflow.plots, '_blank');
  win.focus();
}

function showLogs(item_id) {
  var workflow = listed_workflows[item_id];
  var win = window.open(workflow.plots, '_blank');
  win.focus();
}

$(window).load(function() {
  setInterval(updateUI, 2000);
  setTimeout(updateUI, 0);
});
