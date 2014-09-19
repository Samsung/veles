/*!
 * VELES web status interaction.
 * Copyright 2013 Samsung Electronics
 * Licensed under Samsung Proprietary License.
 */

function renderGraphviz(desc) {
  var result;
  try {
    var raw_result = Viz(desc, "svg", "neato", ["-n"]);
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
var jobs_per_minute = {};

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
      success: function(result) {
        console.log("Received response", result);
        if (!result || !result.result) {
          updating = false;
          console.log("Server returned an empty response, skipped");
          return;
        }
        result = result.result;
        listed_workflows = result;
        var workflows = Object.keys(result).map(function(key) {
          return { "key": key, "value": result[key] };
        });
        if (workflows.length == 0) {
          updating = false;
          return;
        }
        workflows.sort(function(a, b) {
          return a.value.name > b.value.name;
        });
        if (active_workflow_id == null || !(active_workflow_id in result)) {
          active_workflow_id = workflows[0].key;
        }
        var items = '';
        workflows.forEach(function(pair) {
          var workflow = pair.value;
          slaves = Object.keys(workflow.slaves).map(function(key) {
            return { "key": key, "value": workflow.slaves[key] };
          });
          slaves.sort(function(a, b) {
            return a.value.host > b.value.host;
          });
          listed_workflows[pair.key].slaves = slaves;
          var jobs = 0;
          for (var slave in workflow.slaves) {
            jobs += workflow.slaves[slave].value.jobs;
          }
          if (!(pair.key in jobs_per_minute)) {
            jobs_per_minute[pair.key] = [];
          }
          jobs_per_minute[pair.key].push({"time": new Date().getTime() / 60000,
                                          "jobs": jobs});
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
          items += '<a class="view-plots" href="';
          items += workflow.plots;
          items += '" target="_blank">plots</a>';
          items += '<span class="view-plots">&nbsp;</span>\n';
          items += '<a class="view-plots" href=log_viewer.html?id="';
          items += workflow.log_id;
          items += '"target="_blank">logs</a>\n';
          items += '<h4 class="list-group-item-heading graceful-overflow"><a href="#" ';
          items += 'onclick="activateListItem(\'';
          items += pair.key;
          items += '\')">';
          items += workflow.name;
          items += '</a></h4>\n';
          items += '<span class="list-group-item-text">Master: ';
          items += '<a class="graceful-overflow" href="#"><strong>';
          items += workflow.master;
          items += '</strong></a><br/>\n';
          items += '<span class="badge pull-right">';
          var online_slaves = 0;
          for (var slave in workflow.slaves) {
            if (workflow.slaves[slave].value.state != 'Offline') {
              online_slaves++;
            }
          }
          items += online_slaves;
          items += '</span>\n';
          items += 'Slaves: ';
          var jpmp = jobs_per_minute[pair.key];
          if (jobs_per_minute[pair.key].length >= 5) {
            var latest = jpmp.slice(-1)[0];
            var now = latest.time;
            var then = now;
            var offset = 2;
            var jobs_diff = 0;
            while (now - then <= 10 && offset <= jpmp.length) {
              var measure = jpmp[jpmp.length - offset];
              then = measure.time;
              jobs_diff = measure.jobs;
              offset++;
            }
            jobs_diff = latest.jobs - jobs_diff;
            if (now > then) {
              items += (jobs_diff / (now - then)).toFixed(0);
            } else {
              items += 'N/A';
            }
          } else {
            items += 'N/A';
          }
          items += ' jpm<br/>\n';
          items += 'Time running: <strong>';
          items += workflow.time;
          items += '</strong><br/>\n';
          items += 'Started by: <i class="glyphicon glyphicon-user">';
          items += '<a href="#" class="graceful-overflow"><strong>';
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
        $("#workflows-list").empty().append(objs);
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
    $("#indicator").show();
  }
  var workflow = listed_workflows[item_id];
  if (full_update) {
    $("#button-plots").off("click").on("click", function() { showPlots(item_id) });
    $("#button-logs").off("click").on("click", function() { showLogs(item_id) });
    $("#button-plots").off("click").on("click", function() { });
    $("#button-plots").off("click").on("click", function() { });
  }
  var max_power = 1;
  var mean_jobs = 0;
  var online_slaves = 0;
  workflow.slaves.forEach(function(slave_pair) {
    var slave = slave_pair.value;
    var power = slave.power;
    if (max_power < power) {
      max_power = power;
    }
    if (slave.state != 'Offline') {
      mean_jobs += slave_pair.value.jobs;
      online_slaves++;
    }
  });
  mean_jobs /= online_slaves;
  var jobs_stddev = 0;
  if (online_slaves > 1) {
    workflow.slaves.forEach(function(slave_pair) {
      var slave = slave_pair.value;
      var delta = slave.jobs - mean_jobs;
      jobs_stddev += delta * delta;
    });
    jobs_stddev = Math.sqrt(jobs_stddev / (online_slaves - 1));
  }
  var rows = "";
  workflow.slaves.forEach(function(slave_pair) {
    var slave = slave_pair.value;
    rows += '<tr class="';
    var line_style = "";
    switch (slave.state) {
      case "Working":
        line_style = "success";
        break;
      case "Waiting":
        line_style = "warning";
        break;
      case "Offline":
        line_style = "danger";
        break;
      default:
        break;
    }
    if (slave.jobs < mean_jobs - jobs_stddev * 2 && line_style == "success") {
      line_style = "warning";
    }
    rows += line_style + '">\n';
    rows += '<td><div class="slave-id graceful-overflow">';
    rows += slave_pair.key;
    rows += '</div></td>\n';
    rows += '<td><div class="slave-host graceful-overflow"><a href="#">';
    rows += slave.host;
    rows += '</a>';
    if (slave.host == workflow.master) {
      rows += ' <span class="glyphicon glyphicon-flag"></span>';
    }
    rows += '</div></td>\n<td>';
    rows += '<div class="progress"><div class="progress-bar ';
    var pwr = slave.power / max_power;
    if (pwr >= 0.7) {
      rows += 'progress-bar-success';
    } else if (pwr >= 0.4) {
      rows += 'progress-bar-warning';
    } else {
      rows += 'progress-bar-danger';
    }
    rows += '" role="progressbar" aria-valuenow="';
    rows += slave.power.toFixed(0);
    rows += '" aria-valuemin="0" aria-valuemax="';
    rows += max_power.toFixed(0);
    rows += '" style="width: ';
    var spp = slave.power * 100 / max_power;
    rows += spp.toFixed(0);
    rows += '%;">';
    if (pwr >= 0.5) {
      rows += slave.power.toFixed(0);
    }
    rows += '</div>';
    if (pwr < 0.5) {
        rows += '<div class="progress-bar progress-overflow" ';
        rows += 'role="progressbar" style="width: ';
        rows += (100 - spp).toFixed(0);
        rows += '%;">'
        rows += slave.power.toFixed(0);
        rows += '</div>';
    }
    rows += '</div>';
    rows += '</td>\n<td class="center-cell">';
    rows += slave.state;
    rows += '</td>\n<td class="center-cell">\n';
    if (slave.state != 'Offline') {
      rows += '<a href="#"><span class="glyphicon glyphicon-pause"></span></a>\n';
      rows += '<a href="#"><span class="glyphicon glyphicon-remove"></span></a>\n';
      rows += '<a href="#"><span class="glyphicon glyphicon-info-sign"></span></a>\n';
    }
    rows += '</td>\n';
    rows += '</tr>\n';
  });
  if (rows != "") {
    rows = $.parseHTML(rows);
    $("#slaves-table > table > tbody").empty().append(rows);
    $("#slaves-panel").show();
  } else {
    $("#slaves-panel").hide();
  }

  $("#workflow-name").text(workflow.name);
  $("#workflow-description").html(workflow.description);
  $("#workflow-owner").text(workflow.user);
  $("#workflow-master-host").text(workflow.master);
  $("#workflow-online-nodes").text(online_slaves);
  $("#workflow-offline-nodes").text(Object.keys(workflow.slaves).length - online_slaves);
  $("#workflow-log-id").text(workflow.log_id);
  $("#workflow-log-id").attr("href", "logs.html?session=" + workflow.log_id);
  $("#workflow-zeromq-plotting-endpoints").html(workflow.custom_plots);
  $("#content").show();
  if (full_update) {
    $("#workflow-svg-container-fill").resize();
    $("#workflow-svg-container").empty().append(workflow.svg.clone().attr("width", "100%").attr("height", "100%"));
  }
  $("#indicator").hide();
}

function showPlots(item_id) {
  var workflow = listed_workflows[item_id];
  var win = window.open(workflow.plots, '_blank');
  win.focus();
}

function showLogs(item_id) {
  var workflow = listed_workflows[item_id];
  var win = window.open("logs.html?session=" + workflow.log_id, '_blank');
  win.focus();
}

$(window).load(function() {
  setInterval(updateUI, 2000);
  setTimeout(updateUI, 0);
});
