global.smallest_time = Infinity;  // yes, no "-"
global.biggest_time = -Infinity;  // yes, there is "-"
global.elapsed_time = 0;
global.time_scale = 0;
var max_count = 600;
global.event_names = ["hidden"];
var event_names_mapping = {"hidden": 0};
global.nodes = [];
global.nodes_mapping = {};
global.selected_slave = null;
global.current_min_time = null;
global.current_max_time = null;
global.series_data = [[], []];
global.fetching_events = {master: false, slave: false};
global.fetching_logs = {master: false, slave: false};
var need_fetch = false;
var too_many = "too many events in the current interval";
global.ui_is_set_up = false;


function mongoRequest(target, type, query, then) {
  var dataRequest = new XMLHttpRequest();
  dataRequest.open("POST", "service", true);
  dataRequest.onload = function() {
    if (dataRequest.readyState == 4 && dataRequest.status == 200) {
      let result = JSON.parse(dataRequest.responseText).result;
      then(result);
    }
  };
  var message = {request: target};
  message[type] = query;
  dataRequest.send(JSON.stringify(message));
}

///////////////////////////////////////////////////////////////////////////////
// Events
///////////////////////////////////////////////////////////////////////////////

function installSlaveEvents(node) {
  for (var index = 1; index < event_names.length; index++) {
    series_data[1][index].length = 0;
    var events = node.events[event_names[index]].data;
    for (let event of events) {
      series_data[1][index].push(event);
    }
  }
}

function finalizeFetch(node, no_install, callback) {
  var ntype = (node.id == "master")? "master" : "slave";
  if (ntype == "slave" && !no_install) {
    installSlaveEvents(node);
  }
  if (document.readyState === "complete") {
    if (node.id == "master") {
      $("#chart-loading-master").hide();
    } else {
      $("#chart-loading-slave").hide();
    }
  }
  fetching_events[ntype] = false;
  if (!fetching_events.master && !fetching_events.slave) {
    if (need_fetch) {
      fetchEvents();
    } else if (document.readyState === "complete") {
      renderEvents();
    }
  }
  if (callback != undefined) {
    callback(node);
  }
}

function fetchEventsForNode(node, force_install) {
  if (!node) {
    return;
  }
  if (!node.initialized) {
    initialFetchEventsForNode(node, fetchEventsForNode);
    return;
  }
  if (force_install == undefined) {
    force_install = false;
  }
  fetching_events[node.id == "master"? "master" : "slave"] = true;
  var my_min_time = current_min_time;
  var my_max_time = current_max_time;
  var fetched_events = [];
  for (let name in node.events) {
    var event = node.events[name];
    if (event.fetch && (event.min_fetch_time > my_min_time ||
                        event.max_fetch_time < my_max_time)) {
      fetched_events.push(name);
    }
  }
  if (fetched_events.length == 0) {
    finalizeFetch(node, !force_install);
    return;
  }
  if (node.id == "master") {
    $("#chart-loading-master").show();
  } else {
    $("#chart-loading-slave").show();
  }
  mongoRequest("events", "aggregate", [
      { $match: { session: session, instance: node.id,
                  name: { $in: fetched_events },
                  time: { $gt: my_min_time, $lt: my_max_time } } },
      { $group: { _id: { name: "$name" },
                  count: { $sum: 1 },
                  min: { $min: "$time" },
                  max: { $max: "$time" }
                }
      }], function(result) {

    fetched_events = [];
    for (let event of result) {
      var data = node.events[event._id.name].data;
      data.length = 0;
      if (event.count <= max_count) {
        fetched_events.push(event._id.name);
      } else {
        data.push({x: event.min - smallest_time, y: 0, event: too_many});
        data.push({x: event.min - smallest_time, y: 1, event: too_many});
        data.push({x: event.max - smallest_time, y: 1, event: too_many});
        data.push({x: event.max - smallest_time, y: 0, event: too_many});
      }
    }

    if (fetched_events.length > 0) {
      mongoRequest("events", "find", {session: session,
                                      instance: node.id,
                                      name: { $in: fetched_events },
                                      time: { $gt: my_min_time,
                                              $lt: my_max_time }},
        function(data) {

        setEvents(node, data);
        for (let fe of fetched_events) {
          var event = node.events[fe];
          event.min_fetch_time = my_min_time;
          event.max_fetch_time = my_max_time;
        }
        finalizeFetch(node, false);
      });
    } else {
      finalizeFetch(node, !force_install);
    }
  });
}

global.fetchEvents = function() {
  if (fetching_events.master || fetching_events.slave) {
    need_fetch = true;
    return;
  }
  need_fetch = false;
  fetchEventsForNode(nodes[nodes_mapping["master"]]);
  fetchEventsForNode(selected_slave);
}

function setEvents(node, result) {
  var meta_ignored_keys = {"_id": null, "time": null, "name": null,
                           "session": null, "instance": null, "height": null};
  for (var index in result) {
    var event = result[index];
    var meta = {};
    for (var key in event) {
      if (meta_ignored_keys[key] === undefined) {
        meta[key] = event[key];
      }
    }
    meta = JSON.stringify(meta);
    var time = event.time - smallest_time;
    var height = (event.height != undefined)? event.height : 1;
    var data = node.events[event.name].data;

    if (event.type === "single") {
      data.push({x: time, y: 0, event: meta});
      data.push({x: time, y: height, event: meta});
      data.push({x: time, y: 0, event: meta});
      continue;
    }

    if (event.type === "begin") {
      data.push({x: time, y: 0});
    }
    data.push({x: time, y: height, event: meta});
    if (event.type === "end") {
      data.push({x: time, y: 0});
    }
  }
}

function initialFetchEventsForNode(node, callback) {
  fetching_events[node.id == "master"? "master" : "slave"] = true;
  var query = {session: session, instance: node.id, name: { $in: [] }};
  for (var name in node.events) {
    var event = node.events[name];
    if (event.count <= max_count) {
      event.fetch = false;
      query.name.$in.push(event.name);
    } else {
      event.data.push({x: event.min - smallest_time, y: 0, event: too_many});
      event.data.push({x: event.min - smallest_time, y: 1, event: too_many});
      event.data.push({x: event.max - smallest_time, y: 1, event: too_many});
      event.data.push({x: event.max - smallest_time, y: 0, event: too_many});
      event.min_fetch_time = smallest_time;
      event.max_fetch_time = smallest_time;
    }
  }
  if (query.name.$in.length > 0) {
    mongoRequest("events", "find", query, function(result) {
      setEvents(node, result);
      finalizeFetch(node, false, callback);
      node.initialized = true;
    });
  } else {
    node.initialized = true;
    if (callback != undefined) {
      callback(node);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Logs
///////////////////////////////////////////////////////////////////////////////

function fetchLogsForNode(instance, level) {
  var node = (instance == "master")? "master" : "slave";
  fetching_logs[node] = true;
  if (document.readyState === "complete") {
    $("#logs-contents-" + node).hide();
    $("#logs-loading-" + node).show();
  }
  var levels = ["CRITICAL", "ERROR"];
  switch (level) {
    case "DEBUG":
      levels.push("DEBUG");
    case "INFO":
      levels.push("INFO");
    case "WARNING":
      levels.push("WARNING");
    default:
      break;
  }
  mongoRequest("logs", "find", {session: session,
                                node: instance,
                                levelname: { $in: levels }},
    function(data) {
      fetching_logs[node] = false;
      renderLogs(data, instance);
  });
}

function fetchLogs(level) {
  fetchLogsForNode("master", level);
  fetchLogsForNode(selected_slave.id, level);
}

///////////////////////////////////////////////////////////////////////////////
// Startup sequence
///////////////////////////////////////////////////////////////////////////////

mongoRequest("events", "aggregate", [
    { $match: { session: session } },

    { $group: { _id: { instance: "$instance",
                       name: "$name" },
                count: { $sum: 1 },
                min: { $min: "$time" },
                max: { $max: "$time" }
              }
    },
    { $group: { _id: "$_id.instance",
                details: { $addToSet: { name: "$_id.name",
                                        count: "$count",
                                        min: "$min",
                                        max: "$max"
                                      }
                         }
              }
    }],
    function(result) {

  var event_names_unique = {};
  result.forEach(function(estats) {
    estats.details.forEach(function(item) {
      event_names_unique[item.name] = null;
    });
    var node = nodes[nodes.push({ id: estats._id,
                                  initialized: false,
                                  events: {}
                                }) - 1];
    nodes_mapping[estats._id] = nodes.length - 1;
    estats.details.forEach(function(item) {
      item.fetch = true;
      item.data = [];
      node.events[item.name] = item;
      smallest_time = Math.min(smallest_time, item.min);
      biggest_time = Math.max(biggest_time, item.max);
    });
  });

  for (let event_name in event_names_unique) {
    event_names_mapping[event_name] = event_names.push(event_name) - 1;
  }

  elapsed_time = biggest_time - smallest_time;
  time_scale = elapsed_time;
  current_min_time = smallest_time;
  current_max_time = biggest_time;

  // Fill series_data
  series_data[0].push([{x: 0, y: 0}, {x: elapsed_time, y: 0}]);
  series_data[1].push([{x: 0, y: 0}, {x: elapsed_time, y: 0}]);
  var master = nodes[nodes_mapping["master"]];
  for (var index = 1; index < event_names.length; index++) {
    var event = event_names[index];
    // Ensure every node has all event types, some of which are empty
    for (var node_index in nodes) {
      let node = nodes[node_index];
      if (node.events[event] === undefined) {
        node.events[event] = {fetch: false, data: [], count: 0,
                              min: smallest_time, max: biggest_time};
      }
    }

    series_data[0].push(master.events[event].data);
    series_data[1].push([]);
  }

  // Initial events fetch
  initialFetchEventsForNode(master);
  fetchLogsForNode("master", "INFO");
  if (nodes.length > 1) {
    for (let node of nodes) {
      if (node.id != "master") {
        selected_slave = node;
        break;
      }
    }
    initialFetchEventsForNode(selected_slave);
    fetchLogsForNode(selected_slave.id, "INFO");
  }
});
