var smallest_time = Infinity;
var biggest_time = -Infinity;
var elapsed_time = 0;
var time_scale = 0;
var max_count = 600;
var event_names = ["hidden"];
var event_names_mapping = {"hidden": 0};
var nodes = [];
var nodes_mapping = {};
var selected_slave = null;
var current_min_time = null;
var current_max_time = null;
var series_data = [[], []];
var fetching_events = [false, false];
var need_fetch = false;
var too_many = "too many events in the current interval";
var ui_is_setup = false;


function mongo_request(target, type, query, then) {
  var dataRequest = new XMLHttpRequest();
  dataRequest.open("POST", "service", true);
  dataRequest.onload = function() {
    if (dataRequest.readyState == 4 && dataRequest.status == 200) {
      result = JSON.parse(dataRequest.responseText).result;
      then(result);
    }
  };
  var message = {request: target}
  message[type] = query;
  dataRequest.send(JSON.stringify(message));
}

function install_slave_events(node) {
  for (var index = 1; index < event_names.length; index++) {
    series_data[1][index].length = 0;
    var events = node.events[event_names[index]].data;
    for (var di in events) {
      series_data[1][index].push(events[di]);
    }
  }
}

function finalize_fetch(node) {
  var ni = (node.id == "master")? 0 : 1;
  if (ni == 1) {
    install_slave_events(node);
  }
  fetching_events[ni] = false;
  if (!fetching_events[0] && !fetching_events[1]) {
    if (need_fetch) {
      fetch_events();
    } else if (document.readyState === "complete") {
      render_events();
    }
  }
}

function fetch_events_for_node(node_name) {
  var node = nodes[nodes_mapping[node_name]];
  var fetched_events = [];
  for (var name in node.events) {
    if (node.events[name].fetch) {
      fetched_events.push(name);
    }
  }
  mongo_request("events", "aggregate", [
      { $match: { session: session, instance: node.id,
                  name: { $in: fetched_events },
                  time: { $gt: current_min_time, $lt: current_max_time } } },
      { $group: { _id: { name: "$name" },
                  count: { $sum: 1 },
                  min: { $min: "$time" },
                  max: { $max: "$time" }
                }
      }], function(result) {

    fetched_events = [];
    for (var index in result) {
      var event = result[index];
      if (event.count <= max_count) {
        fetched_events.push(event._id.name);
      } else {
        var data = node.events[event._id.name].data;
        data.length = 0;
        data.push({x: event.min - smallest_time, y: 0, event: too_many});
        data.push({x: event.min - smallest_time, y: 1, event: too_many});
        data.push({x: event.max - smallest_time, y: 1, event: too_many});
        data.push({x: event.max - smallest_time, y: 0, event: too_many});
      }
    }

    if (fetched_events.length > 0) {
      mongo_request("events", "find", {session: session,
                                       instance: node.id,
                                       name: { $in: fetched_events },
                                       time: { $gt: current_min_time,
                                               $lt: current_max_time }},
        function(data) {
        set_events(node, data);
        finalize_fetch(node);
      });
    } else {
      finalize_fetch(node);
    }
  });
}

function fetch_events() {
  if (fetching_events[0] || fetching_events[1]) {
    need_fetch = true;
    console.log(fetching_events);
    return;
  }
  fetching_events[0] = fetching_events[1] = true;
  need_fetch = false;
  fetch_events_for_node("master");
  fetch_events_for_node(selected_slave.id);
}

function set_events(node, result) {
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

function initial_fetch_events_for_node(node) {
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
    }
  }
  if (query.name.$in.length > 0) {
    mongo_request("events", "find", query, function(result) {
      set_events(node, result);
      finalize_fetch(node);
    });
  }
}


mongo_request("events", "aggregate", [
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
    var node = nodes[nodes.push({id: estats._id}) - 1];
    nodes_mapping[estats._id] = nodes.length - 1;
    node.events = {}
    estats.details.forEach(function(item) {
      item.fetch = true;
      item.data = [];
      node.events[item.name] = item;
      smallest_time = Math.min(smallest_time, item.min);
      biggest_time = Math.max(biggest_time, item.max);
    });
  });

  for (var event_name in event_names_unique) {
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
      var node = nodes[node_index];
      if (node.events[event] === undefined) {
        node.events[event] = {fetch: false, data: [], count: 0,
                              min: smallest_time, max: biggest_time};
      }
    }

    series_data[0].push(master.events[event].data);
    series_data[1].push([]);
  }

  // Initial events fetch
  fetching_events[0] = true;
  initial_fetch_events_for_node(master);
  if (nodes.length > 1) {
    fetching_events[1] = true;
    for (var index in nodes) {
      if (nodes[index].id != "master") {
        selected_slave = nodes[index];
        break;
      }
    }
    initial_fetch_events_for_node(selected_slave);
    install_slave_events(selected_slave);
  }
});
