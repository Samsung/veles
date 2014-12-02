var alias_reversed = {};

for (var key in opts.alias) {
  alias_reversed[opts.alias[key]] = key;
}

initial_disable();

var substringMatcher = function(strs) {
  return function findMatches(q, cb) {
    var matches, substrRegex;
    matches = [];
    substrRegex = new RegExp(q, 'i');
    $.each(strs, function(i, str) {
      if (substrRegex.test(str)) {
        matches.push({ value: str });
      }
    });
    cb(matches);
  };
};

$('.typeahead').typeahead({
  hint: true,
  highlight: true,
  minLength: 1
},
{
  name: 'workflows',
  displayKey: 'value',
  source: substringMatcher(workflows)
});

function disable(mode) {
  $(".argument:not(." + mode + ")").addClass("disabled");
  $(".panel-heading:not(." + mode + ")").addClass("disabled");
  $(".form-control:not(." + mode + ")").attr("disabled", "disabled");
  $(".switch:not(." + mode + ")").bootstrapSwitch('disabled', true);
  $(".dropdown-toggle:not(." + mode + ")").attr("disabled", "disabled");
};

function enable_all() {
  $(".argument").removeClass("disabled");
  $(".panel-heading").removeClass("disabled");
  $(".form-control").removeAttr("disabled", "disabled");
  $(".switch").bootstrapSwitch('disabled', false);
  $(".dropdown-toggle").removeAttr("disabled", "disabled");
};

function initial_disable() {
  common_mode = $("#dropdownMode").text().trim();
  common_mode = common_mode.toLowerCase();
  disable(common_mode);
}

function activate_mode(mode) {
  $("#dropdownMode").get(0).firstChild.nodeValue = mode + " ";
  mode = mode.toLowerCase();
  enable_all();
  disable(mode);
}

function select_mode(mode) {
  activate_mode(mode);
  var cl = $("#command-line").val().replace(/\s+/, " ");
  var args = parse_cl(cl);
  cmdline_states = from_args_to_states(args, cmdline_states);
  $.each(states, function(opt, arg) {
    var elem_opt = $("#" + opt);
    if (!elem_opt.hasClass(mode)) {
      delete cmdline_states[opt];
      cl = cl.replace(opt + " " + arg, "");
      cl = cl.replace(opt, "");
    }
  });
  elem.val(cl);
}

function select(choice, id) {
  $("#dropdown_menu" + id).get(0).firstChild.nodeValue = choice + " ";
  choice = choice.toLowerCase();
  var id = id;
  var value = choice;
  var prev_value = set_arg_state(id, value);
  if (prev_value == value) {
    return;
  }
  var elem = $("#command-line");
  var cl = elem.val();
  var reg = /\s+/;
  cl = cl.replace(reg, " ");
  if (cl.indexOf(id + " " + prev_value) != -1) {
     cl = cl.replace(id + " " + prev_value, id + ' ' + value);
  } else {
     cl = id + ' ' + value + ' ' + cl;
  }
  elem.val(cl);
}

function get_diff(obj1, obj2) {
  var result = {};
  var keys_result = Object.keys(obj1).concat(Object.keys(obj2));
  for (var ind in keys_result) {
    var key = keys_result[ind];
    var ok1 = obj1[key];
    var ok2 = obj2[key];
    if (ok1 == undefined && ok2 != undefined) {
      result[key] = "created";
    } else if (ok1 != undefined && ok2 == undefined) {
      result[key] = "deleted";
    } else if (ok1 != undefined && ok2 != undefined && ok1 == ok2) {
      result[key] = "unchanged";
    } else {
      result[key] = "updated";
    }
  }
  return result;
}

function parse_cl(cl) {
  if (cl.match(/(^|\s+)-\w\w+/)) {
    return null;
  }
  var args = minimist_parse(cl.split(/\s+/), opts);
  var positional = args["_"];
  var config_list = [];
  if (positional.length <= positional_opts.length) {
    for (var ind in positional) {
      args[positional_opts[ind]] = positional[ind];
    }
  }
  else {
    for (var ind in positional) {
      if (positional_opts[ind] != undefined && (ind != positional_opts.length - 1)) {
        args[positional_opts[ind]] = positional[ind];
      } else {
        config_list.push(positional[ind]);
        args[positional_opts[positional_opts.length - 1]] = config_list;
      }
    }
  }
  delete args["_"];
  return args;
}

function from_args_to_states(args, states) {
  $.each(args, function(opt, arg) {
    if ($.inArray(opt, positional_opts) != -1) {
      states[opt] = arg;
    } else if (opts.alias[opt] == undefined) {
      states["--" + opt] = arg;
    }
  });
  return states;
}

function build_switch_regexp(id) {
  return new RegExp("(\\s|^)" + id + "(\\s|$)");
}

function build_arg_regexp(id) {
  return new RegExp("(\\s|^)" + id + "([ =]|$)[^\\s-]*");
}

function set_arg_state(id, value) {
  var previous_value = cmdline_states[id];
  cmdline_states[id] = value;
  return previous_value;
}

function change_cl_by_input(cl, prev_value, value, id) {
  function generate_tail(id, value, wf, config, config_list) {
    var kwargs = {"workflow": 0, "config": 1, "config_list": 2};
    var tail = [].slice.call(arguments, 2);
    return tail.join(" ");
  }

  function cl_tail(wf, config, config_list) {
    return [].filter.call(arguments, function(s) { return s }).join(" ");
  }

  var wf = cmdline_states["workflow"];
  var config = cmdline_states["config"];
  var config_list = cmdline_states["config_list"];
  if ($.inArray(id, positional_opts) == -1) {
    var aid = "-" + alias_reversed[id.substring(2)];
    switch (aid) {
    case "m":
      activate_mode('Slave');
      break
    case "l":
      activate_mode('Master');
      break
    default:
      activate_mode('Standalone');
      break
    }
    if (cl.match(build_arg_regexp(id)) || (aid && cl.match(build_arg_regexp(aid)))) {
      cl = cl.replace(build_arg_regexp(id), id + ' ' + value);
      if (aid) {
        cl = cl.replace(build_arg_regexp(aid), aid + ' ' + value);
      }
    } else {
      cl = id + ' ' + value + ' ' + cl;
    }
  } else {
    var tail = cl_tail(wf, config, config_list);
    var new_tail = generate_tail(id, value, wf, config, config_list);
    if (cl.indexOf(prev_value) != -1) {
      cl = cl.replace(prev_value, value);
    } else {
      if (cl.indexOf(tail) != -1) {
        cl = cl.replace(tail, new_tail);
      } else {
          cl += " " + value;
      }
    }
  }
  return cl;
}

function set_error_test(text) {
  $("#error-message").text(text);
}

// This function modified command line for each change of value of widgets: input, switch, typeahead.
// Also it modified value of widgets for each change of command line.
$(function() {
    // This event changes value of option and value of command line if value of input has been changed.
    $("input:not(#command-line)").on("change", function() {
      var id = this.id;
      var value = this.value;
      var prev_value = set_arg_state(id, value);
      if (prev_value == value) {
        return;
      }
      var elem = $("#command-line");
      var cl = elem.val();
      var reg = /\s+/;
      cl = cl.replace(reg, " ");
      cl = change_cl_by_input(cl, prev_value, value, id);
      elem.val(cl);
    });
    // This event changes value of option and value of command line if switch has been switched.
    $(".switch").bootstrapSwitch().on('switchChange.bootstrapSwitch', function(event, state) {
      var id = this.id;
      var value = state;
      var prev_value = set_arg_state(id, value);
      if (prev_value == value) {
        return;
      }
      var elem = $("#command-line");
      var cl = elem.val();

      var aid = "-" + alias_reversed[id.substring(2)];
      if (value) {
        if (!aid || !cl.match(build_switch_regexp(aid))) {
          cl = id + ' ' + cl;
        }
      } else {
        cl = cl.replace(build_switch_regexp(id), '');
        if (aid) {
          cl = cl.replace(build_switch_regexp(aid), '');
        }
      }
      elem.val(cl);
    });
    // This event changes value of option and value of command line if workflow was chosen.
    $('.typeahead').on('typeahead:selected', function(event, value) {
      var selected_wf = value;
      var prev_wf = cmdline_states["workflow"];
      var wf = selected_wf["value"];
      var config = cmdline_states["config"];
      var spase = " ";
      var config_list = cmdline_states["config_list"];
      if (config == undefined) {
        config = "";
        space = "";
      }
      if (config_list == undefined) {
        config_list = "";
        space = "";
      }
      cmdline_states["workflow"] = wf;
      var elem = $("#command-line");
      var cl = elem.val();
      var reg = /\s+/;
      cl = cl.replace(reg, " ");
      if (cl.indexOf(prev_wf) != -1) {
        cl = cl.replace(prev_wf, wf);
      } else {
        var conf_str = config + space + config_list;
        if (cl.indexOf(config + space + config_list) != -1) {
          cl = cl.replace(config + space + config_list, wf + " " + conf_str);
        } else {
          cl += wf;
        }
      }
      elem.val(cl);
    });
    // This event changes value of widgets for each change of command line.
    $("#command-line").on("input", function() {
      var command_line = $("#command-line").val();
      var args = parse_cl(command_line);
      // There is a potential parsing error
      if (args === null) {
        return;
      }
      var prev_cmdline_states = {};
      $.each(cmdline_states, function(opt, arg) {
        prev_cmdline_states[opt] = arg;
        opt_for_arg = opt.replace(/^-+/, "");
        if (args[opt_for_arg] == undefined) {
          cmdline_states[opt] = defaults[opt];
        }
      });
      cmdline_states = from_args_to_states(args, cmdline_states);
      var diff = get_diff(prev_cmdline_states, cmdline_states);
      invalid = false;
      mode = 'Standalone';
      for (var option in diff) {
        var status_arg = diff[option];
        var arg = cmdline_states[option];
        var elem_opt = $("#" + option);
        if (status_arg == "unchanged" || status_arg == "deleted") {
          continue;
        }
        if (status_arg == "created" || status_arg == "updated") {
          switch (option) {
          case "--master-address":
            mode = 'Slave';
            break
          case "--listen-address":
            mode = 'Master';
            break
          default:
            break
          }
          if (elem_opt.hasClass("switch")) {
            elem_opt.bootstrapSwitch('state', arg);
          }
          if (elem_opt.hasClass("typeahead") || (elem_opt.hasClass("form-control") && !(elem_opt.hasClass("typeahead")))) {
            elem_opt.val(arg);
          }
          if (elem_opt.hasClass("dropdown-menu")) {
            var elem = $("#dropdown_menu" + option)
              if ($.inArray(arg, choices[option]) != -1) {
                elem.get(0).firstChild.nodeValue = arg;
              } else {
                set_error_test("Wrong value of " + option + ". You should choose from " + choices[option]);
                invalid = true;
              }
          }
          if (!(elem_opt.hasClass("switch")) && !(elem_opt.hasClass("form-control")) &&
              !(elem_opt.hasClass("dropdown-menu")) && $.inArray(option, special_opts) == -1) {
            set_error_test("Wrong option " + option);
            invalid = true;
            delete cmdline_states[option];
          }
          for (var key in opts.alias) {
            if (command_line.match(build_switch_regexp("-" + key)) &&
                command_line.match(build_switch_regexp("--" + opts.alias[key]))) {
              set_error_test("Duplicate option: --" + opts.alias[key]);
              invalid = true;
            }
          }
        }
      }
      activate_mode(mode);
      $("#error-message").css("opacity", invalid? "1" : "0");
    });
});

function run() {
  $.ajax({
    url: "cmdline",
    type: "POST",
    data: JSON.stringify({cmdline: $("#command-line").val()}),
    contentType: "application/json; charset=utf-8",
    async: true,
    success: function(result) {
      window.close();
    }
  });
}

$(function() {
  $("#command-line").trigger("input");
});
