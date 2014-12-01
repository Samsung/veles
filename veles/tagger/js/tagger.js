if (!String.prototype.$) {
  String.prototype.$ = function() {
    var args = arguments;
    if (args.length > 1) {
      return this.replace(/{(\d+)}/g, function(match, number) {
        return typeof args[number] != 'undefined'
        ? args[number]
        : match;
      });
    }
    return this.replace("{}", args[0]);
  };
}

$(function() {
  "use strict";
  $("button#prev").button({
    icons: {
      primary: "fa fa-arrow-left"
    },
    text: false
  });
  $("button#next").button({
    icons: {
      primary: "fa fa-arrow-right"
    },
    text: false
  });
  $("input#menu").button({
    icons: {
      primary: "fa fa-bars"
    },
    text: false
  });

  $("input[type=radio]").change(function(event) {
    uploadResult();
    active_index = $(this).index() / 2;
    loadCanvasImageAsync();
  });
  canvas.addEventListener('mousedown', mouseDown, false);
  canvas.addEventListener('mouseup', mouseUp, false);
  canvas.addEventListener('mousemove', mouseMove, false);

  $(document).keypress(function() {
    if (event.keyCode == 32 && selections.length > 0) {
      selections.pop().remove();
      event.preventDefault();
    }
    if (event.keyCode == 127) {
      removeAllSelections();
      event.preventDefault();
    }
    if (event.keyCode == 13) {
      next();
    }
  });

  loadCanvasImage(0);
  $(window).resize(function() {
    calculateRatio();
  });
  updateFileIndicators();
  var touched_request = JSON.stringify(files.map(function(file) { return file.path; }));
  setInterval(function() {
    $.ajax({
      url: "touched",
      type: "POST",
      data: touched_request,
      contentType: "application/json; charset=utf-8",
      success: function(response) {
        var touched = JSON.parse(response);
        files.forEach(function(file) {
          file.touched = touched[file.path];
        });
        updateFileIndicators();
      }
    });
  }, 1000);

  $("#nav > button, #nav > input").button("enable");
});

var canvas = $("canvas#main").get(0);
var canvas_ctx = canvas.getContext('2d');
var dragging = false;
var selections_hidden = false;
var drect = {};
var ratio = 1.0;
var selections = [];
var active_index = 0;
var toolbar_collapsed_height = $("#toolbar").css("height");
var selection_colors = [
    [255, 0,   0  ], [166, 204, 20 ], [255, 252, 25 ],
    [236, 0,   255], [20,  204, 146], [255, 199, 25 ],
    [25,  0,   255], [40,  204, 20 ], [255, 116, 25 ],
    [255, 0,   133], [255, 255, 255], [96,  96,  96]];

function calculateRatio() {
  var size = files[active_index].size;
  ratio = size[0] / canvas.clientWidth;
}

function updateFileIndicators() {
  var list = $("#list > label");
  for (var index in files) {
    var file = files[index];
    var classes = list[index].className;
    var touched = classes.indexOf(" touched") >= 0;
    if (file.touched && !touched) {
      list[index].className = classes + " touched";
    } else if (!file.touched && touched) {
      list[index].className = classes.replace(" touched", "");
    }
  }
}

function removeAllSelections() {
  selections.forEach(function(sel) {
    sel.remove();
  });
  selections.length = 0;
}

function loadCanvasImageAsync(index) {
  if (index == undefined) {
    index = active_index;
  }
  if (index != active_index) {
    $("input[type=radio]:nth-of-type({})".$(index + 1))
    .prop("checked", true);
    active_index = index;
  }
  var start_time = new Date();
  var preload = new Image();
  preload.onload = function() {
    delete this;
    var elapsed = new Date() - start_time;
    if (elapsed < 300) {
      setTimeout(function() { loadCanvasImage(index); }, 300 - elapsed);
    } else {
      loadCanvasImage(index);
    }
  };
  preload.src = files[index].url;
}

function loadCanvasImage(index) {
  if (index == undefined) {
    index = active_index;
  }
  removeAllSelections();
  var size = files[index].size;
  $(canvas).css("background", "url({})".$(files[index].url))
           .attr("width", size[0]).attr("height", size[1]);
  calculateRatio();
  if (index != active_index) {
    $("input[type=radio]:nth-of-type({})".$(index + 1))
        .prop("checked", true);
    active_index = index;
  }
  $.ajax({
      url: "selections",
      type: "POST",
      data: JSON.stringify({file: files[index].path}),
      contentType: "application/json; charset=utf-8",
      success: function(response) {
        JSON.parse(response).forEach(function(sel) {
          addSelection(sel);
        });
      },
  });
}

function next() {
  uploadResult();
  if (active_index < files.length) {
    loadCanvasImageAsync(active_index + 1);
  }
}

function previous() {
  uploadResult();
  if (active_index > 0) {
    loadCanvasImageAsync(active_index - 1);
  }
}

function hideSelections() {
  selections_hidden = true;
  selections.forEach(function(sel) {
    sel.hide();
  });
}

function showSelections() {
  selections.forEach(function(sel) {
    sel.show();
  });
  selections_hidden = false;
}

function addSelection(rect) {
  selections.push($("<div class=\"selection\"></div>")
      .appendTo($("#image-container"))
      .css("left", rect.left * 100 / (ratio * canvas.offsetWidth) + "%")
      .css("top", rect.top * 100 / (ratio * canvas.offsetHeight) + "%")
      .css("width", rect.width * 100 / (ratio * canvas.offsetWidth) + "%")
      .css("height", rect.height * 100 / (ratio * canvas.offsetHeight) + "%")
      .css("border-color",
           "rgb({})".$(selection_colors[selections.length].join(", ")))
      .css("background-color",
           "rgba({}, 0.15)".$(selection_colors[selections.length].join(", ")))
      .resizable().draggable().bind("contextmenu", function(e) {
        selections.splice(selections.indexOf($(this)), 1);
        $(this).remove();
        return false;
      })
      .data("rect", rect)
  );
}

function mouseDown(e) {
  if (selections.length >= selection_colors.length || dragging) {
    return;
  }
  var offset = $(this).offset();
  drect.left = e.pageX - offset.left;
  drect.top = e.pageY - offset.top;
  dragging = true;
  if (e.shiftKey) {
    hideSelections();
  }
}

function mouseMove(e) {
  if (!dragging) {
    return;
  }
  if (selections_hidden && !e.shiftKey) {
    showSelections();
  }
  if (e.shiftKey) {
    hideSelections();
  }
  var offset = $(this).offset();
  drect.width = (e.pageX - offset.left) - drect.left;
  drect.height = (e.pageY - offset.top) - drect.top;
  canvas_ctx.clearRect(0, 0, canvas.width, canvas.height);
  canvas_ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
  canvas_ctx.fillRect(drect.left * ratio, drect.top * ratio,
      drect.width * ratio, drect.height * ratio);
  canvas_ctx.fillStyle = "black";
  canvas_ctx.strokeRect(drect.left * ratio, drect.top * ratio,
      drect.width * ratio, drect.height * ratio);
}

function mouseUp() {
  if (selections_hidden) {
    showSelections();
  }
  dragging = false;
  canvas_ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (Math.abs(drect.width|0) < 16 || Math.abs(drect.height|0) < 16) {
    return;
  }
  if (drect.width < 0) {
    drect.left += drect.width;
    drect.width *= -1;
  }
  if (drect.height < 0) {
    drect.top += drect.height;
    drect.height *= -1;
  }
  addSelection({left: drect.left * ratio, top: drect.top * ratio,
                width: drect.width * ratio, height: drect.height * ratio});
  drect.width = drect.height = 0;
}

function uploadResult(index, overwrite, result) {
  if (result != undefined || selections.length > 0) {
    if (index == undefined) {
      index = active_index;
    }
    var file = files[index];
    if (overwrite == undefined) {
      overwrite = false;
    }
    if (result == undefined) {
      result = selections.map(function(sel) {
        return sel.data("rect");
      });
    }
    $.ajax({
      url: "update",
      type: "POST",
      data: JSON.stringify({file: file.path,
                            selections: result,
                            overwrite: overwrite}),
      contentType: "application/json; charset=utf-8",
      statusCode: {
        403: function() {
          if (!overwrite) {
            $("#dialog-overwrite").dialog({
              resizable: false,
              height: 300,
              width: 400,
              modal: true,
              buttons: {
                "Overwrite": function() {
                  uploadResult(index, true, result);
                  $(this).dialog("close");
                },
                "Skip": function() {
                  $(this).dialog("close");
                }
              }
            });
          }
        }
      }
    });
  }
}

function menu() {
  var toolbar = $("#toolbar");
  var list = $("#list-container");
  if (toolbar.css("height") == toolbar_collapsed_height) {
    toolbar.css("height", "10em");
    list.css("margin-top", "10em");
  } else {
    toolbar.css("height", toolbar_collapsed_height);
    list.css("margin-top", toolbar_collapsed_height);
  }
}
