function show_image(name) {
  var win = window.open("image.html?name=" + name, name, "width=600px,height=600px,menubar=no,toolbar=no,location=no,status=no");
  win.document.title = name;
}


var query = "";
var found = null;


function inputKeyPress(e) {
  if (e.keyCode === 13){
    search();
    return false;
  }
}

function search() {
  var needle = $("#search").val();
  if (!needle) {
    return;
  }
  var skipping = (query == needle);
  query = needle;
  $(".flex-row").each(function() {
    if (skipping) {
      if (this === found) {
        skipping = false;
      }
      return;
    }
    found = null;
    var row = $(this);
    ["h2", "h4", ".details-value"].forEach(function(sel) {
      if (row.find(sel).text().search(needle) >= 0) {
        found = row.get(0);
        location.href = "#" + found.id;
        return false;
      }
    });
    if (found != null) {
      return false;
    }
  });
  if (found == null) {
    query = "";
    $("#search").css("background-color", "Tomato");
    location.href = "#";
  } else {
    $("#search").css("background-color", "GreenYellow ");
  }
  $("#search").focus();
}