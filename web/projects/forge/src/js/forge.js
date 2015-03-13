var query = "";
var found = null;

global.show_image = function(name) {
  let win = window.open("image.html?name=" + name, name,
    "width=600px,height=600px,menubar=no,toolbar=no,location=no,status=no");
  win.document.title = name;
};

global.inputKeyPress = function(e) {
  if (e.keyCode === 13){
    search();
    return false;
  }
};

export function search() {
  let search = $("#search");
  const needle = search.val();
  if (!needle) {
    return;
  }
  let skipping = (query == needle);
  query = needle;
  const regexp = new RegExp(needle, "i");
  $(".flex-row").each(function() {
    if (skipping) {
      if (this === found) {
        skipping = false;
      }
      return;
    }
    found = null;
    const row = $(this);
    ["h2", "h4", ".details-value"].forEach(function(sel) {
      if (row.find(sel).text().search(regexp) >= 0) {
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
    search.css("background-color", "Tomato");
    location.href = "#";
  } else {
    search.css("background-color", "GreenYellow ");
  }
  search.focus();
}