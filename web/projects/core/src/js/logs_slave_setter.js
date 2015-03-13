var slave_setter = null;
$(function () {
  $("#slave-setter-horizontal-centering").hide();
  $("#timescale").hide();
  $("#preview").hide();
  $("select").selectmenu().each(function () {
    var menu = $(this);
    menu.selectmenu("menuWidget").addClass("overflow").addClass("cool-select");
    menu.selectmenu("widget").removeAttr("style").addClass("cool-select");
  });
  $(".slave-setter").selectmenu("widget").addClass("slave-setter");
  $("#timescale").slider({
    range: false, min: 0, max: 100, value: 100
  });
});
