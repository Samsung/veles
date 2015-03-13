$("#content").hide();

function svgContainerResize() {
  var div = $("#workflow-svg-container-fill");
  var width = div.width();
  var height = div.height();
  var size = Math.min(width, height);
  console.log("resize " + size);
  $("#workflow-svg-container").css("width", size + "px").css("height", size + "px");
}

$("#workflow-svg-container-fill").resize(function () {
  svgContainerResize();
});
$(window).resize(function () {
  svgContainerResize();
});
