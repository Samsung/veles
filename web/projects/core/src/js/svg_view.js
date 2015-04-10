$(document).ready(function () {
  $("body").append(window.svg);
  var svg = d3.select("svg > g");
  var initt = d3.transform(svg.attr("transform")).translate;

  function zoom() {
    svg.attr("transform", "translate(" + d3.event.translate +
    ")scale(" + d3.event.scale + ")");
  }

  svg.call(d3.behavior.zoom().translate(initt).scaleExtent([1, 8]).on("zoom",
    zoom));

  svg.insert("rect", "title")
    .attr("class", "overlay")
    .attr("width", "100%")
    .attr("height", "100%")
    .attr("x", -initt[0]).attr("y", -initt[1]);
});

