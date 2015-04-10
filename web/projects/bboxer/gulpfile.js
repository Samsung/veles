plugins = require('gulp-load-plugins')();
require('../common/gulpcommon.js');


gulp.task('awesome_fonts', function() {
  return gulp.src('src/libs/font-awesome/fonts/*')
    .pipe(gulp.dest(dist + "fonts"));
});

gulp.task('jquery.ui', function () {
  var $ = function (file) {
    return 'src/libs/jquery.ui/ui/' + file + '.js';
  };
  return gulp.src([
    $('core'), $('widget'), $('mouse'), $('button'), $('position'),
    $('autocomplete'), $('resizable'), $('draggable'), $('menu'), $('dialog')])
    .pipe(plugins.newer('build/js/jquery.ui.js'))
    .pipe(plugins.sourcemaps.init())
    .pipe(plugins.concat('jquery.ui.js'))
    .pipe(plugins.sourcemaps.write('../maps/js'))
    .pipe(gulp.dest('build/js'));
});

gulp.task('bboxer', ['sass', 'browserify', 'jquery.ui', 'awesome_fonts', 'media'],
  function () {
    var assets = plugins.useref.assets({}, assets_pipeline);
    return gulp.src('src/bboxer.html')
      .pipe(assets)
      .pipe(plugins.if(['**/*.js', '!**/jquery.min.js'],
        plugins.uglify({ie_proof: false}).on('error', plugins.util.log)))
      .pipe(plugins.if('*.css', lazypipe()
        //.pipe(plugins.shorthand)
        .pipe(plugins.autoprefixer)
        .pipe(plugins.minifyCss,
        {roundingPrecision: -1, keepSpecialComments: 0})()))
      .pipe(plugins.sourcemaps.write('maps'))
      .pipe(assets.restore())
      .pipe(plugins.useref())
      .pipe(plugins.if('*.html', gulp.dest(templates_dist), gulp.dest(dist)));
  });

gulp.task('default', ['bboxer']);
