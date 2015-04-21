plugins = require('gulp-load-plugins')();
require('../common/gulpcommon.js');


gulp.task('status', ['sass', 'browserify'], function () {
  var assets = plugins.useref.assets({}, assets_pipeline);
  return gulp.src('src/status.html')
    .pipe(plugins.nunjucksHtml())
    .pipe(assets)
    .pipe(plugins.if(['**/*.js', '!**/viz.js', '!**/jquery.min.js'],
      plugins.uglify({ie_proof: false}).on('error', plugins.util.log)))
    .pipe(plugins.if('*.css', lazypipe()
      //.pipe(plugins.shorthand)
      .pipe(plugins.autoprefixer)
      .pipe(plugins.minifyCss,
      {roundingPrecision: -1, keepSpecialComments: 0})()))
    .pipe(plugins.sourcemaps.write('maps'))
    .pipe(assets.restore())
    .pipe(plugins.useref())
    .pipe(plugins.if('*.html', lazypipe()
      .pipe(plugins.minifyHtml, {empty: true, loose: true})
      .pipe(plugins.replace, 'viz.js', 'viz.js async defer')()))
    .pipe(gulp.dest(dist));
});

gulp.task('view', ['browserify'], function () {
  var assets = plugins.useref.assets({}, assets_pipeline);
  return gulp.src('src/view.html')
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
    .pipe(plugins.if('*.html', plugins.minifyHtml()))
    .pipe(gulp.dest(dist));
});

gulp.task('jquery.ui-js', ['bower'], function () {
  var $ = function (file) {
    return 'src/libs/jquery.ui/ui/' + file + '.js';
  };
  return gulp.src([
    $('core'), $('widget'), $('mouse'), $('button'), $('position'), $('menu'),
    $('selectmenu'), $('slider'), $('sortable')])
    .pipe(plugins.newer('build/js/jquery.ui.js'))
    .pipe(plugins.sourcemaps.init())
    .pipe(plugins.concat('jquery.ui.js'))
    .pipe(plugins.sourcemaps.write('../maps/js'))
    .pipe(gulp.dest('build/js'));
});

gulp.task('jquery.ui-css', ['bower'], function() {
  var $ = function(file) {
    return 'src/libs/jquery.ui/themes/base/' + file + '.css';
  };
  return gulp.src([
    $('core'), $('button'), $('menu'), $('selectmenu'), $('slider'),
    $('sortable'), $('theme')])
    .pipe(plugins.newer('build/css/jquery.ui.css'))
    .pipe(plugins.replace('images/', '../img/'))
    .pipe(plugins.sourcemaps.init())
    .pipe(plugins.concat('jquery.ui.css'))
    .pipe(plugins.sourcemaps.write('../maps/css'))
    .pipe(gulp.dest('build/css'));
});


gulp.task('jquery.ui-images', ['bower'], function () {
  var img = dist + "img";
  return gulp.src('src/libs/jquery.ui/themes/base/images/*')
    .pipe(plugins.newer(img))
    .pipe(gulp.dest(img));
});

gulp.task('jquery.ui', ['jquery.ui-js', 'jquery.ui-css', 'jquery.ui-images']);

gulp.task('logs', ['sass', 'browserify', 'jquery.ui'], function () {
  var assets = plugins.useref.assets({}, assets_pipeline);
  return gulp.src('src/logs.html')
    .pipe(assets)
    .pipe(plugins.if(['**/*.js', '!**/jquery.min.js'],
      plugins.uglify({
        ie_proof: false, mangle: { except: ["$super"] }})
        .on('error', plugins.util.log)))
    .pipe(plugins.if('*.css', lazypipe()
      //.pipe(plugins.shorthand)
      .pipe(plugins.autoprefixer)
      .pipe(plugins.minifyCss,
      {roundingPrecision: -1, keepSpecialComments: 0})()))
    .pipe(plugins.sourcemaps.write('maps'))
    .pipe(assets.restore())
    .pipe(plugins.useref())
    .pipe(plugins.if('*.html', lazypipe()
        .pipe(plugins.minifyHtml)
        .pipe(gulp.dest, templates_dist)(),
      gulp.dest(dist)));
});

gulp.task('frontend', ['sass', 'browserify'], function () {
  var assets = plugins.useref.assets({}, assets_pipeline);
  return gulp.src('src/frontend.html')
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
    .pipe(plugins.if('*.html', lazypipe()
      .pipe(plugins.minifyHtml)
      .pipe(gulp.dest, templates_dist)(),
      gulp.dest(dist)));
});

gulp.task('default', ['status', 'logs', 'frontend', 'view', 'fonts', 'media']);
