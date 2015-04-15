plugins = require('gulp-load-plugins')();
require('../common/gulpcommon.js');


gulp.task('forge', ['sass', 'browserify'], function() {
  var assets = plugins.useref.assets({}, assets_pipeline);
  return gulp.src('src/forge.html')
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

gulp.task('forge_image', function() {
  return gulp.src('src/forge_image.html')
    .pipe(plugins.minifyHtml())
    .pipe(gulp.dest(templates_dist));
});

gulp.task('default', ['forge', 'forge_image', 'fonts', 'media']);
