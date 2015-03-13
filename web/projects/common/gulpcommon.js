var fs = require('fs');
try {
  fs.statSync('node_modules/patched');
}
catch (e) {
  var exec = require('child_process').execFileSync;
  exec('/bin/sh', ['patch.sh'], {cwd: __dirname});
}


global.gulp = require('gulp');
global.babelify = require('babelify');
global.browserify = require('browserify');
global.through2 = require('through2');
global.shrthnd = require('shrthnd');
global.del = require('del');
global.pngquant = require('imagemin-pngquant');
global.lazypipe = require('lazypipe');
global.dist = '../../dist/';
global.templates_dist = dist + "templates/";


plugins.shorthand = function () {
  return through2.obj(function (file, enc, next) {
    var contents = file.contents.toString();
    var res = shrthnd(contents).string;
    console.log(file.path + ": shrthnd saved " + (contents.length - res.length) + " bytes");
    file.contents = new Buffer(res);
    next(null, file);
  });
};

global.assets_pipeline = lazypipe().pipe(plugins.sourcemaps.init,
  {loadMaps: true, debug: true});


gulp.task('bower', function () {
  return gulp.src('bower.json')
    .pipe(plugins.newer('src/libs/status-bower.json'))
    .on('data', function () {
      plugins.bower('src/libs');
    })
    .pipe(plugins.rename('status-bower.json'))
    .pipe(gulp.dest('src/libs'));
});

gulp.task('watch-bower', function () {
  gulp.watch(['bower.json'], ['bower']);
});

gulp.task('sass', ['bower'], function () {
  return gulp.src('src/sass/*.scss')
    .pipe(plugins.newer({dest: 'build/css', ext: '.css'}))
    .pipe(plugins.sourcemaps.init())
    .pipe(plugins.sass({
      precision: 8,
      includePaths: ['src/libs/bootstrap-sass/assets/stylesheets']
    }))
    .on('error', function (err) {
      console.log(err);
    })
    .pipe(plugins.sourcemaps.write())
    .pipe(gulp.dest('build/css'));
});

gulp.task('watch-sass', function () {
  gulp.watch(['src/sass/*.scss'], ['sass', 'default']);
});

gulp.task('clean', function () {
  del(dist + '*', {force: true});
});

gulp.task('mrproper', ['clean'], function () {
  del('build/*');
});

gulp.task('nuke', ['mrproper'], function () {
  del('src/libs/*');
});

gulp.task('fonts-common', function () {
  return gulp.src('../common/src/fonts/*')
    .pipe(plugins.newer(dist + 'fonts'))
    .pipe(gulp.dest(dist + 'fonts'));
});

gulp.task('fonts', ['fonts-common'], function () {
  return gulp.src('src/libs/bootstrap-sass/assets/fonts/bootstrap/*')
    .pipe(plugins.newer(dist + 'fonts/bootstrap'))
    .pipe(gulp.dest(dist + 'fonts/bootstrap'));
});

gulp.task('media', function () {
  return gulp.src(['src/img/[^_]*', '../common/src/img/*'])
    .pipe(plugins.newer(dist + 'img'))
    .pipe(plugins.imagemin({
      multipass: true,
      progressive: true,
      optimizationLevel: 6,
      use: [pngquant()]
    }))
    .pipe(gulp.dest(dist + 'img'));
});

gulp.task('watch-media', function () {
  gulp.watch(['src/img/[^_]*'], ['media']);
});

gulp.task('browserify', function () {
  return gulp.src('src/js/*.js')
    .pipe(plugins.newer('build/js'))
    .pipe(through2.obj(function (file, enc, next) {
      browserify(file.path, {debug: true})
        .transform(babelify)
        .bundle(function (err, res) {
          if (err) {
            return next(err);
          }
          file.contents = res;
          next(null, file);
        });
    }))
    // Super ugly fix for missing semicolon preventing from concatting
    .pipe(plugins.replace("},{}]},{},[1])", "},{}]},{},[1]);"))
    .pipe(gulp.dest('build/js'))
});

gulp.task('watch-browserify', function () {
  gulp.watch(['src/js/*.js'], ['default']);
});
