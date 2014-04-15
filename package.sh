#!/bin/sh

if test "$1" = "smudge"; then
  commit=$(git rev-parse HEAD)
  date=$(git log --pretty=format:"%ad" --date=rfc2822 -1)
  sed -e "s/\\\$Commit\\\$/$commit/g" -e "s/\\\$Date\\\$/$date/g" <&0
  cat <&0
elif test "$1" = "clean"; then
  sed -e "s/\(__git__ = \"\).*/\\1\\\$Commit\\\$\"/g" -e "s/\\(__date__ = mktime_tz(parsedate_tz(\"\\).*/\\1\\\$Date\\\$\"))/g" <&0
fi
