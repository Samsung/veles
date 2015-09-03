#!/bin/sh -e

if [ ! -e /etc/debian_version ]; then
  echo "Only Debian-based distros are supported by this script."
fi

uver=$(lsb_release -cs)

if [ $(id -u) != 0 ]; then
  echo "Requesting sudo..."
  sudo echo "Thanks"
fi

wget -O - https://velesnet.ml/apt/velesnet.ml.gpg.key | sudo apt-key add -
if ! grep -q "deb [arch=amd64] https://velesnet.ml/apt $uver main" /etc/apt/sources.list; then
  echo "deb [arch=amd64] https://velesnet.ml/apt $uver main" | sudo tee -a /etc/apt/sources.list
fi
sudo apt-get update
sudo apt-get install python3-veles