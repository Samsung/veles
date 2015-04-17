#!/bin/sh

sfood -I libZnicz -I externalz ../veles | \
grep -v /usr/lib/python | \
sfood-cluster -f sfood_clusters.txt | \
sfood-graph -p | dot -T svg -o deps.svg
