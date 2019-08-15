#!/usr/bin/env bash

rm -rf include

mkdir include

g++ -I../../../src/ -M main.cpp | python parse.py | xargs -I % sh -c "dirname %" | sort | uniq | xargs -I % sh -c "mkdir -p include/%"

g++ -I../../../src/ -M main.cpp | python parse.py | xargs -I % sh -c "cp ../../../src/% include/%"
