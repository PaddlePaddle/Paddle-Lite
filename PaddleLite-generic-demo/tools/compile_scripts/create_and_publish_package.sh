#!/bin/bash
set -e

source settings.sh

cd $ROOT_DIR/.. && tar -czvf PaddleLite-generic-demo.tar.gz --exclude=".git" --exclude="*.nb" --exclude="log.txt" --exclude="Makefile" --exclude="CMakeFiles" --exclude="assets/models/*" --exclude="projects/*"  --exclude="tools/compile_scripts/sdk/*" --exclude="tools/compile_scripts/workspace/*" --exclude="tools/compile_scripts/settings.sh" PaddleLite-generic-demo

echo "Done."
