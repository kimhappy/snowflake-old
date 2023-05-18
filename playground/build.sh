#!/bin/bash

clean_build_directory() {
  rm -rf build/*
}

while getopts ":f" opt; do
  case ${opt} in
    f )
      clean_build_directory
      ;;
    \? )
      echo "Usage: cmd [-f]"
      ;;
  esac
done

if [ ! -d "build" ]; then
  mkdir build
fi

cd build
cmake .. && cmake --build . && cp playground ../playground
cd ..
rm ../compile_commands.json
ln -s build/compile_commands.json ..
