#!/bin/bash

add_path()
{
    FILE=$PWD/$1

    if [ -d "$FILE" ]; then
        export PYTHONPATH=$FILE:$PYTHONPATH
    else
        echo "[ERROR] $FILE is nonexistent"
    fi
}

add_path data_gen
add_path feeders
add_path graph
add_path model

python3 $@

rm -rf */__pycache__ __pycache__