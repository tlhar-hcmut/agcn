#!/bin/sh
ENTRY_PATH=`readlink -f "$0"`
PROJECT_HOME=`dirname "$ENTRY_PATH"`
cd $PROJECT_HOME
SERVER=${1-khoidd@khoidd.xyz}
USER=${2-khoidd}
APP=${2-train_khoidd}
PATH_DEPLOY="/code/$USER/stgcn"

rsync -aurv -e 'ssh -p 3000'  \
    "$PROJECT_HOME/src" \
    "$PROJECT_HOME/requirements.txt" \
    $SERVER:$PATH_DEPLOY/  --delete