#!/bin/bash
if [ $# -eq 0 ]
  then
    REMOTE='skywalker'
  else
    REMOTE=$1
fi
echo "Syncing with $REMOTE"
ssh archith@$REMOTE "mkdir -p $PWD"
rsync -av ./* archith@$REMOTE:$PWD

echo "Synced with $REMOTE"
