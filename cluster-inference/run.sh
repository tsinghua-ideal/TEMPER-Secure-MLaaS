#!/bin/bash
TARGET_NAME=$1

projects=$(find ${TARGET_NAME} -mindepth 1 -maxdepth 1 -type d)

for project in $projects; do
  project_name=$(basename "$project")
  if [ "$project_name" == "client" ]; then
    continue
  fi

  echo "enter: $project"
  pushd "$project"
  pwd
  cargo run --release &
  sleep 5
  echo "leave: $project"
  popd
done

pushd ${TARGET_NAME}/client
cargo run --release
popd