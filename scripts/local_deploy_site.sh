#!/bin/bash
set -euxo pipefail

scripts/build_site.sh

cd site && jekyll serve  --skip-initial-build --host localhost --baseurl /mlc-llm --port 8888
