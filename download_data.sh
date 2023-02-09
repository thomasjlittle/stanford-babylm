#!/usr/bin/env bash

set -x

rm -f babylm_data.zip
wget https://github.com/babylm/babylm.github.io/raw/main/babylm_data.zip
unzip babylm_data.zip
rm babylm_data.zip
rm -r __MACOSX
