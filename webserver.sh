#!/bin/bash
docker compose run --rm  -p 127.0.0.1:8099:8099 freqtrade webserver --config user_data/webconfig.json
