#!/bin/bash

docker compose run --rm freqtrade download-data  --timeframes 1m 5m 15m 1h 4h 1d 1M --days 1000 --config user_data/btconf.json

docker compose run --rm freqtrade download-data  --timeframes 1m 5m 15m 1h 4h 1d 1M --days 1000 --config user_data/btconf-usd.json

