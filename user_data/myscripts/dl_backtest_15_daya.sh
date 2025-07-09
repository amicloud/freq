#!/usr/bin/env bash
#
# download_all_timeframes.sh
#
# Downloads 45 days of history for multiple timeframes from Kraken,
# ending one day prior to today (i.e. up through 2025-06-01).
#

# 1) Set “today” (static for this example) and compute start/end dates:
TODAY="2025-06-02"

# End date = one day before TODAY (i.e. 2025-06-01)
END_DATE=$(date -d "${TODAY} -1 day" +%Y%m%d)

# Start date = 15 days before TODAY (i.e. 2025-04-18)
START_DATE=$(date -d "${TODAY} -5 days" +%Y%m%d)

echo "Downloading from ${START_DATE} to ${END_DATE} (inclusive)."
echo

# 2) Define the list of timeframes you want:
TIMEFRAMES=( "1m" "5m" )

# 3) Define your pair list (separated by spaces).
#    Make sure each pair is exactly as Kraken expects (e.g., “ETH/USDT” not “ETHUSDT”).
PAIRS="ETH/USDT BTC/USDT DOGE/USDT TRUMP/USDT XRP/USDT LPT/USD SOL/USDT"

# 4) Loop over each timeframe and invoke download-data via Docker Compose:
for TF in "${TIMEFRAMES[@]}"; do
	    echo "-----------------------------------------------"
	        echo "Downloading ${TF} candles for pairs: ${PAIRS}"
		    echo "Timerange: ${START_DATE}-${END_DATE}"
		        echo "-----------------------------------------------"
			    docker compose run --rm freqtrade download-data \
			          --exchange kraken\
 				        --timeframe "${TF}" \
				      --pairs ${PAIRS} \
			            --timerange "${START_DATE}-${END_DATE}"  --erase 

			        echo
			done

echo "All timeframes downloaded!"
