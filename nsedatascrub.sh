#!/usr/bin/env bash
# File: nsedatascrup.sh
# Name: D.Saravanan
# Date: 03/12/2020

# Script to scrup the NSE historical index data of the following structure:
# "Date","Open","High","Low","Close","Shares Traded","Turnover (Rs. Cr)"
# "01-Jan-2019","    10881.70","    10923.60","    10807.10","    10910.10","      159404542","         8688.26"
# "02-Jan-2019","    10868.85","    10895.35","    10735.05","    10792.50","      309665939","        15352.25"
# "03-Jan-2019","    10796.80","    10814.05","    10661.25","    10672.25","      286241745","        15030.45"
# "04-Jan-2019","    10699.70","    10741.05","    10628.65","    10727.35","      296596655","        14516.74"
# "07-Jan-2019","    10804.85","    10835.95","    10750.15","    10771.80","      269371080","        12731.29"
# "08-Jan-2019","    10786.25","    10818.45","    10733.25","    10802.15","      277697672","        13433.48"
# "09-Jan-2019","    10862.40","    10870.40","    10749.40","    10855.15","      333010535","        16213.30"
# "10-Jan-2019","    10859.35","    10859.35","    10801.80","    10821.60","      254365477","        12031.26"
# "11-Jan-2019","    10834.75","    10850.15","    10739.40","    10794.95","      260792200","        13084.60"

# Data Source: https://www1.nseindia.com/products/content equities/indices/historical_index_data.htm

echo "Enter NSE data file name: "
read fname

if [ -f $fname ]
then

    for n in {2..14..2}
    do
        cut -d '"' -f $n $fname > file$n.txt
    done

    # combine data column wise
    paste -d ',' file{2..14..2}.txt > result.csv

    # remove spaces between the columns
    tr -d '[:blank:]' < result.csv > $fname

    # delete file{2..14..2}.txt and result.csv
    rm file{2..14..2}.txt result.csv
    
else
    echo "Not a regular file."

fi
