#!/usr/bin/env bash
# File: mailserv.sh
# Name: D.Saravanan
# Date: 05/01/2021
# Script to count the mail services from a file

echo -e "\nInput file name is $1"

services="hotmail.com gmail.com microsoft.com verizon.net yahoo.com"

for service in $services
do
    echo "Count of $service: `grep -icw $service $1`"
done
