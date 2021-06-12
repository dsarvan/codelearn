#!/bin/bash

echo "This scripts checks the existence of the messages file."
echo "Checking..."
if [ -f /var/log/messages ]
then
	echo "/var/log/messages exists."
fi
echo
echo "...done."
