#!/usr/bin/env bash
# File: doasconf.sh
# Name: D.Saravanan
# Date: 25/06/2024
# Script to check /etc/doas.conf for syntax errors

if doas -C /etc/doas.conf; then
	echo "No syntax errors in doas configuration."
else
	echo "Syntax errors in doas configuration."
fi
