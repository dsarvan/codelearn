#!/usr/bin/env bash
# File: pigconf.sh
# Name: D.Saravanan
# Date: 20/09/2020
# Script to install and configure Apache Pig in Hadoop

user='raman'

# Nodes
mnode='172.17.0.2'
enode='172.17.0.5'
nodes='172.17.0.3 172.17.0.4'

for ip in $mnode $enode $nodes
do
	$user@$ip

done
