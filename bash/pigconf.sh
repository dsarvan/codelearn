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
	ssh -p 22 $user@$ip << EOF

	if [ $ip == $mnode ]
	then

		wget -c http://apachemirror.wuchna.com/pig/pig-0.17.0/pig-0.17.0.tar.gz -P /home/$user/Downloads/
		tar -xzf /home/$user/Downloads/pig-0.17.0.tar.gz
		mv /home/$user/pig-0.17.0/ /usr/local/hadoop/pig/

	fi
		
	logout

EOF
done
