#!/usr/bin/env bash
# File: pigconf.sh
# Name: D.Saravanan
# Date: 20/09/2020
# Script to install and configure Apache Pig-0.17.0 in Hadoop-3.2.1

user='raman'

# Nodes
mnode='172.17.0.2'
enode='172.17.0.5'
nodes='172.17.0.3 172.17.0.4'

for ip in $mnode $enode $nodes; do
	ssh $user@$ip <<EOF

	if [ $ip == $mnode ]
	then
		wget -c http://apachemirror.wuchna.com/pig/pig-0.17.0/pig-0.17.0.tar.gz -P /home/$user/Downloads/
		tar -xzf /home/$user/Downloads/pig-0.17.0.tar.gz
		mv /home/$user/pig-0.17.0/ /usr/local/hadoop/pig/

		# copy pig to nodes
		scp -r /usr/local/hadoop/pig/ $user@172.17.0.3:/usr/local/hadoop/pig/
		scp -r /usr/local/hadoop/pig/ $user@172.17.0.4:/usr/local/hadoop/pig/

		# copy pig to edge node
		scp -r /usr/local/hadoop/pig/ $user@172.17.0.5:/usr/local/hadoop/pig/
	fi

	# .bashrc
	sed -i '$ a export PIG_HOME=\/usr\/local\/hadoop\/pig \
	\nexport PIG_CONF_DIR=\$PIG_HOME\/conf \
	\nexport PIG_CLASSPATH=\$HADOOP_HOME\/conf \
	\nexport PATH=\$PIG_HOME\/bin:\$PATH' /home/$user/.bashrc

	source ~/.bashrc
		
	logout

EOF
done
