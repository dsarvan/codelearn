#!/usr/bin/env bash
# File: hbaseconf.sh
# Name: D.Saravanan
# Date: 09/09/2020
# Bash script to install and configure HBase-2.2.5 in Hadoop-3.2.1

user="raman"

# Nodes
mnode='172.17.0.2'
enode='172.17.0.5'
nodes='172.17.0.3 172.17.0.4'

for ip in $mnode $enode $nodes
do 
	ssh $user@$ip << EOF

	if [ $ip == $mnode ]
	then
		wget -c http://apachemirror.wuchna.com/hbase/2.2.5/hbase-2.2.5-bin.tar.gz -P /home/$user/Downloads/
		tar -xzf /home/$user/Downloads/hbase-2.2.5-bin.tar.gz
		mv /home/$user/hbase-2.2.5/ /usr/local/hadoop/hbase/
	
		# copy hbase to nodes
		scp -r /usr/local/hadoop/hbase/ $user@172.17.0.3:/usr/local/hadoop/hbase/
		scp -r /usr/local/hadoop/hbase/ $user@172.17.0.4:/usr/local/hadoop/hbase/
	
		# hbase-site.xml
		sed -i '/<configuration>/,/<\/configuration>/{//!d}' /usr/local/hadoop/hbase/conf/hbase-site.xml
	
		sed -i 's/<configuration>/& \
		\n\t<property> \
		\n\t\t<name>hbase.master<\/name> \
		\n\t\t<value>172.17.0.2:60000<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>hbase.rootdir<\/name> \
		\n\t\t<value>hdfs:\/\/172.17.0.2:9000\/hbase<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>hbase.cluster.distributed<\/name> \
		\n\t\t<value>true<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>hbase.zookeeper.property.dataDir<\/name> \
		\n\t\t<value>hdfs:\/\/172.17.0.2:9000\/zookeeper<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>hbase.zookeeper.quorum<\/name> \
		\n\t\t<value>172.17.0.2<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>hbase.zookeeper.property.clientPort<\/name> \
		\n\t\t<value>2181<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>hbase.unsafe.stream.capability.enforce<\/name> \
		\n\t\t<value>false<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>zookeeper.znode.parent<\/name> \
		\n\t\t<value>\/hbase<\/value> \
		\n\t<\/property> \
		/' /usr/local/hadoop/hbase/conf/hbase-site.xml

		# copy hbase to edge node
		scp -r /usr/local/hadoop/hbase/ $user@172.17.0.5:/usr/local/hadoop/hbase/

		# region servers
		sed -i '/localhost/s/^/#/g' /usr/local/hadoop/hbase/conf/regionservers
		sed -i '$ a 172.17.0.3 \
		\n172.17.0.4' /usr/local/hadoop/hbase/conf/regionservers		
	fi
		
	if [ $ip != $mnode ] && [ $ip != $enode ]
	then
		sed -i '/<configuration>/,/<\/configuration>/{//!d}' /usr/local/hadoop/hbase/conf/hbase-site.xml
		
		sed -i 's/<configuration>/& \
		\n\t<property> \
		\n\t\t<name>hbase.rootdir<\/name> \
		\n\t\t<value>hdfs:\/\/172.17.0.2:9000\/hbase<\/value> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>hbase.cluster.distributed<\/name> \
		\n\t\t<value>true<\/value> \
		\n\t<\/property> \
		/' /usr/local/hadoop/hbase/conf/hbase-site.xml
	fi
	
	# hbase-env.sh
	sed -i 's/JAVA_HOME/& \
	export JAVA_HOME=\/usr\/lib\/jvm\/adoptopenjdk-8-hotspot-amd64 \
	/' /usr/local/hadoop/hbase/conf/hbase-env.sh 
	sed -i '/# export JAVA_HOME/d' /usr/local/hadoop/hbase/conf/hbase-env.sh
	sed -i '/HBASE_MANAGES_ZK=true/s/^#//g' /usr/local/hadoop/hbase/conf/hbase-env.sh
	
	# .bashrc
	sed -i '$ a export HBASE_HOME=\/usr\/local\/hadoop\/hbase \
	\nexport PATH=\$HBASE_HOME\/bin:\$PATH' /home/$user/.bashrc

	source ~/.bashrc

	logout

EOF
done
