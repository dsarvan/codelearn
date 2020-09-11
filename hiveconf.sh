#!/usr/bin/env bash
# File: hiveconf.sh
# Name: D.Saravanan
# Date: 28/08/2020
# Bash script to install and configure Hive-3.1.2 in Hadoop-3.2.1

user="raman"

mnode='172.17.0.2'
enode='172.17.0.5'
nodes='172.17.0.3 172.17.0.4'

for ip in $mnode $enode
do
	ssh $user@$ip << EOF

	if [ $ip == $mnode ]
	then
		wget -c http://apachemirror.wuchna.com/hive/hive-3.1.2/apache-hive-3.1.2-bin.tar.gz -P /home/$user/Downloads/
		tar -xzf /home/$user/Downloads/hive-3.1.2-bin.tar.gz
		mv /home/$user/hive-3.1.2/ /usr/local/hadoop/hive/

		# hive-site.xml
		sed -i '/<configuration>/,/<\/configuration>/{//!d}' /usr/local/hadoop/hive/conf/hive-site.xml

		sed -i 's/<configuration>/& \
		\n\t<property> \
		\n\t\t<name>javax.jdo.option.ConnectionURL<\/name> \
		\n\t\t<value>jdbc.mysql://localhost/metastore?createDatabaseIfNotExist=true<\/value> \
		\n\t\t<description>metadata is stored in a MySQL server<\/description> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>javax.jdo.option.ConnectionDriverName<\/name> \
		\n\t\t<value>com.mysql.jdbc.Driver<\/value> \
		\n\t\t<description>MySQL JDBC driver class<\/description> \
		\n\t<\/property> \
		\n\t<property> \
		\n\t\t<name>javax.jdo.option.ConnectionUserName<\/name> \
		\n\t\t<value>raman<\/value> \
		\n\t\t<description>user name for connecting to mysql server<\/description> \
		\n\t<\/property> \
		/' /usr/local/hadoop/hive/conf/hive-site.xml

		# copy hive to edge node
		scp -r /usr/local/hadoop/hive/ $user@$enode:/usr/local/hadoop/hive/
		fi

		# .bashrc
		sed -i '$ a export HIVE_HOME=\/usr\/local\/hadoop\/hive \
		\nexport HIVE_CONF_DIR=\/usr\/local\/hadoop\/hive\/conf \
		\nexport PATH=\$HIVE_HOME\/bin:\$PATH \
		\nexport CLASSPATH=$CLASSPATH:\/usr\/local\/hadoop\/lib\/*:. \
		\nexport CLASSPATH=$CLASSPATH:\/usr\/local\/hadoop\/hive\/lib\/*:.' /home/$user/.bashrc

		source ~/.bashrc

		if [ $ip == $enode ]
		then
			hdfs dfs -mkdir -p /user/hive/warehouse
		fi

		logout

EOF
done
