#!/usr/bin/env bash
# File: sparkconf.sh
# Name: D.Saravanan
# Date: 11/09/2020
# Bash script to install and configure Spark-3.0.1 in Hadoop-3.2.1

user='raman'

# Nodes
mnode='172.17.0.2'
enode='172.17.0.5'
nodes='172.17.0.3 172.17.0.4'

for ip in $mnode $enode $nodes; do
	ssh $user@$ip <<EOF
	
	if [ $ip == $mnode ]
	then
		# scala installation
		wget -c https://downloads.lightbend.com/scala/2.11.12/scala-2.11.12.tgz -P /home/$user/Downloads/
		tar -xvzf /home/$user/Downloads/scala-2.11.12.tgz
		mv /home/$user/scala-2.11.12/ /usr/local/hadoop/scala/

		# spark installation
		wget -c https://mirrors.estointernet.in/apache/spark/spark-3.0.1/spark-3.0.1-bin-hadoop3.2.tgz -P /home/$user/Downloads/
		tar -xvzf /home/$user/Downloads/spark-3.0.1-bin-hadoop3.2.tgz
		mv /home/$user/spark-3.0.1-bin-hadoop3.2/ /usr/local/hadoop/spark/

		# spark-env.sh
		cp /usr/local/hadoop/spark/conf/spark-env.sh.template /usr/local/hadoop/spark/conf/spark-env.sh
		echo " " >> /usr/local/hadoop/spark/conf/spark-env.sh
		sed -i '$ a export HADOOP_HOME=\/usr\/local\/hadoop \
	   	\nexport HADOOP_CONF_DIR=\$HADOOP_HOME\/etc\/hadoop \
		\nexport YARN_CONF_DIR=\$HADOOP_HOME=\$HADOOP_HOME\/etc\/hadoop \
		\nexport SPARK_LOG_DIR=\$HADOOP_HOME\/spark\/log \
		\nexport SPARK_WORKER_DIR=\$HADOOP_HOME\/spark\/work \
		\nexport JAVA_HOME=\/usr\/lib\/jvm\/adoptopenjdk-8-hotspot-amd64 \
		\nexport SPARK_HOME=\/usr\/local\/hadoop\/spark \
		\nexport SCALA_HOME=\/usr\/local\/hadoop\/scala \
		\nexport SPARK_MASTER_HOST=172.17.0.2' /usr/local/hadoop/spark/conf/spark-env.sh

		# slaves
		cp /usr/local/hadoop/spark/conf/slaves.template /usr/local/hadoop/spark/conf/slaves
		sed -i '/localhost/s/^/#/g' /usr/local/hadoop/spark/conf/slaves
		sed -i '$ a 172.17.0.3 \
		\n172.17.0.4' /usr/local/hadoop/spark/conf/slaves

		# copy scala and spark to nodes
		scp -r /usr/local/hadoop/scala/ $user@172.17.0.3:/usr/local/hadoop/scala/
		scp -r /usr/local/hadoop/spark/ $user@172.17.0.3:/usr/local/hadoop/spark/

		scp -r /usr/local/hadoop/scala/ $user@172.17.0.4:/usr/local/hadoop/scala/
		scp -r /usr/local/hadoop/spark/ $user@172.17.0.4:/usr/local/hadoop/spark/

		# copy scala and spark to edge node
		scp -r /usr/local/hadoop/scala/ $user@172.17.0.5:/usr/local/hadoop/scala/
		scp -r /usr/local/hadoop/spark/ $user@172.17.0.5:/usr/local/hadoop/spark/
	fi

	# make directory
	#mkdir -p /var/log/spark
	#chown -R $user:$user /var/log/spark
	#mkdir -p /tmp/spark
	#chown -R $user:$user /tmp/spark

	# .bashrc
	sed -i '$ a export SCALA_HOME=\/usr\/local\/hadoop\/scala \
	\nexport PATH=\$PATH:\$SCALA_HOME\/bin \
	\nexport SPARK_HOME=\/usr\/local\/hadoop\/spark \
	\nexport PATH=\$PATH:\$SPARK_HOME\/bin' /home/$user/.bashrc

	source ~/.bashrc

	logout

EOF
done
