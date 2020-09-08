#!/usr/bin/env bash

# Nodes
mnode='172.17.0.2'
enode='172.17.0.5'
nodes='172.17.0.3 172.17.0.4'

for ip in $mnode $enode $nodes
do 

ssh $user@$ip << EOF

user="raman"
mnode='172.17.0.2'
enode='172.17.0.5'
nodes='172.17.0.3 172.17.0.4'

ipaddr=`hostname -I`

if [ "$ipaddr -eq $mnode" ]
then

wget -c http://apachemirror.wuchna.com/hbase/2.2.5/hbase-2.2.5-bin.tar.gz -P /home/$user/Downloads/
tar -xzf /home/$user/Downloads/hbase-2.2.5-bin.tar.gz
mv /home/$user/Downloads/hbase-2.2.5/ /usr/local/hadoop/hbase/

# copy hbase to nodes
for ip in $nodes
do
scp -r /usr/local/hadoop/hbase/ raman@$ip:/usr/local/hadoop/hbase/
done

# hbase-site.xml
sed -i '/<configuration>/,/<\/configuration>/{//!d}' /usr/local/hadoop/hbase/conf/hbase-site.xml

sed -i 's/<configuration>/& \
\t<property> \
\t\t<name>hbase.master<\/name> \
\t\t<value>172.17.0.2:60000<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>hbase.rootdir<\/name> \
\t\t<value>hdfs:\/\/172.17.0.2:9000\/hbase<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>hbase.cluster.distributed<\/name> \
\t\t<value>true<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>hbase.zookeeper.property.dataDir<\/name> \
\t\t<value>hdfs:\/\/172.17.0.2:9000\/zookeeper<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>hbase.zookeeper.quorum<\/name> \
\t\t<value>172.17.0.2<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>hbase.zookeeper.property.clientPort<\/name> \
\t\t<value>2181<\/value> \
\t<\/property> \
/' /usr/local/hadoop/hbase/conf/hbase-site.xml

# region servers
sed -i '/localhost/s/^/#/g' /usr/local/hadoop/hbase/conf/regionservers
for ip in $nodes
do
echo $ip >> /usr/local/hadoop/hbase/conf/regionservers
done
fi
	
if [ "$ipaddr -ne $mnode" ] && [ "$ipaddr -ne $enode" ]
then
sed -i 's/<configuration>/& \
\t<property> \
\t\t<name>hbase.rootdir<\/name> \
\t\t<value>hdfs:\/\/172.17.0.2:9000\/hbase<\/value> \
\t<\/property> \
\t<property> \
\t\t<name>hbase.cluster.distributed<\/name> \
\t\t<value>true<\/value> \
\t<\/property> \
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
export PATH=$HBASE_HOME\/bin:$PATH' /home/$user/.bashrc
source /home/$user/.bashrc

logout

EOF
done
