#!/usr/bin/env bash
# File: bkpcloud.sh
# Name: D.Saravanan
# Date: 05/01/2021
# Bash script to sync files to pcloud using rclone

user=$(whoami)
tput clear

echo -e "\nSync PCloud"
echo -e "Today's date is $(date | tr -s ' ' | cut -d ' ' -f 2,3,6)\n"
count=0
echo "$count BookMarks"
rclone sync -P /home/$user/.config/qutebrowser/bookmarks/urls pcloud:Bookmarks/

directories="Articles Books DataScience Numerical"

for directory in $directories; do
	count=$(expr $count + 1)
	echo -e "\n$count $directory"
	rclone sync -P /home/$user/$directory pcloud:$directory
done

count=$(expr $count + 1)
echo -e "\nThe number of synced directories: $count \nAll done!"
