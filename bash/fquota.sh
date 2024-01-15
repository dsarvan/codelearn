#!/usr/bin/env bash
# File: fquota
# Name: D.Saravanan
# Date: 12/07/2021
# Disk quota analysis tool for Unix (assumes all user accounts are >= UID 1000)

MAXDISKUSAGE=20000 # in megabytes

for name in $(cut -d: -f1,3 /etc/passwd | awk -F: '$2 > 999 {print $1}'); do
	/bin/echo -n "User $name exceeds disk quota. Disk usage is: "
	find / /usr /var /home -xdev -user $name -type f -ls |
		awk '{ sum += $7 } END { print sum / (1024*1024) " MB" }'
done | awk "\$9 > $MAXDISKUSAGE { print \$0 }"

exit 0
