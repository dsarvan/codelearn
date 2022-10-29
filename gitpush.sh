#!/usr/bin/env bash
# File: gitpush.sh
# Name: D.Saravanan
# Date: 03/12/2021
# Script to push changes into remote git repository

remote="origin1 origin2 origin3 origin4 origin5"

for origin in $remote
do
    git push -u "$origin" master
done
