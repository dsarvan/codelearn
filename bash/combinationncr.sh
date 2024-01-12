# File: combinationncr.sh
# Name: D.Saravanan
# Date: 24/02/2020
# Script to find combination C(n,r)

echo "Enter value n: "
read n

echo "Enter value r: "
read r

if [ $n -lt $r ]; then
	echo "Error: The value of n should be greater than or equal to r."
	echo "Enter value n: "
	read n
fi

diff=$(expr $n - $r)

x=1

nvalue=1
while [ $n -ge $x ]; do
	nvalue=$(expr $nvalue \* $n)
	((n--))
done

rvalue=1
while [ $r -ge $x ]; do
	rvalue=$(expr $rvalue \* $r)
	((r--))
done

diffvalue=1
while [ $diff -ge $x ]; do
	diffvalue=$(expr $diffvalue \* $diff)
	((diff--))
done

nCr=$(expr "$nvalue/$rvalue * $diffvalue" | bc)
echo "The number of combinations is $nCr."
