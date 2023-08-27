-- File: qsort.hs
-- Name: D.Saravanan
-- Date: 27/08/2023
-- Quick sorting algorithm function

{-
ghci> :load qsort.hs
[1 of 1] Computing Main         ( qsort.hs, interpreted )
Ok, one module loaded.
ghci> qsort [3, 1, 4, 2]
[1,2,3,4]
ghci> qsort (reverse [1 .. 10])
[1,2,3,4,5,6,7,8,9,10]
ghci> qsort "haskell"
"aehklls"
ghci> qsort [True, False, True, False]
[False, False, True, True]
ghci> :type qsort
qsort :: Ord a => [a] -> [a]
-}

qsort [] = []
qsort (x:xs) = qsort ys ++ [x] ++ qsort zs
               where
                  ys = [a | a <- xs, a <= x]
                  zs = [b | b <- xs, b > x]
