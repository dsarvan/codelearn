set terminal pngcairo font 'Times,11'
set encoding utf8
set multiplot \
    title 'Trigonometric Function' \
    layout 2,2 columnsfirst \
    margins 0.08,0.99,0.1,0.93 \
    spacing 0.2,0.1

set tics scale 0.5 font ',10'
set xtics pi format "%.0P\317\200"
set xtics add ('0' 0); set ytics 0.5
set key off; set grid
set xrange [-2*pi:2*pi]
set ylabel 'sin(x)'
plot sin(x) lc rgb '#009e73'
set xlabel 'x'
set ylabel 'cos(x)'
plot cos(x) lc rgb '#9a0200'
unset xlabel
set ylabel 'sin(2x)'
plot sin(2*x) lc rgb '#f7022a'
set ylabel 'cos(2x)'
set xlabel 'x'
plot cos(2*x) lc rgb '#001146'

unset multiplot
