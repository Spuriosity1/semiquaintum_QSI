set terminal png
set output "out.png"

#set logscale y
set logscale x
set xlabel "Cluster Size"
set ylabel "Num Spins in Cluster"

#set xrange [0:20]
plot for [p in "020 040 060 080 100 120 140 180"] "../tmp/hist_L10_p0.".p."_s0.csv" u 1:($1*$2/10/10/10/16/100) title "p=0.".p w linespoints


