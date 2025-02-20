 chmod +x make_bin.sh && ./make_bin.sh 
 ./cut 707,0,2848,2848 data/*/*/
 chmod +x do_align.sh && ./do_align.sh data/*/*/
 ./stack -m11 -a12 data/*/*/
 ./findEdges data/*/*/wip/Stack-m11.Aligned-a12.png
