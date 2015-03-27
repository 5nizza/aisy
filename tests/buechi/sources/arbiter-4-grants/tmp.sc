$aut = load tmp.gff;
$aut2 = determinization -m bk09 $aut;
$aut3 = $aut2;
#simplify $aut2;
$aut4 = acc -min $aut3;
save $aut4 tmp-out.gff;
