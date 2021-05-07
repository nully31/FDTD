#-------------------------------------------------------------------------------
# gnuplotの設定
#-------------------------------------------------------------------------------
reset
set nokey                 # 凡例の非表示
set xrange [0:40]       # x軸方向の範囲の設定
set yrange [0:40]       # y軸方向の範囲の設定
set zrange [-10000:10000]
set cbrange [-0.0005:0.0005]
set size square           # 図を正方形にする
set ticslevel 0
set dgrid3d 30,30
set hidden3d
set pm3d
set pm3d map
set palette defined ( 0 '#000090',1 '#000fff',2 '#0090ff',3 '#0fffee',4 '#90ff70',5 '#ffee00',6 '#ff7000',7 '#ee0000',8 '#7f0000')
#set palette rgb 33,13,10

#set term gif animate      # 出力をgifアニメに設定
#set output "output_2d.gif"  # 出力ファイル名の設定

#-------------------------------------------------------------------------------
# 変数の設定
#-------------------------------------------------------------------------------
n0 = 0    # ループ変数の初期値
n1 = 1000   # ループ変数の最大値
dn = 1    # ループ変数の増加間隔

#-------------------------------------------------------------------------------
# ループ処理
#-------------------------------------------------------------------------------
if(exist("n")==0 || n<0) n = n0  # ループ変数の初期化

#-------------------------------------------------------------------------------
# プロット
#-------------------------------------------------------------------------------
splot "result.dat" index n u 1:2:3 # n番目のデータのプロット
#pause 8e-8
#-------------------------------------------------------------------------------
# ループ処理
#-------------------------------------------------------------------------------
n = n + dn            # ループ変数の増加
if ( n < n1 ) reread  # ループの評価
undefine n
