import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import Slider, TextInput, Button, Div
from bokeh.layouts import Column, Row, layout
from bokeh.io import curdoc
from bokeh import events
from bokeh.models import Range1d, Paragraph
from bokeh.models.tools import PanTool, PolyDrawTool
import sys


'''
RA_Deg：赤経(単位：度, 0~360)
Dec_Deg：赤緯(単位：度, -90~90)
mercator_x, mercator_y：赤経赤緯をメルカトル図法に変換した座標
gnomonic_x, gnomonic_y：赤経赤緯を心射方位図法に変換した座標
Magnitude：視等級(-2~15程度)
'''

def load_csv(mode):
    if mode == "a":
        header = ["HIP_num", "RA_h", "RA_m", "RA_s", "Dec_code", "Dec_h", "Dec_m", "Dec_s", "Magnitude"]
        return pd.read_csv("./constellation/hip_lite_a.csv", names = header)
    elif mode == "b":
        header = ["HIP_num", "RA_h", "RA_m", "RA_s", "Dec_code", "Dec_h", "Dec_m", "Dec_s", "Magnitude"]
        return pd.read_csv("./constellation/hip_lite_b.csv", names = header)
    elif mode == "100":
        header = ["HIP_num", "RA_h", "RA_m", "RA_s", "Dec_code", "Dec_h", "Dec_m", "Dec_s", "Magnitude", "Annual_Parallax", "R_PM","D_PM", "BV_Color_Index", "Spectral","Constellation", "Bayer","Name_Eng", "Name_Jpn"]
        return pd.read_csv("./constellation/hip_100.csv", names=header, encoding="SHIFT-JIS")
    elif mode == "name":
        return pd.read_csv("./constellation/hip_name.csv",names=["HIP_num", "Constellation"])
    elif mode == "line":
        return pd.read_csv("./constellation/hip_constellation_line.csv",names=["Constellation", "HIP_num_1", "HIP_num_2"])
    elif mode == "conste":
        return pd.read_csv("./constellation/constellation.csv",names=["Constellation", "Conste_Jpn", "Square_Deg"], encoding="utf-8")
    elif mode == "messier":
        return pd.read_csv("./constellation/messier.csv")
    else:
        return None


Mercator_Base = 180
ROTATE = 90

def Mercator_x(RA_Deg, x0=Mercator_Base):
    #Deg => mercator_x座標
    return -(RA_Deg -x0) / 180 * np.pi
def Mercator_y(Dec_Deg):
    rad = Dec_Deg / 180 * np.pi
    return np.log((1+np.sin(rad))/(1-np.sin(rad))) /2

def inv_Mercator_x(x, x0=Mercator_Base):
    #mercator_x座標 => Deg
    RA_Deg = x0 - 180 * x / np.pi
    return RA_Deg
def inv_Mercator_y(y):
    Dec_Deg = 180 * np.arcsin((np.exp(2*y)-1)/(np.exp(2*y)+1)) / np.pi
    return Dec_Deg

def add_Deg_row_stars(stars):
    #DataFrameに赤経赤緯の情報を追加
    RA_h, RA_m, RA_s = stars["RA_h"], stars["RA_m"], stars["RA_s"]
    RA_Deg = pd.Series((RA_h * 15 + RA_m /4 + RA_s /240 + ROTATE) % 360, name="RA_Deg")
    
    Dec_code, Dec_h, Dec_m, Dec_s =stars["Dec_code"], stars["Dec_h"], stars["Dec_m"], stars["Dec_s"]
    Dec_Deg = pd.Series(2*(Dec_code - 0.5) * (Dec_h + Dec_m / 60 + Dec_s / 3600), name="Dec_Deg")

    mercator_x = pd.Series(Mercator_x(RA_Deg), name="mercator_x") #x座標
    mercator_y = pd.Series(Mercator_y(Dec_Deg), name="mercator_y")
    return pd.concat([stars, RA_Deg, Dec_Deg, mercator_x, mercator_y],axis=1)


def add_Deg_row_galaxy(galaxy):
    #DataFrameに赤経赤緯の情報を追加
    RA_str, Dec_str = galaxy["RA"], galaxy["Dec"]

    RA_h = np.array([int(string[:2]) for string in RA_str.values])
    RA_m = np.array([float(string[5:]) for string in RA_str.values])
    Dec_code = np.array([string[0] for string in Dec_str.values])
    Dec_code = np.where(Dec_code == "+", 1, -1)
    Dec_deg = np.array([int(string[1:3]) for string in Dec_str.values])
    Dec_m = np.array([int(string[6:]) for string in Dec_str.values])

    RA_Deg = (pd.Series(RA_h*15 + RA_m / 4, name="RA_Deg") + ROTATE) % 360
    Dec_Deg = pd.Series(Dec_code*(Dec_deg + Dec_m / 60), name="Dec_Deg")

    mercator_x = pd.Series(Mercator_x(RA_Deg), name="mercator_x")
    mercator_y = pd.Series(Mercator_y(Dec_Deg), name="mercator_y")
    
    return pd.concat([galaxy, RA_Deg, Dec_Deg, mercator_x, mercator_y],axis=1)



def size_config(magnitude):
    #視等級によって星の大きさを変化させる関数
    if magnitude < 1: size = 11
    elif magnitude < 2: size = 9
    elif 1 <= magnitude & magnitude < 6: size = int(7-magnitude)
    else: size=1
    return size





class AngleSimulator():
    #カメラの撮影範囲をシミュレート
    def __init__(self,flag):
        #センサーサイズ(標準はAPS-C)
        self.sensor_size_h = 23.4
        self.sensor_size_v = 16.7
        #焦点距離
        self.focus_dis = 20
        
        self.center_x = 0
        self.center_y = 0
        #回転角
        self.rot = 0
        
        self.angle_simu_flag = flag
        self.rotate_flag = False

        self.calcu_vertex()
    
    def rotate(self, x, y):
        theta = np.deg2rad(self.rot)
        trans_x = x -self.center_x
        trans_y = y -self.center_y

        costheta, sintheta = np.cos(theta), np.sin(theta)

        xd = trans_x*costheta - trans_y*sintheta + self.center_x
        yd = trans_x*sintheta + trans_y*costheta + self.center_y

        return xd, yd
    

    def calcu_vertex(self):
        half_width = self.sensor_size_v / (2*self.focus_dis)
        half_height = self.sensor_size_h / (2*self.focus_dis)

        self.xs = np.array([self.center_x-half_width, self.center_x+half_width, self.center_x+half_width, self.center_x-half_width])
        self.ys = np.array([self.center_y-half_height, self.center_y-half_height, self.center_y+half_height, self.center_y+half_height])
        self.xs, self.ys = self.rotate(self.xs, self.ys)
        self.xs, self.ys = [self.xs], [self.ys]


class StarViewer():

    def __init__(self):
        #データの読み込み・整形・必要な情報の追加
        ##全恒星データ
        self.stars = pd.concat([load_csv("a"), load_csv("b")])
        self.stars = add_Deg_row_stars(self.stars)[["HIP_num","Magnitude", "RA_Deg", "Dec_Deg", "mercator_x", "mercator_y"]]

        ##明るい恒星100個のデータ、self.starsより詳しい
        self.stars_100 = load_csv("100")
        self.stars_100 = add_Deg_row_stars(self.stars_100)
        self.stars_100 = pd.merge(self.stars_100, self.stars[["HIP_num", "mercator_x", "mercator_y"]], how="inner")

        ##星座線のデータ
        self.stars_line = load_csv("line")                
        self.stars_line = pd.merge(self.stars_line, self.stars[["HIP_num","RA_Deg","Dec_Deg", "mercator_x", "mercator_y"]], left_on="HIP_num_1", right_on="HIP_num")
        self.stars_line = self.stars_line.drop(columns="HIP_num")
        self.stars_line = pd.merge(self.stars_line, self.stars[["HIP_num","RA_Deg","Dec_Deg", "mercator_x", "mercator_y"]], left_on="HIP_num_2", right_on="HIP_num")
        self.stars_line = self.stars_line.drop(columns="HIP_num")
        self.stars_line.columns = ["Constellation","HIP_num_1","HIP_num_2",  "RA_Deg_1",  "Dec_Deg_1", "mercator_x_1", "mercator_y_1", "RA_Deg_2", "Dec_Deg_2", "mercator_x_2", "mercator_y_2"]
        ###平面に展開したときに頂点が両端にわかれてしまう線を除く
        self.stars_line = self.stars_line[np.abs(self.stars_line["RA_Deg_1"] - self.stars_line["RA_Deg_2"]) < 180]
        ###メルカトル図法における線の頂点のデータ
        self.mercator_lines_x = [[i1,i2] for i1,i2 in zip(self.stars_line.mercator_x_1.values, self.stars_line.mercator_x_2.values)]
        self.mercator_lines_y = [[i1,i2] for i1,i2 in zip(self.stars_line.mercator_y_1.values,self.stars_line.mercator_y_2.values)]


        '''
        HIPs = self.stars_100["HIP_num"]
        winter = [self.stars_100[HIPs==21421], self.stars_100[HIPs==24436], self.stars_100[HIPs==24608], self.stars_100[HIPs==27989], self.stars_100[HIPs==32349], self.stars_100[HIPs==37279], self.stars_100[HIPs==37826]]
        spring = [self.stars_100[HIPs==65474], self.stars_100[HIPs==49669], self.stars_100[HIPs==69673]]
        summer = [self.stars_100[HIPs==91262], self.stars_100[HIPs==102098], self.stars_100[HIPs==97649]]
        fall = [self.stars_100[HIPs==677], self.stars_100[HIPs==113881], self.stars_100[HIPs==113963], self.stars[self.stars["HIP_num"]==1065]]
        '''


        ##星座名を英語の略称から日本語に変換する辞書
        self.constellations = load_csv("conste")
        self.trans_constellations_name = dict([(conste, name) for conste, name in zip(self.constellations["Constellation"], self.constellations["Conste_Jpn"])])
        
        ##銀河のデータ
        self.galaxy = load_csv("messier")
        self.galaxy = add_Deg_row_galaxy(self.galaxy)

        ##星座を構成している恒星のデータ
        self.stars_constellation = load_csv("name")
        self.stars_constellation = pd.merge(self.stars_constellation, self.stars, on="HIP_num", how="inner")

        
        def calcu_constellations_pos(stars_constellation, constellations):
            ##左右にわかれてしまった星座のリストと、投影する星座の中心座標のDataFrameを計算
            constellations_pos = pd.DataFrame({}, columns = ["Constellation", "RA_Deg", "Dec_Deg"])
            not_plot_constellations = []
            for conste in set(stars_constellation["Constellation"]):
                stars = stars_constellation[stars_constellation["Constellation"] == conste]
                if max(stars["RA_Deg"]) - min(stars["RA_Deg"]) > 180:
                    not_plot_constellations.append(conste)
                else:   
                    RA_Deg = stars["RA_Deg"].mean()
                    Dec_Deg = stars["Dec_Deg"].mean()
                    row = pd.Series([conste, RA_Deg, Dec_Deg], index = ["Constellation", "RA_Deg", "Dec_Deg"])
                    constellations_pos = constellations_pos.append(row, ignore_index=True)
            constellations_pos = pd.merge(constellations_pos, constellations, on="Constellation", how="inner")
            mercator_x = pd.Series(Mercator_x(constellations_pos["RA_Deg"]), name="mercator_x")
            mercator_y = pd.Series(Mercator_y(constellations_pos["Dec_Deg"]), name="mercator_y")
            constellations_pos = pd.concat([constellations_pos, mercator_x, mercator_y],axis=1)
            return constellations_pos,  not_plot_constellations

        self.constellations_pos, self.not_plot_constellations = calcu_constellations_pos(self.stars_constellation, self.constellations)
        print(self.not_plot_constellations)

        #変数の初期設定

        ##現在の投影方法
        self.plotting_mode = "mercator" #or gnomonic
        ##一つ前の投影法
        self.plotting_mode_old = "mercator"
        ##心射方位図法の時の中心座標の計算の仕方
        self.renew_mode = "zoom" #(他に取りうる値：slide, slider, search)


        ##グラフの中心座標
        self.center_pos = (0,0)
        ##ドラッグしているかどうか
        self.LoD_flag = False
        ##明るい恒星の名前の表示
        self.starslabel_flag = True
        ##カメラの撮影範囲をシミュレートしているかどうか
        self.angle_simu_flag = False
        ##銀河の表示(0:何も表示しない, 1:点のみ,2:メシエ番号も)
        self.galaxy_flag = 0
        ##星座線の表示
        self.star_line_flag = False
        ##星座名の表示
        self.constellation_flag = False
        #拡大縮小の時に視等級を連動
        self.plotting_linkage_flag = True



        #sourceの初期状態(空)
        ##self.star_sourcesのリセットに利用する空のデータソース、視等級ごとに別々のデータソースを作成
        self.default_source = dict([(i, ColumnDataSource(data=dict(x=[], y=[]))) for i in range(-2, 16)])
        ##表示している星のデータソース
        self.star_sources = self.default_source.copy()
        ##表示してる星のDataFrame (検索などの判定に利用)
        self.plotting_stars = pd.DataFrame([], columns=self.stars.columns)
        ##恒星の名前のデータソース
        self.stars_label_sources = ColumnDataSource(data=dict(x=[], y=[], label=[]))
        ##検索でヒットした天体のデータソース
        self.searched_celestial_body_source = ColumnDataSource(data=dict(x=[], y=[]))
        ##検索でヒットした天体のデータ
        self.matched_celestial_body = None #pd.Series
        ##星座線のデータソース
        self.star_line_source = ColumnDataSource(data=dict(x=[], y=[]))
        ##星座名のデータソース
        self.constellations_name_source = ColumnDataSource(data=dict(x=[], y=[], label=[]))
        ##カメラの撮影範囲をシミュレートしている枠の座標
        self.square_source1 = ColumnDataSource(data=dict(x=[[]], y=[[]]))
        ##銀河のデータソース
        self.galaxy_source = ColumnDataSource(data=dict(x=[], y=[]))
        ##銀河の名前(メシエ番号)のデータソース
        self.galaxy_label_sources = ColumnDataSource(data=dict(x=[], y=[], label=[]))
        ##表示してる銀河のDataFrame (検索などの判定に利用)
        self.plotting_galaxies = pd.DataFrame([], columns=self.galaxy.columns)

        self.make_tools()
        self.make_figure()
        self.AngleSimu = AngleSimulator(self.angle_simu_flag)




    def make_figure(self):
        #図を作成
        TOOLS = "pan,wheel_zoom,zoom_in,zoom_out,reset"
        TOOLTIPS=[("x","@x"),("y","@y")]
        import ctypes
        screen_width = ctypes.windll.user32.GetSystemMetrics(0)
        print(screen_width)
        x,y = (4,4)
        sizing = 300
        
        self.fig = figure(tools=TOOLS, title="target", x_range=(-x,x), y_range=(-y,y), background_fill_color ="#17184b", output_backend="webgl")
        self.add_figure_event()

        def plot_for(sources, color="#FFFF66"):
            #プロットの簡略化
            for i, source in enumerate(sources):
                size = size_config(i)
                self.fig.circle("x", "y", source=source, size=size, color=color, alpha=0.7)

        ##天体をプロット、初期状態はデータソースが空なので何も表示しない
        plot_for(self.star_sources.values())
        plot_for(self.star_sources.values())

        self.fig.circle("x", "y", source=self.galaxy_source, size=2, color="#FF9966", alpha=0.7)

        self.fig.text("x", "y", "label", source=self.stars_label_sources, text_font_size="5pt", color="white")
        self.fig.text("x", "y", "label", source=self.galaxy_label_sources, text_font_size="6pt", color="white")
        self.fig.text("x", "y", "label", source=self.constellations_name_source, text_font_size="5pt", color="white")
        self.fig.circle("x", "y", source=self.searched_celestial_body_source, alpha=0.4,size=12, color="red",)
        self.fig.multi_line("x","y",source=self.star_line_source, line_width=0.3, color="lightsteelblue")
        ss = self.fig.patches("x", "y", source = self.square_source1, alpha=0.2, color="#3399FF")

        ##カメラの撮影範囲をシミュレートしている枠をマウスクリックで移動できるツールを追加
        self.fig.add_tools(PolyDrawTool(renderers=[ss]))

        ##視等級を調整するスライダーを作成
        self.magnitude_slider = self.make_slider(self.update_stardata, "視等級", 1, -2, 14.5, 0.1)
        self.update_projection()
        self.plot = Column(self.magnitude_slider, self.fig)

    def add_figure_event(self):
        #図にcallback関数を定義
        def callback_LoD(event):
            #図全体の移動・拡大・縮小を有効にする
            self.LoD_flag = not self.LoD_flag
        def callback_wheel(event):
            #ホイール操作による拡大・縮小
            if self.LoD_flag:
                self.center_pos = (event.x, event.y)
                self.renew_mode = "wheel"
                self.update_projection()
        def callback_tap(event):
            #マウスクリックによる検索
            x, y = event.x, event.y
            x_col, y_col = self.col_name()
            try:
                star_label = (np.abs(self.plotting_stars[x_col]-x) + np.abs(self.plotting_stars[y_col]-y)).idxmin()
                nearly_star = self.plotting_stars.loc[star_label]
                star_distance = np.abs(nearly_star[x_col]-x) + np.abs(nearly_star[y_col]-y)
            except:
                star_distance = 100
            try:
                galaxy_label = (np.abs(self.plotting_galaxies[x_col]-x) + np.abs(self.plotting_galaxies[y_col]-y)).idxmin()
                nearly_galaxy = self.plotting_galaxies.loc[galaxy_label]
                galaxy_distance = np.abs(nearly_galaxy[x_col]-x) + np.abs(nearly_galaxy[y_col]-y)
            except:
                galaxy_distance = 100

            if star_distance + galaxy_distance == 200:
                pass
            else:
                try:
                    if star_distance < galaxy_distance:
                        star = self.stars_100[self.stars_100["HIP_num"]== nearly_star["HIP_num"]]
                        self.celestial_text.text = self.star_explain_text(star.iloc[0])
                    else:
                        self.celestial_text.text = self.galaxy_explain_text(nearly_galaxy)
                except:
                    self.celestial_text.text = ""

        def callback_slide_e(event):
            #ドラッグの操作
            if self.LoD_flag:
                ##表示画面の範囲を更新
                self.center_pos = ((self.fig.x_range.end-self.fig.x_range.start)/2+self.fig.x_range.start, (self.fig.y_range.end-self.fig.y_range.start)/2+self.fig.y_range.start)
                self.renew_mode = "slide"
                self.update_projection()
            else:
                if self.AngleSimu.angle_simu_flag:
                    ##カメラの撮影範囲の中心座標を更新
                    self.calcu_angle_center_pos()

        self.fig.on_event(events.LODStart, callback_LoD)
        self.fig.on_event(events.LODEnd, callback_LoD)
        self.fig.on_event(events.MouseWheel, callback_wheel)
        self.fig.on_event(events.Tap, callback_tap)
        self.fig.on_event(events.PanEnd, callback_slide_e)
    
    def col_name(self):
        #投影法によって使うxyの座標が違うので、その場合分けを簡略化
        x_col = self.plotting_mode + "_x"
        y_col = self.plotting_mode + "_y"
        return x_col, y_col


    def star_explain_text(self,star):
        #star(明るい恒星)に対して説明文を作成
        star_name = star["Name_Jpn"]

        star_mag_int = int((star["Magnitude"]-0.5)//1+1)
        if star_mag_int < 1:
           star_mag_int = 1 
        mags = np.sort(self.stars_100["Magnitude"].values)
        star_mag = star["Magnitude"]
        mag_zyuni = np.where(mags==(star_mag))[0][0]+1
        
        text = ""
        text += f'{star_name}<br><br>'
        text += f'{self.trans_constellations_name[star["Constellation"]]}座の{star_mag_int}等星で、'
        text += f'赤経 {(star["RA_Deg"]-ROTATE)%360:.1f}度、赤緯 {star["Dec_Deg"]:.1f}度に位置する。<br>'
        text += f'太陽を除いた恒星の中で{mag_zyuni}番目に明るい。<br>'

        if star_name in ["シリウス", "プロキオン"]:
            text += "冬の大三角と冬のダイヤモンドの恒星のうちのひとつ。"
        if star_name in ["アルデバラン","ボルックス","カペラ", "リゲル"]:
            text += "冬のダイヤモンドの恒星のうちの１つ。"
        if star_name in ["ベテルギウス"]:
            text += "冬の大三角の恒星のうちの１つ。"
        if star_name in ["デネブ", "アルタイル", "ベガ"]:
            text += "夏の大三角の恒星のうちの１つ。"
        return text

    def galaxy_explain_text(self,galaxy):
        #銀河などにたいして説明文を作成
        text = ""
        text += galaxy["messier_num"]

        if pd.isnull(galaxy["NGC_num"]):
            text += "<br><br>"
        else:
            text += f'(NGC{galaxy["NGC_num"]})<br><br>'

        text += f'{galaxy["constellation"]}座の{galaxy["type"]}'
        if pd.isnull(galaxy["name"]):
            text += "。<br>"
        else:
            text += f'で、{galaxy["name"]}と呼ばれる。<br>'

        text += f'地球から約{galaxy["distance"]}千光年離れた場所に存在する。<br>'

        def calcu_focus_distance(viewing_angle, sensor_size=(36.0, 24.0)):
            #視野角から焦点距離を計算
            sensor_v, sensor_h = sensor_size
            viewing_v, viewing_h = float(max(viewing_angle)), float(min(viewing_angle))
            f_v = sensor_v / (2 * np.tan(viewing_v*np.pi/21600))
            f_h = sensor_h / (2 * np.tan(viewing_h*np.pi/21600))
            return int(min(f_v, f_h))
        square = galaxy["angle"].split(",")
        if len(square) == 2:
            text += f'視野角は{square[0]}×{square[1]}分で、焦点距離{calcu_focus_distance(square)}mmで撮影すると天体が画面全体に映る 。'
        else:
            square = [square[0], square[0]]
            text += f'視野角は{square[0]}×{square[0]}分で、焦点距離{calcu_focus_distance(square)}mmで撮影すると天体が画面全体に映る 。'

        return text


  
    def update_stardata(self, attr, old, new):
        #self.magnitude_slider(視等級のスライダー)を更新したときの表示の更新
        if self.plotting_mode == "mercator":
            self.update_mercator_stardata()
            self.plotting_mode_old = "mercator"
        elif self.plotting_mode == "gnomonic":
            self.renew_mode = "slider"
            self.calcu_gnomonic_center()
            self.update_gnomonic_stardata()
            self.plotting_mode_old = "gnomonic"
        self.update_starlabel()


    def update_mercator_stardata(self):
        #メルカトル図法の時の恒星データの更新
        stars_mag = self.stars["Magnitude"] # -1.44 ~14.08
        magnitude = self.magnitude_slider.value
        self.plotting_mode_old = "mercator"

        self.plotting_stars, self.star_sources = self.update_plotting_stars_sources(magnitude, self.stars, stars_mag)

    def update_gnomonic_stardata(self):
        #心射方位図法の時の恒星データの更新
        ##投影する範囲内(中心の星から一定の赤経赤緯内)の星を列挙
        stars = self.select_gnomonic_celestial_bodies(self.stars, self.gnomonic_center)
        ##心射方位図法のxy座標を計算
        stars = self.calcu_gnomonic_pos(stars, self.gnomonic_center)

        stars_mag = stars["Magnitude"] # -1.44 ~14.08
        magnitude = self.magnitude_slider.value

        self.plotting_stars, self.star_sources = self.update_plotting_stars_sources(magnitude, stars, stars_mag)


    def calcu_gnomonic_center(self):
        #心射方位図法の中心となる場所の赤経赤緯を計算
        if self.plotting_mode_old == "mercator":
            ##メルカトル図法からの切り替えで心射方位図法になった場合
            (x,y) = self.center_pos
            center_RA_Deg = inv_Mercator_x(x)
            center_Dec_Deg = inv_Mercator_y(y)
        else:
            if self.renew_mode == "slider":
                ##self.magnitude_sliderを動かした場合
                try:
                    ##以前に中心を計算したことがある場合はそれを利用
                    (x,y,center_RA_Deg,center_Dec_Deg) = self.gnomonic_center
                except:
                    ##そうでない場合には最も近い星を中心にする
                    (x,y) = self.center_pos
                    label = (np.abs(self.plotting_stars["gnomonic_x"]-x) + np.abs(self.plotting_stars["gnomonic_y"]-y)).idxmin()
                    nearly_star = self.plotting_stars.loc[label]

                    center_RA_Deg = nearly_star["RA_Deg"]
                    center_Dec_Deg = nearly_star["Dec_Deg"]
                
            elif self.renew_mode == "wheel":
                ##拡大縮小のときは中心は変化させない
                return None
            elif self.renew_mode == "slide":
                ##ドラッグに酔って図の表示範囲が変わった場合
                (x,y) = self.center_pos #画面の中心
                (old_x,old_y,old_center_RA_Deg,old_center_Dec_Deg) = self.gnomonic_center
                center_RA_Deg = old_center_RA_Deg - np.arctan(x-old_x) * 180 / np.pi
                center_Dec_Deg = old_center_Dec_Deg + np.arctan(y-old_y) * 180 / np.pi

                
            elif self.renew_mode == "search":
                ##検索によって表示の中心が移動した場合
                (x,y) = self.center_pos
                center_RA_Deg = (self.center_star["RA_Deg"]).iat[0]
                center_Dec_Deg = self.center_star["Dec_Deg"].iat[0]
            else:
                raise ValueError
        self.gnomonic_center = (x, y, center_RA_Deg, center_Dec_Deg)
    
    def calcu_gnomonic_pos(self,celestial_bodies, gnomonic_center):
        ##celestial_bodies::pd.DataFrameに対して心射方位図法のxy座標を計算しれ列を追加
        (x, y, center_RA_Deg, center_Dec_Deg) = gnomonic_center
        gnomonic_x = x-np.tan((celestial_bodies["RA_Deg"] - center_RA_Deg)*np.pi/180)
        gnomonic_y = y+np.tan((celestial_bodies["Dec_Deg"] - center_Dec_Deg)*np.pi/180)
        gnomonic_x.name = "gnomonic_x"
        gnomonic_y.name = "gnomonic_y"
        celestial_bodies = pd.concat([celestial_bodies, gnomonic_x, gnomonic_y],axis=1)
        return celestial_bodies

    def select_gnomonic_celestial_bodies(self, celestial_bodies, gnomonic_center, diff_range=45):
        #天体のDataFrameと中心の赤経赤緯にたいしてdiff_range内の天体に絞ったDataFrameを返す
        (x, y, center_RA_Deg, center_Dec_Deg) = gnomonic_center
        RA_Deg, Dec_Deg = celestial_bodies["RA_Deg"], celestial_bodies["Dec_Deg"]
        selected_celestial_bodies = celestial_bodies[(center_RA_Deg-diff_range < RA_Deg)&(RA_Deg < center_RA_Deg+diff_range)&(center_Dec_Deg-diff_range < Dec_Deg)&(Dec_Deg < center_Dec_Deg+diff_range)]
        return selected_celestial_bodies

    def update_plotting_stars_sources(self, magnitude, stars, stars_mag):
        #投影する恒星を更新
        plotting_stars = pd.DataFrame([], columns=self.stars.columns)
        sources = self.default_source.copy()
        x_col, y_col = self.col_name()

        ##視等級別にデータソースを更新
        for i in range(-2, int(magnitude)+1):
            range_in_stars = stars[(i-0.5 <= stars_mag ) & (stars_mag < i+0.5)]
            plotting_stars = pd.concat([plotting_stars, range_in_stars])
            sources[i].data = dict(x=range_in_stars[x_col], y=range_in_stars[y_col])
        for i in range(int(magnitude)+1, 16):
            sources[i].data = dict(x=[], y=[])
        range_in_stars = stars[(int(magnitude)+0.5 <= stars_mag) & (stars_mag < magnitude+0.5)]
        plotting_stars = pd.concat([plotting_stars, range_in_stars])
        sources[int(magnitude)+1].data = dict(x=range_in_stars[x_col], y=range_in_stars[y_col])
        return plotting_stars, sources



    def update_starlabel(self):
        #恒星の名前を更新
        if self.starslabel_flag:
            plotting_stars_named = pd.merge(self.plotting_stars, self.stars_100[["HIP_num", "Name_Jpn"]], on="HIP_num",how="inner")
            x_col, y_col = self.col_name()
            self.stars_label_sources.data = dict(x=plotting_stars_named[x_col], y=plotting_stars_named[y_col], label=plotting_stars_named["Name_Jpn"])
        else:
            self.stars_label_sources.data = dict(x=[], y=[], label=[])

    def update_projection(self):
        #拡大縮小やスライドが起きたときの全データの更新
        left, right = self.fig.x_range.start, self.fig.x_range.end
        plot_range = right - left

        if plot_range >= 2:
            ##メルカトル図法で投影
            self.plotting_mode = "mercator"
            self.plotting_mode_text.text="投影図法：メルカトル"
            self.update_mercator_stardata()
            self.plotting_mode_old = "mercator"
        else:
            ##心射方位図法で投影
            if self.plotting_linkage_flag:
                ##表示範囲に応じて星の数を調整
                self.magnitude_slider.value = min(np.exp(-plot_range+2), 14.5)
            
            self.plotting_mode = "gnomonic"
            self.plotting_mode_text.text="投影図法：心射方位"
            self.calcu_gnomonic_center()
            self.update_gnomonic_stardata()
            self.plotting_mode_old = "gnomonic"
        self.update_starlabel()
        self.update_star_line()
        self.update_galaxies_and_label()
        self.update_constellation_label()
        self.chase_searched_star()
        



    def chase_searched_star(self):
        #拡大縮小やスライドが起きたときに検索にヒットしている天体を追う
        if self.matched_celestial_body is not None:
            x_col, y_col = self.col_name()
            if self.plotting_mode == "gnomonic":
                try:
                    ##過去に心射方位図法の座標を計算していると列名が競合するので削除
                    self.matched_celestial_body = self.matched_celestial_body.drop(["gnomonic_x","gnomonic_y"], axis=1)
                except:
                    pass  
                self.matched_celestial_body = self.calcu_gnomonic_pos(self.matched_celestial_body, self.gnomonic_center)
            self.searched_celestial_body_source.data = dict(x=self.matched_celestial_body[x_col], y=self.matched_celestial_body[y_col])

    def update_galaxies_and_label(self):
        #銀河全般の更新
        '''
        0 : 何も表示しない
        1 : 銀河の点だけ
        2 : メシエ番号も
        '''
        def upadate_galaxies():
            #点の更新
            x_col, y_col = self.col_name()
            if self.plotting_mode == "mercator":
                galaxies = self.galaxy
            else:
                self.calcu_gnomonic_center()
                galaxies = self.select_gnomonic_celestial_bodies(self.galaxy, self.gnomonic_center)
                galaxies = self.calcu_gnomonic_pos(galaxies, self.gnomonic_center)

            self.galaxy_source.data = dict(x=galaxies[x_col], y=galaxies[y_col])
            self.plotting_galaxies = galaxies

        def update_galaxylabel():
            #メシエ番号の更新
            if self.galaxy_flag == 2:
                x_col, y_col = self.col_name()
                self.galaxy_label_sources.data = dict(x=self.plotting_galaxies[x_col], y=self.plotting_galaxies[y_col], label=self.plotting_galaxies["messier_num"])
            else:
                self.galaxy_label_sources.data = dict(x=[], y=[], label=[])
                    
        if self.galaxy_flag:
            #1,2
            upadate_galaxies()
        else:
            #0
            self.galaxy_source.data = dict(x=[], y=[])
            self.plotting_galaxies = pd.DataFrame([], columns=self.galaxy.columns)
        update_galaxylabel()



    def update_star_line(self):
        #星座線の更新
        def update_gnomonic_star_line(gnomonic_center):
            ##心射方位図法の時は複雑なので関数化
            diff_range=60
            (x, y, center_RA_Deg, center_Dec_Deg) = gnomonic_center
            RA_Deg1, Dec_Deg1, RA_Deg2, Dec_Deg2 = self.stars_line["RA_Deg_1"], self.stars_line["Dec_Deg_1"], self.stars_line["RA_Deg_2"], self.stars_line["Dec_Deg_2"]
            ##中心近くの星座線を抽出
            Deg1 = (center_RA_Deg-diff_range < RA_Deg1)&(RA_Deg1 < center_RA_Deg+diff_range)&(center_Dec_Deg-diff_range < Dec_Deg1)&(Dec_Deg1 < center_Dec_Deg+diff_range)
            Deg2 = (center_RA_Deg-diff_range < RA_Deg2)&(RA_Deg2 < center_RA_Deg+diff_range)&(center_Dec_Deg-diff_range < Dec_Deg2)&(Dec_Deg2 < center_Dec_Deg+diff_range)
            stars_line = self.stars_line[Deg1 & Deg2]

            #星座線のそれぞれの頂点の座標を計算
            stars_line1 = stars_line[["RA_Deg_1", "Dec_Deg_1"]]
            stars_line1.columns = ["RA_Deg", "Dec_Deg"]
            vertex1 = self.calcu_gnomonic_pos(stars_line1, gnomonic_center)

            stars_line2 = stars_line[["RA_Deg_2", "Dec_Deg_2"]]
            stars_line2.columns = ["RA_Deg", "Dec_Deg"]
            vertex2 = self.calcu_gnomonic_pos(stars_line2, gnomonic_center)

            gnomonic_lines_x = [[i1,i2] for i1,i2 in zip(vertex1.gnomonic_x.values, vertex2.gnomonic_x.values)]
            gnomonic_lines_y = [[i1,i2] for i1,i2 in zip(vertex1.gnomonic_y.values, vertex2.gnomonic_y.values)]
            
            self.star_line_source.data = dict(x=gnomonic_lines_x, y=gnomonic_lines_y)

        if self.star_line_flag:
            if self.plotting_mode == "mercator":
                self.star_line_source.data = dict(x=self.mercator_lines_x, y=self.mercator_lines_y)
            else:
                update_gnomonic_star_line(self.gnomonic_center)
        else:
            self.star_line_source.data = dict(x=[], y=[])

    def update_constellation_label(self):
        #星座名を更新
        if self.constellation_flag:
            x_col,y_col = self.col_name()
            constellations_pos = self.constellations_pos
            if self.plotting_mode == "gnomonic":
                constellations_pos = self.calcu_gnomonic_pos(constellations_pos, self.gnomonic_center)
            self.constellations_name_source.data = dict(x=constellations_pos[x_col], y=constellations_pos[y_col], label=constellations_pos["Conste_Jpn"])
        else:
            self.constellations_name_source.data = dict(x=[], y=[], label=[])



    def find_match_celestial_body(self, search_word):
        #文字列にあう天体と、それがどの種類の天体かの組みを返す
        matched_star = self.stars_100[search_word == self.stars_100["Name_Jpn"]]
        if len(matched_star.index) != 0:
            return matched_star, "star"
        matched_galaxy_num = self.galaxy[search_word == self.galaxy["messier_num"]]
        if len(matched_galaxy_num.index) != 0:
            return matched_galaxy_num, "galaxy_num"
        matched_galaxy_name = self.galaxy[search_word == self.galaxy["name"]]
        if len(matched_galaxy_name.index) != 0:
            return matched_galaxy_name, "galaxy_name"
        matched_constellation = self.constellations_pos[search_word == self.constellations_pos["Conste_Jpn"]]
        if len(matched_constellation) != 0:
            return matched_constellation, "constellation"
        return None, ""

    
    def celestial_bodies_search_handler(self,attr, old, new):
        #検索ボタンがおされたら検索Boxの文字列に合う天体を中心にして図を更新
        self.searched_celestial_body_source.data = dict(x=[], y=[])
        search_word = self.searchbox.value
        self.matched_celestial_body, mode = self.find_match_celestial_body(search_word)
        if self.matched_celestial_body is None:
            ##検索に合う天体が存在しない場合
            print("No match object")
            return None
        
        ##検索に合う天体を表示
        matched_celestial_body = self.matched_celestial_body.iloc[0]
        self.searched_celestial_body_source.data = dict(x=self.matched_celestial_body["mercator_x"], y=self.matched_celestial_body["mercator_y"])

        ##検索した天体が生じされていなかった場合表示する
        if mode == "star":
            self.celestial_text.text = self.star_explain_text(matched_celestial_body)
            self.magnitude_slider.value = max(self.magnitude_slider.value, int(matched_celestial_body["Magnitude"])+1)
            self.starslabel_flag = True
        if "galaxy" in mode:
            self.celestial_text.text = self.galaxy_explain_text(matched_celestial_body)
            self.galaxy_flag = 2
        if mode == "constellation":
            self.star_line_flag = True

        ##図の中心を天体にして更新
        self.center_pos = (matched_celestial_body["mercator_x"], matched_celestial_body["mercator_y"])
        self.set_fig_range(self.center_pos)

        ##その中心に合わせた各種データの更新
        if self.plotting_mode == "mercator":
            pass
        else:
            self.gnomonic_center = (matched_celestial_body["mercator_x"], matched_celestial_body["mercator_y"], matched_celestial_body["RA_Deg"], matched_celestial_body["Dec_Deg"])
            self.update_gnomonic_stardata()

        self.update_starlabel()
        self.update_star_line()
        self.update_galaxies_and_label()



    def set_fig_range(self,center_pos):
        #表示する範囲の広さは変えずに、center_pos中心に図の表示範囲を更新
        (center_x, center_y) = center_pos
        xs = self.fig.x_range.start
        xe = self.fig.x_range.end
        ys = self.fig.y_range.start
        ye = self.fig.y_range.end
        x_range = xe - xs
        y_range = ye - ys

        self.fig.x_range.start = center_x - x_range /2
        self.fig.x_range.end = center_x + x_range /2
        self.fig.y_range.start = center_y - y_range /2
        self.fig.y_range.end = center_y + y_range /2
        self.fig.x_range.reset_start
        self.fig.x_range.reset_end
        self.fig.y_range.reset_start
        self.fig.y_range.reset_end

    def calcu_angle_center_pos(self):
        #カメラのシミュレートをする枠の中心座標を計算(ドラッグしたとき図全体を動かさない場合)
        xs = self.square_source1.data["x"][0]
        ys = self.square_source1.data["y"][0]
        self.AngleSimu.center_x = (xs[0] + xs[2]) /2
        self.AngleSimu.center_y = (ys[0] + ys[2]) /2


    def update_angle_square_enlarge(self, attr, old, new):
        #焦点距離を設定するスライダーの値を更新した場合に枠の座標を計算
        self.AngleSimu.focus_dis = self.angle_simu_enlarge_slider.value
        self.AngleSimu.calcu_vertex()
        if self.AngleSimu.angle_simu_flag:
            self.square_source1.data = dict(x=self.AngleSimu.xs, y=self.AngleSimu.ys)
    
    def update_angle_square_rotate(self, attr, old, new):
        #枠の回転を設定するスライダーの値を更新した場合に枠の座標を計算
        self.AngleSimu.rot = self.angle_simu_rotate_slider.value
        self.AngleSimu.calcu_vertex()
        if self.AngleSimu.angle_simu_flag:
            self.square_source1.data = dict(x=self.AngleSimu.xs, y=self.AngleSimu.ys)

        
    def make_button(self, callback_func, label, default_size=300, button_type="default", background=(100, 100, 255)):
        #ボタンを作る関数
        button=Button(label=label, default_size=default_size, button_type=button_type, background=background)
        button.on_click(callback_func)
        return button
    def make_slider(self, callback_func, title, init_val, start_val, end_val, step):
        #スライダーを作る関数
        slider = Slider(title=title, value=init_val, start=start_val, end=end_val, step=step)
        slider.on_change('value', callback_func)
        return slider
    def exit_hander(self):
        #終了ボタン
        sys.exit()
    def make_search_input_textbox(self):
        #検索Boxの作成
        searchbox = TextInput(value="", placeholder="天体の検索")
        return searchbox
    def switch_starlabel_hander(self):
        #恒星の名前の表示を切り替え
        self.starslabel_flag = not self.starslabel_flag
        if self.starslabel_flag:
            self.star_label_button.button_type = "primary"
        else:
            self.star_label_button.button_type = "default"
        self.update_starlabel()
    def switch_angle_simulator_handler(self):
        #カメラのシミュレーションの枠の表示の切り替え
        self.AngleSimu.angle_simu_flag = not self.AngleSimu.angle_simu_flag
        if self.AngleSimu.angle_simu_flag:
            self.angle_simu_button.button_type = "primary"
            print(self.fig.center[0])
            print(type(self.fig.center[0]))
            self.AngleSimu.center_x = (self.fig.x_range.start + self.fig.x_range.end) /2
            self.AngleSimu.center_y = (self.fig.y_range.start + self.fig.y_range.end) /2
            self.AngleSimu.calcu_vertex()
            self.square_source1.data= dict(x=self.AngleSimu.xs, y=self.AngleSimu.ys)
        else:
            self.angle_simu_button.button_type = "default"
            self.square_source1.data = dict(x=[],y=[])
    def switch_galaxy_handler(self):
        #銀河の表示を切り替え
        self.galaxy_flag = (self.galaxy_flag +1)%3
        if self.galaxy_flag==0:
            self.galaxy_button.button_type = "default"
        elif self.galaxy_flag==1:
            self.galaxy_button.button_type = "success"
        else:
            self.galaxy_button.button_type = "primary"
        self.update_galaxies_and_label()
    def switch_star_line_handler(self):
        #星座線の表示を切り替え
        self.star_line_flag = not self.star_line_flag
        if self.star_line_flag:
            self.star_line_button.button_type = "primary"
        else:
            self.star_line_button.button_type = "default"
        self.update_star_line()
    def switch_constellation_handler(self):
        #星座名の表示を切り替え
        self.constellation_flag = not self.constellation_flag
        if self.constellation_flag:
            self.constellation_button.button_type = "primary"
        else:
            self.constellation_button.button_type = "default"
        self.update_constellation_label()

    def switch_plotting_linkage_handler(self):
        #拡大縮小の時にself.magnitude_slider(視等級のスライダー)
        self.plotting_linkage_flag = not self.plotting_linkage_flag
        if self.plotting_linkage_flag:
            self.plotting_linkage_button.button_type = "primary"
        else:
            self.plotting_linkage_button.button_type = "default"


    def make_tools(self):
        #ボタンやスライダーを作成
        end = self.make_button(self.exit_hander, "終了")

        self.searchbox= self.make_search_input_textbox()
        self.searchbox.on_change("value", self.celestial_bodies_search_handler)

        self.celestial_text = Div(text="",width=300, height=220)
        self.celestial_text.background = (10,10,10,0.05)

        self.plotting_mode_text= Div(text="投影図法：メルカトル",width=140, height=20)
        self.plotting_mode_text.background = (10,10,10,0.05)
        self.plotting_linkage_button = self.make_button(self.switch_plotting_linkage_handler, "視等級とズームの連動", default_size=100)
        self.plotting_linkage_button.button_type = "primary"

        self.angle_simu_enlarge_slider = self.make_slider(self.update_angle_square_enlarge, "焦点距離", 35, 10, 200,1)
        self.angle_simu_rotate_slider  = self.make_slider(self.update_angle_square_rotate,  "回転角度", 0, -360, 360, 1)
        self.angle_simu_button = self.make_button(self.switch_angle_simulator_handler, "撮影シミュレーション") 
        angle_simu = Column(self.angle_simu_button, self.angle_simu_enlarge_slider, self.angle_simu_rotate_slider)
        angle_simu.background = (0, 100, 60,0.2)
        
        self.star_label_button = self.make_button(self.switch_starlabel_hander, "恒星の名前：on/off", button_type="primary")
        self.galaxy_button = self.make_button(self.switch_galaxy_handler, "銀河：off/点のみ/メシエ番号")
        self.star_line_button = self.make_button(self.switch_star_line_handler, "星座線：on/off")
        self.constellation_button = self.make_button(self.switch_constellation_handler, "星座の名前：on/off")
        switches = Column(self.star_label_button, self.galaxy_button, self.star_line_button, self.constellation_button)
        switches.background = (0, 40, 100, 0.2)

        tools = Column(end, self.searchbox, self.celestial_text, Row(self.plotting_mode_text, self.plotting_linkage_button), angle_simu, switches )

        return tools



Viewer = StarViewer()

doc = curdoc()
tools = Viewer.make_tools()
layout = Row(Viewer.plot, tools)
doc.add_root(layout)

print("initalized...")