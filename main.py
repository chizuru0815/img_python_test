import os
import cv2
​
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
​
import numpy as np
import time
import pandas as pd
import itertools
​
import datetime
from datetime import timedelta
​
import re
import shutil
import zipfile
from logging import getLogger,config
from settings.logging import LOGGING
​
config.dictConfig(config=LOGGING)
logger = getLogger('functions')
execute_status='with errors'
​
from settings.path_settings import DAWNLOAD_ROOT, OUTPUT_ROOT
​
​
##parameter
DOWNLOAD_PATH = os.path.join(DAWNLOAD_ROOT)
OUT_DIR_PATH = os.path.join(OUTPUT_ROOT)
​

def chromeopt():
    try:
        ###Chromeへオプションを設定
        chromeOptions = webdriver.ChromeOptions()
        prefs = {
            "download.default_directory": os.path.join(DAWNLOAD_ROOT)}
        chromeOptions.add_experimental_option("prefs", prefs)
        chromedriver = ChromeDriverManager().install()
        driver = webdriver.Chrome(executable_path=chromedriver, chrome_options=chromeOptions)
​
    except Exception as e:
        raise e
    else:
        return driver
​
​
def download_complete():
    # ファイルダウンロードしたか確認する　True:ダウンロード完了、False:ダウンロード未完了
    # (01_downloadフォルダのファイルの拡張子がcrdownloadの場合はFalseを返す)
    try:
        for file in os.listdir(DOWNLOAD_PATH):
            file_path = os.path.join(DOWNLOAD_PATH, file)
            if os.path.splitext(file)[1] == '.crdownload':
                logger.info('ダウンロード中です {}'.format(file))
                return False
​
        return True
​
    except Exception as e:
        logger.exception(e)
        raise e
    else:
        pass
​

def jepx_spot_curve_download(yyyymm):
    try:
        yyyymm = [yyyymm[ym].strftime("%Y%m") for ym in range(len(yyyymm))]
        driver = chromeopt()
        # JEPXサイトにアクセス
        target_url = 'http://www.jepx.org/market/index.html'
        driver.get(target_url)
​
        time.sleep(7)
​
        # 検索結果の各リンクをelem_urlに各々リスト型として保存して各々のリンクを1行ずつprintで表示
        elem_url = []
        elems = driver.find_element_by_class_name('arwList').find_elements_by_tag_name('a')
        for elem in elems:
            elem_url.append(elem.get_attribute("href"))
        print(elem_url)
​
        filename_zip=[]
​
        for el in range(len(elem_url)):
            if re.split('//|/', elem_url[el])[4].split('.')[0].split('_')[1] in yyyymm:
                ## ファイルのリンクをクリック
                driver.find_element_by_xpath('//*[@id="curveFile{}"]'.format(el + 1)).click()
                filename_zip.append('curve_' + re.split('//|/', elem_url[el])[4].split('.')[0].split('_')[1] + '.zip')
                # ダウンロードに1分程かかる
                time.sleep(60)
                # com_ope.download_complete()
                # ダウンロード完了確認
                dwld_comp_cnt = 0
                while download_complete() == False:
                    time.sleep(5)
                    dwld_comp_cnt += 1
                    if dwld_comp_cnt > 60:
                        err_msg = '{}秒経過しましたがダウンロード完了しないため処理中断します'.format(dwld_comp_cnt)
                        logger.error(err_msg)
                        text = 'ダウンロードの処理途中でエラー発生しました。\n'
                        text += 'エラーメッセージ\n'
                        text += err_msg
                        # send_alert_mail('異常終了', text, [], ':ng:')
                        raise Exception(err_msg)
​
        driver.quit()
​
    except Exception as e:
        raise e
    else:
        return filename_zip

# グラフの最頻度（各RGB合計値）
def get_rgb_info(df_all, img, yyyymmdd):
    try:
        logger.info('グラフの最頻度（各RGB合計値）を取得開始')
        # 各RGB合計値の頻度を出力、前三位　白　青　赤　と考え、確認することが必要
        list_ = []
        for i in df_all.columns:
            list_.extend(df_all[i].tolist())
​
        # 画像から取得した各pxのRGB合計値のユニークリスト
        unique_list = pd.unique(list_)
        color_list = []
        counts=[]
​
        # 各RGB合計値の頻度
        for x in unique_list:
            color_list.append(x)
            counts.append(list_.count(x))
​
        df_color = pd.DataFrame({'color':color_list, 'counts':counts})
        # 頻度が多い順にソート
        df_color = df_color.sort_values('counts',ascending=False).reset_index(drop=True)
        logger.info('各RGB合計値の頻度')
​
        # 縦横軸：出現回数の多い列を検索したい
        df_all_count = df_all.apply(pd.value_counts)
​
        # 背景色は必ず一番多い
        background_rgb_sum = df_color['color'][0]
​
        if df_color['color'][1] - df_color['color'][2] in [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]:
            # グラフの縦データの色（この場合は、2色）
            data_rgb_sum = [df_color['color'][1], df_color['color'][2]]
            # グラフの縦横軸の色
            line_rgb_sum = df_color['color'][3]
        else:
            # グラフの縦データの色
            data_rgb_sum = [df_color['color'][1]]
            # グラフの縦横軸の色
            line_rgb_sum = df_color['color'][2]
​
​
        # 各RGB合計値の頻度出力データから、対象のRGB合計値、index名、列名を取得する
        # 縦軸の対象行：index名を取得
        rgb_line_index = df_all_count.index.get_loc(line_rgb_sum)
        # 縦軸の対象行：値をソート（最頻値が一番上）
        df_all_counts = df_all_count.iloc[rgb_line_index].sort_values(ascending=False)
        # 縦軸の対象列番号
        line_column_start = df_all_counts.index.values.tolist()[0]
​
​
        # 縦軸のメモリ（pxリスト）
        line_index_ruler = df_all[line_column_start - 1].tolist()
        index_rulers = [index for index in range(len(line_index_ruler))
                        if line_index_ruler[index] != background_rgb_sum]
​
        # 買い（青線）のメモリの価格リスト
        if yyyymmdd >= '20220601':
            y_price = [price * 10 for price in reversed(range(len(index_rulers)))]
        else:
            # 2022/5/31までは、縦軸の買い価格もメモリは0～1000であった。
            y_price = [price * 100 for price in reversed(range(len(index_rulers)))]
​
        df_info_y = pd.DataFrame({'price': y_price, 'index': index_rulers})
​
        # 横軸のメモリ（pxリスト）　縦軸の最後が横軸
        line_column_ruler = df_all.iloc[index_rulers[-1] + 1,:].tolist()
        column_rulers = [column for column in range(len(line_column_ruler))
                         if line_column_ruler[column] != background_rgb_sum and column >= line_column_start]
        column_rulers.insert(0, line_column_start)
        # 横軸のメモリの価格リスト
        x_price = [price * 1000 for price in range(len(column_rulers))]
​
        df_info_x = pd.DataFrame({'price': x_price, 'columns': column_rulers})
​
        # # 各RGB合計値の頻度出力データから、対象のRGB取得
        rgb_blue_serch = df_all.iloc[index_rulers[0], :].tolist()
        serch_rulers = [column for column in range(len(rgb_blue_serch))
                         if rgb_blue_serch[column] != background_rgb_sum and column > line_column_start]
        for rgb in range(len(serch_rulers)):
            if all(img[index_rulers[0]][serch_rulers[rgb]] == np.array([255, 0, 0])):
                rgb_red = img[index_rulers[0]][serch_rulers[rgb]]
                break
            elif all(img[index_rulers[0]][serch_rulers[rgb]] == np.array([0, 0, 254])) | \
                    all(img[index_rulers[0]][serch_rulers[rgb]] == np.array([0, 0, 255])):
                rgb_blue = img[index_rulers[0]][serch_rulers[rgb]]
            else:
                pass
​
    except Exception as e:
        raise e
    else:
        return background_rgb_sum, data_rgb_sum, line_rgb_sum, line_column_start, df_info_y, df_info_x, rgb_red, rgb_blue


# スポット入札需給曲線（青線）取得座標から量の計算
def get_data_sell(df_all, img, df_info_y, df_info_x, get_date, coma_time, background_rgb_sum,line_column_start, rgb_red, rgb_blue):
    try:
        logger.info('スポット入札需給曲線（青線）、取得座標から値の計算開始')
        df_data = pd.DataFrame()
        data_list = []
​
        df_info_y['index'][len(df_info_y) - 1] = df_info_y['index'][len(df_info_y) - 1] - 1
        sell_price_data = ['売', 9999]
​
        for price,index in df_info_y.iterrows():
            print(str(index[0]) + '円')
            for price2,columns in df_info_x.iterrows():
                # # グラフの上の座標を取得（index_rulers[0]行のどこかであるはず）※一番下の座標は、index_rulers[-1]行のどこかである
                # price_data = [売買, 価格,　開始, 終了]
                buy_price_data = ['買', index[0]]
​
                target_px = df_all.iloc[index[1], :].tolist()
                line_datas = [c for c in range(len(target_px))
                              if target_px[c] != background_rgb_sum and c > line_column_start]
​
                continue_list = [list(g) for _, g in itertools.groupby(line_datas, key=lambda n, c=itertools.count(): n - next(c))]
                check_blue = rgb_blue
                check_red = rgb_red
​
                buy_price_xy = []
                sell_price_xy = []
                for l in range(len(continue_list)):
                    for m in range(len(continue_list[l])):
                        if all(check_blue == img[index[1]][continue_list[l][m]]) == True:
                            buy_price_xy.append(continue_list[l][m])
                            # print('青')
                        elif all(check_red == img[index[1]][continue_list[l][m]]) == True:
                            sell_price_xy.append(continue_list[l][m])
                            # print('赤')
                        else:
                            pass
​
                buy_price_xy = [buy_price_xy[0], buy_price_xy[-1]]
                if price == 0:
                    sell_price_xy = [sell_price_xy[-1]]
​
​
                for buy in range(len(buy_price_xy)):
                    for p,c in df_info_x.iterrows():
                        p_col = np.abs(np.asarray(df_info_x['columns']) - buy_price_xy[buy]).argmin()
                        xy = df_info_x['columns'][p_col]
​
                        # 価格四捨五入、10円単位は切り捨て。
                        if xy > buy_price_xy[buy]:
                            # 座標がメモリのほうが大きい＝メモリから離れている分だけ価格をマイナス
                            price_xy = df_info_x['price'][p_col] - int(round(1000 / (xy - df_info_x['columns'][p_col - 1]) * (xy - buy_price_xy[buy]), -2))
                            buy_price_data.append(price_xy)
                            break
​
                        elif xy < buy_price_xy[buy]:
                            # 座標がメモリのほうが小さい＝メモリから離れている分だけ価格をプラス
                            price_xy = df_info_x['price'][p_col] - int(round(1000 / (xy - df_info_x['columns'][p_col + 1]) * (buy_price_xy[buy] - xy), -2))
                            buy_price_data.append(price_xy)
                            break
​
                        else:
                            # 座標が同じ＝価格もそのまま
                            buy_price_data.append(df_info_x['price'][p_col])
                            break
                data_list.append(buy_price_data)
                # break
​
                if price == 0:
                    # 売り
                    for sell in range(len(sell_price_xy)):
                        for p,c in df_info_x.iterrows():
                            p_col = np.abs(np.asarray(df_info_x['columns']) - sell_price_xy[sell]).argmin()
                            xy = df_info_x['columns'][p_col]
​
                            # 価格四捨五入、10円単位は切り捨て。
                            if xy > sell_price_xy[sell]:
                                # 座標がメモリのほうが大きい＝メモリから離れている分だけ価格をマイナス
                                price_xy = df_info_x['price'][p_col] - int(round(1000 / (xy - df_info_x['columns'][p_col - 1]) * (xy - sell_price_xy[sell]), -2))
                                sell_price_data.append(price_xy)
                                sell_price_data.append('')
                                break
​
                            elif xy < sell_price_xy[sell]:
                                # 座標がメモリのほうが小さい＝メモリから離れている分だけ価格をプラス
                                price_xy = df_info_x['price'][p_col] - int(round(1000 / (xy - df_info_x['columns'][p_col + 1]) * (sell_price_xy[sell] - xy), -2))
                                sell_price_data.append(price_xy)
                                sell_price_data.append('')
                                break
​
                            else:
                                # 座標が同じ＝価格もそのまま
                                sell_price_data.append(df_info_x['price'][p_col])
                                sell_price_data.append('')
                                break
​
                break
​
        data_list.append(sell_price_data)
        df_data = pd.DataFrame(data_list, columns=['売買', '価格', '開始', '終了'])
        df_data['日付'] = get_date
        df_data['時刻'] = coma_time
        df_data = df_data.reindex(columns=['日付','時刻','売買', '価格', '開始', '終了'])
        df_data = df_data.rename(columns={'日付':0,'時刻':1,'売買':2, '価格':3, '開始':4, '終了':5})
​
    except Exception as e:
        raise e
    else:
        return df_data


def main():
    try:
        # サイトからデータを取得(12時以降に取得すると翌日分の画像もzipに格納されている)
        today = datetime.datetime.now()
        target_start = today + timedelta(days=1)
        target_end = ''
       
        if target_end == '':
            target_end = target_start
        day_list = pd.date_range(start=target_start.strftime('%Y-%m-%d'), end=target_end.strftime('%Y-%m-%d'),
                                 freq="D").to_list()
​
        if target_start.month == target_end.month:
            month_list = [target_start]
        else:
            month_list = [target_start, target_end]
​
​
        # 実行開始時にzipファイルと解凍したフォルダがあれば削除しておく
        # 前回取得したzipと画像ファイルを削除
        if os.listdir(DOWNLOAD_PATH):
            shutil.rmtree(DOWNLOAD_PATH)
            os.mkdir(DOWNLOAD_PATH)
​
        # # JEPXサイトからデータをダウンロード
        filename_zip = jepx_spot_curve_download(month_list)
        # print(filename_zip)
        # filename_zip = ['curve_202206.zip']
​
        # zipファイルをs3に格納
        img_spot_s3 = s3.s3_concat('pps')
        # # スポット入札需給曲線へ投入upload_s3(loacl_dir, s3_dir, file_path)
        for zip in filename_zip:
            img_spot_s3.upload_s3(DOWNLOAD_PATH + '/', 'スポット入札需給曲線/zip/', zip)
​
            # # 日付リストごとに画像データを抽出する　#####################################
​
            dirpath = DOWNLOAD_PATH + '/' + zip.split('.')[0]
​
            if os.path.isdir(dirpath) == False:
                # 対象月のzipを解凍
                shutil.unpack_archive(DOWNLOAD_PATH + '/' + zip, extract_dir=dirpath)
​
​
        # コマ設定　#####################################
        coma_flag = 0
​
        if coma_flag == 0:
            # 全48コマを取得
            get_coma = [i for i in range(1, 49)]
            # get_coma = [i for i in range(45, 49)]
        elif coma_flag == 1:
            # 偶数コマを取得
            get_coma = [i for i in range(2, 48 + 2, 2)]
        else:
            # テスト用（手入力等、上記以外）
            get_coma = [17]
            pass
        print(get_coma)
​
​
        data_buy_max_price = []
        data_sell_all_price = pd.DataFrame()
​
        for month in range(len(day_list)):
            # 取得する月
            get_yyyymm = day_list[month].strftime("%Y%m")
            print(get_yyyymm)
​
            img_path = DOWNLOAD_PATH + '/curve_' + get_yyyymm + '/' + day_list[month].strftime("%Y%m%d")
​
            # 対象日のzipを解凍
            with zipfile.ZipFile(img_path + '.zip','r')as f:
                f.extractall(img_path)
​
            logger.info('{} の処理を開始します。'.format(get_yyyymm))
​
            for coma in range(len(get_coma)):
               # ファイル名
                coma_filename = str(get_coma[coma]).zfill(2) + '.png'
                # filename = ['16.png']
                if get_coma[coma] % 2 == 0:
                    # 偶数コマ
                    coma_time = str(int((get_coma[coma]-1)/2)).zfill(2) + ':30'
                    except_flg = 0
                else:
                    # 奇数コマ
                    coma_time = str(get_coma[coma]//2).zfill(2) + ':00'
                    except_flg = 0
​
​
                # 共通処理 ###################################
                # グラフ画像読み込み
                img_file_path = img_path + '/' + coma_filename
                logger.info('取得日：{}、{} コマ({})の画像データを抽出します。'.format(day_list[month].strftime("%Y%m%d"), str(get_coma[coma]), coma_time))
​
                # グラフ画像読み込み
                df_all, i_range, j_range, img \
                    = get_img_data(img_file_path, day_list[month], coma_time)
​
                # 取得したデータからメモリ座標や価格などを設定
                # background_rgb_sum　/ 画像の背景色
                # data_rgb_sum　      / 画像の折れ線の色
                # line_rgb_sum　      / 画像の縦・横軸の色
                # line_column_start　　/ 画像の背景色
                # df_buy_info　      / 買い（縦軸・行）の座標と価格
                # df_sell_info　      / 売り（横軸・列）の座標と価格
​
                get_date = day_list[month].strftime("%Y%m%d")
​
                background_rgb_sum, data_rgb_sum, line_rgb_sum,\
                line_column_start, df_info_y, df_info_x, rgb_red, rgb_blue = get_rgb_info(df_all, img, get_date)
​
                # 買い価格の取得方法
                # 各価格の行：開始位置から最終位置までの列を取得
                # 　　　　　　列の位置により、y軸の価格を算出
​
                # 売り価格の取得方法
                # 各価格の列：各価格列のmin位置（線の角+1 or 2px上）のまでの行を取得
                # 　　　　　　行の位置により、x軸の価格を算出
​
                get_date = day_list[month].strftime("%Y/%m/%d")
                # 取得座標から計算
                # if except_flg == 1:
                data_sell_all = get_data_sell(df_all, img, df_info_y, df_info_x, get_date, coma_time,
                                              background_rgb_sum, line_column_start, rgb_red, rgb_blue)
                data_sell_all_price = pd.concat([data_sell_all_price, data_sell_all]).reset_index(drop=True)
​
​
        # スポット入札需給曲線（青線）のデータ保存
        sell_all_filenames = OUT_DIR_PATH + 'test.xlsx'
        sell_all_data = pd.read_excel(sell_all_filenames, index_col=None, sheet_name='Sheet1', header=None)
        df_sell_spot_all = pd.concat([sell_all_data, data_sell_all_price]).reset_index(drop=True)
        df_sell_spot_all.to_excel(sell_all_filenames, sheet_name='Sheet1', index=False, header=False)
​
​
    except Exception as e:
        raise e
    else:
        pass

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
