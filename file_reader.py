import pandas as pd
import datetime

def load_raw_fish_data():

    path_dir = './fish_data/freshwater-fish-observational-data-1977-2015-variables.xlsx'
    selected_columns = ['m','y', 'locality', 'seg_phos', 'seg_psize', 'seg_pet', 'seg_mat','seg_decs'
                        ,'US_RockPhos', 'USCalcium', 'native.bio', 'exotic.bio']


    data_frame = pd.read_excel(path_dir,sheetname='freshwater-fish-observational')

    data_frame = data_frame[selected_columns]


    index_less_2008 = data_frame[ (data_frame['y'] < 2008)].index
    index_greater_2013 = data_frame[ (data_frame['y'] > 2013)].index
    data_frame = data_frame.drop(index_less_2008)
    data_frame = data_frame.drop(index_greater_2013)

    dataframe = data_frame.sort_values(by=['m','y','locality'])

    data_frame.to_csv("raw_fish_output.csv",index=False)
    print(data_frame)


def load_raw_water_data():
    path_dir = './water_data/river-water-quality-raw-data-by-site-1975-2013.csv'
    selected_columns = ['river','location','sdate','fdval']
    data_frame = pd.read_csv(path_dir)
    data_frame = data_frame[selected_columns]
    data_frame['sdate'] = pd.to_datetime(data_frame['sdate'])

    index_less_2008 = data_frame[ (data_frame['sdate'] < datetime.date(year=2008,month=1,day=1))].index
    index_greater_2013 = data_frame[ (data_frame['sdate'] > datetime.date(year=2013,month=12,day=31))].index

    data_frame = data_frame.drop(index_less_2008)
    data_frame = data_frame.drop(index_greater_2013)
    data_frame.to_csv("raw_water_output.csv",index=False)
    print(data_frame)


def parse_fish_data():

    raw_fish_data = pd.read_csv('raw_fish_output.csv')
    raw_fish_data.dropna(axis=0, how='any')

    categories = ['y', 'locality', 'seg_phos', 'seg_psize', 'seg_pet', 'seg_mat',
                  'seg_decs','US_RockPhos', 'USCalcium', 'native.bio', 'exotic.bio']

    ret = []
    for year in range(2008,2014):
        exactly_year_data = raw_fish_data[raw_fish_data['y'] == year].copy()

        rivers = []
        for local in exactly_year_data['locality']:
            if local not in rivers:
                rivers.append(local)


        for river in rivers:
            exactly_year_local_data = exactly_year_data[ exactly_year_data['locality'] == river]
            merge_year_local = [year, river,]
            for cate in categories[2:]:
                merge_year_local.append(exactly_year_local_data[cate].mean().round(4))

            ret.append(merge_year_local)

    fish_data_year = pd.DataFrame(ret, columns=categories)
    fish_data_year.to_csv("fish_output.csv", index=False)

def parse_water_data():
    raw_water_data = pd.read_csv('raw_water_output.csv')
    raw_water_data.dropna(axis=0, how='any')
    raw_water_data['sdate'] = pd.to_datetime(raw_water_data['sdate'])

    ret = []
    categories = ['river','sdate','fdval']
    for year in range(2008,2014):
        date_year_lower, date_year_upper = datetime.date(year=year,month=1,day=1),datetime.date(year=year,month=12,day=31)

        exactly_year_data = raw_water_data[ (raw_water_data['sdate'] >= date_year_lower) & (raw_water_data['sdate'] <= date_year_upper)].copy()


        rivers = []
        for local in exactly_year_data['river']:
            if local not in rivers:
                rivers.append(local)

        for river in rivers:
            exactly_year_local_data = exactly_year_data[exactly_year_data['river'] == river]
            merge_year_local = [year, river,exactly_year_local_data['fdval'].mean().round(4) ]
            ret.append(merge_year_local)

    water_data_year = pd.DataFrame(ret, columns=categories)
    water_data_year.to_csv("water_output.csv", index=False)

