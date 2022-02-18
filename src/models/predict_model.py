import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_log_error, mean_absolute_percentage_error


class Predictor:
    def run_model(self):
        # Define config
        dd = '../../data'
        s_date = '2011-01-01'
        e_date = '2012-12-01'
        s_date_1 = '2012-12-02'
        s_date_2 = '2012-12-31'

        # Read data
        df_tr = pd.read_csv(
            '''{data_dir}/interim/day_wise_{start_date}_{end_date}.csv'''.format(
                data_dir=dd,
                start_date=s_date,
                end_date=e_date
            ))
        df_ts = pd.read_csv(
            '''{data_dir}/interim/day_wise_{start_date}_{end_date}.csv'''.format(
                data_dir=dd,
                start_date=s_date_1,
                end_date=s_date_2
            ))

        # Process data
        df_tr['label'] = 'train'
        df_ts['label'] = 'test'

        com = df_tr.append(df_ts)
        com_1 = com.drop(
            columns={
                'serial_no', 'casual_user_count', 'registered_user_count',
                'date_of_event'
            }
        )

        # Slight sub-optimal design introduced in the code here where
        # we combined train and test before and here split them up again.
        # This was done because there were transformations that were common
        # to both sets in the above code.
        tr_df = com_1[com_1['label'] == 'train'].copy()
        ts_df = com_1[com_1['label'] == 'test'].copy()

        # Outlier removal
        iso = IsolationForest(contamination=0.1, random_state=42)
        yhat = iso.fit_predict(
            tr_df[['temperature', 'real_feel_temperature', 'humidity', 'wind_speed',
                   'total_user_count']]
        )
        tr_df['is_outlier'] = yhat
        tr_df['is_outlier'] = tr_df['is_outlier'].apply(
            lambda x: True if x == -1 else False
        )
        tr_wo_outlier = tr_df[tr_df['is_outlier'] != True]
        com_2 = tr_wo_outlier.drop(columns='is_outlier').append(ts_df)

        com_3 = com_2.copy()
        com_3[
            ['month_number', 'num_day_of_week', 'year']
        ] = com_2[
            ['month_number', 'num_day_of_week', 'year']
        ].copy().astype(str)

        com_3_dummy = pd.get_dummies(
            com_3,
            columns={'month_number', 'num_day_of_week', 'season', 'weather'}
        )

        tr_f, tr_l, ts_f, ts_l, fl = self.train_test_split(
            com_3_dummy)

        # Fit model
        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(tr_f, tr_l)

        # Make predictions
        tr_p = rf.predict(tr_f)
        ts_p = rf.predict(ts_f)

        # Compute diagnostics
        train_rmsle = np.sqrt(mean_squared_log_error(tr_l, tr_p))
        train_mape = mean_absolute_percentage_error(tr_l, tr_p)
        test_rmsle = np.sqrt(mean_squared_log_error(ts_l, ts_p))
        test_mape = mean_absolute_percentage_error(ts_l, ts_p)
        print(f'Train RMSLE for {"First RF Model"} is - {round(train_rmsle, 4)}')
        print(f'Train MAPE for {"First RF Model"} is - {round(train_mape * 100, 2)} %')
        print('----------------------------------------------------------------')
        print(f'Test RMSLE for {"First RF Model"} is - {round(test_rmsle, 4)}')
        print(f'Test MAPE for {"First RF Model"} is - {round(test_mape * 100, 2)} %')

    def train_test_split(self, df):
        tr_df = df[df['label'] == 'train'].drop(
            columns={'label', 'total_user_count', 'date_string'}
        )

        tr_f = np.array(tr_df)
        tr_l = np.array(
            df[df['label'] == 'train']['total_user_count']
        )

        ts_f = np.array(
            df[df['label'] == 'test'].drop(
                columns={'label', 'total_user_count', 'date_string'}
            )
        )
        ts_l = np.array(
            df[df['label'] == 'test']['total_user_count']
        )

        fl = tr_df.columns

        return tr_f, tr_l, ts_f, ts_l, fl


def main():
    p = Predictor()
    p.run_model()


if __name__ == '__main__':
    main()
