from dataset import *
from gradient_descent import *
from normal_equation import *

class Activity01:

    @classmethod
    def run_activity(cls):

        print('Finding best parameters')
        cls.find_best_parameters()
        print()

        features = [
            'bias',
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'weekday_is_monday',
            'weekday_is_tuesday',
            'weekday_is_wednesday',
            'weekday_is_thursday',
            'weekday_is_friday',
            'weekday_is_saturday',
            'weekday_is_sunday',
            'is_weekend',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]

        target = 'shares'

        training_ds = Dataset('train_one_std_normalized_x.csv', features, target)
        training_ds.load()

        gd = GradientDescent(training_ds)
        gd.calculate_optimal(0.4)

        test_dataset = Dataset('test_normalized_x.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(gd.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)


        features = [
            'bias',
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]

        target = 'shares'

        training_ds = Dataset('train_no_category_normalized_x.csv', features, target)
        training_ds.load()

        gd = GradientDescent(training_ds)
        gd.calculate_optimal(0.4)

        test_dataset = Dataset('test_no_category_normalized_x.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(gd.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)



        features = [
            'bias',
            'n_tokens_title',
            'n_tokens_content',
            'num_hrefs',
            'num_imgs',
            'num_videos',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]

        target = 'shares'

        training_ds = Dataset('train_deleted_columns_normalized_x.csv', features, target)
        training_ds.load()

        gd = GradientDescent(training_ds)
        gd.calculate_optimal(0.7)

        test_dataset = Dataset('test_deleted_columns_normalized_x.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(gd.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys

        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)


        features = [
            'bias',
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'complex_n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'complex_num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'complex_kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'complex_self_reference_avg_sharess',
            'weekday_is_monday',
            'weekday_is_tuesday',
            'weekday_is_wednesday',
            'weekday_is_thursday',
            'weekday_is_friday',
            'weekday_is_saturday',
            'weekday_is_sunday',
            'is_weekend',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]

        target = 'shares'

        training_ds = Dataset('train_complex_quadratic_normalized_x.csv', features, target)
        training_ds.load()

        gd = GradientDescent(training_ds)
        gd.calculate_optimal(0.4)

        test_dataset = Dataset('test_complex_quadratic_normalized_x.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(gd.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)

        cls.plot_results()


        features = [
            'bias',
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'weekday_is_monday',
            'weekday_is_tuesday',
            'weekday_is_wednesday',
            'weekday_is_thursday',
            'weekday_is_friday',
            'weekday_is_saturday',
            'weekday_is_sunday',
            'is_weekend',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]
        target = 'shares'

        ds = Dataset('train_start.csv', features, target)
        ds.load()

        ne = NormalEquation(ds)
        ne.calculate_optimal()

        test_dataset = Dataset('test_start.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(ne.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)


        features = [
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'weekday_is_monday',
            'weekday_is_tuesday',
            'weekday_is_wednesday',
            'weekday_is_thursday',
            'weekday_is_friday',
            'weekday_is_saturday',
            'weekday_is_sunday',
            'is_weekend',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]
        target = 'shares'

        ds = Dataset('train_ne.csv', features, target)
        ds.load()

        ne = NormalEquation(ds)
        thetas = ne.calculate_optimal()

        test_dataset = Dataset('test.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(ne.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)


        features = [
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]
        target = 'shares'

        ds = Dataset('train_ne_normalized_x.csv', features, target)
        ds.load()

        ne = NormalEquation(ds)
        thetas = ne.calculate_optimal()

        test_dataset = Dataset('test_ne_normalized_x.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(ne.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)


        features = [
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'weekday_is_monday',
            'weekday_is_tuesday',
            'weekday_is_wednesday',
            'weekday_is_thursday',
            'weekday_is_friday',
            'weekday_is_saturday',
            'weekday_is_sunday',
            'is_weekend',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]
        target = 'shares'

        ds = Dataset('train_ne_half.csv', features, target)
        ds.load()

        ne = NormalEquation(ds)
        thetas = ne.calculate_optimal()

        test_dataset = Dataset('test.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(ne.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)


        features = [
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'weekday_is_monday',
            'weekday_is_tuesday',
            'weekday_is_wednesday',
            'weekday_is_thursday',
            'weekday_is_friday',
            'weekday_is_saturday',
            'weekday_is_sunday',
            'is_weekend',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]
        target = 'shares'

        ds = Dataset('train_ne_quarter.csv', features, target)
        ds.load()

        ne = NormalEquation(ds)
        thetas = ne.calculate_optimal()

        test_dataset = Dataset('test.csv', features, '')
        test_dataset.load()

        predicted_ys = np.apply_along_axis(ne.predict, axis = 1, arr = test_dataset.x)

        test_target_dataset = pandas.read_csv('test_target.csv')
        target_ys = test_target_dataset.loc[:, 'shares']

        errors = predicted_ys - target_ys
        mse = ((errors ** 2).sum()) / len(errors)
        print(mse)


    @classmethod
    def plot_results(cls):

        plt.plot(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290, 1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500],
            [63253842, 62420107, 61840253, 61395104, 61026976, 60707149, 60420764, 60159715, 59919270, 59696435, 59489141, 59295845, 59115315, 58946521, 58788570, 58640673, 58502117, 58372257, 58250500, 58136302, 58029159, 57928607, 57834214, 57745577, 57662325, 57584108, 57510604, 57441510, 57376545, 57315446, 57257966, 57203876, 57152963, 57105026, 57059877, 57017341, 56977255, 56939465, 56903828, 56870211, 56838487, 56808540, 56780258, 56753541, 56728291, 56704418, 56681837, 56660471, 56640244, 56621088, 56602938, 56585732, 56569415, 56553932, 56539233, 56525272, 56512005, 56499391, 56487391, 56475968, 56465090, 56454724, 56444841, 56435413, 56426412, 56417816, 56409600, 56401744, 56394227, 56387029, 56380134, 56373524, 56367183, 56361097, 56355252, 56349635, 56344233, 56339036, 56334031, 56329210, 56324563, 56320080, 56315754, 56311575, 56307538, 56303634, 56299858, 56296202, 56292662, 56289231, 56285905, 56282678, 56279546, 56276505, 56273550, 56270678, 56267884, 56265166, 56262521, 56259944, 56257434, 56254988, 56252602, 56250275, 56248004, 56245787, 56243621, 56241506, 56239439, 56237418, 56235442, 56233508, 56231616, 56229765, 56227951, 56226176, 56224436, 56222731, 56221061, 56219423, 56217817, 56216241, 56214695, 56213179, 56211690, 56210229, 56208795, 56207386, 56206002, 56204643, 56203308, 56201995, 56200706, 56199438, 56198192, 56196966, 56195761, 56194576, 56193410, 56192263, 56191135, 56190025, 56188932, 56187857, 56186799, 56185757, 56184732, 56183722, 56182728, 56181749, 56180784],
            label = 'Learning rate (0.01)'
        )
        plt.plot(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290, 1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500],
            [62419888, 59281839, 57908375, 57186458, 56795806, 56577082, 56449042, 56369823, 56317642, 56281025, 56253818, 56232639, 56215561, 56201439, 56189555, 56179428, 56170720, 56163182, 56156619, 56150879, 56145835, 56141385, 56137442, 56133935, 56130803, 56127993, 56125463, 56123174, 56121094, 56119198, 56117460, 56115862, 56114385, 56113015, 56111740, 56110548, 56109430, 56108377, 56107382, 56106439, 56105542, 56104688, 56103870, 56103087, 56102334, 56101609, 56100909, 56100233, 56099577, 56098941, 56098323, 56097721, 56097134, 56096561, 56096002, 56095455, 56094919, 56094394, 56093878, 56093373, 56092876, 56092387, 56091907, 56091434, 56090968, 56090509, 56090057, 56089611, 56089171, 56088736, 56088307, 56087884, 56087465, 56087051, 56086643, 56086238, 56085839, 56085443, 56085052, 56084665, 56084282, 56083903, 56083527, 56083156, 56082788, 56082423, 56082062, 56081705, 56081351, 56081000, 56080652, 56080307, 56079966, 56079628, 56079292, 56078960, 56078630, 56078304, 56077980, 56077659, 56077341, 56077025, 56076713, 56076402, 56076095, 56075790, 56075488, 56075188, 56074891, 56074596, 56074303, 56074014, 56073726, 56073441, 56073158, 56072878, 56072599, 56072323, 56072050, 56071778, 56071509, 56071242, 56070977, 56070715, 56070454, 56070196, 56069939, 56069685, 56069433, 56069183, 56068935, 56068688, 56068444, 56068202, 56067962, 56067724, 56067487, 56067253, 56067020, 56066789, 56066560, 56066333, 56066108, 56065885, 56065663, 56065443, 56065225, 56065009, 56064794, 56064581, 56064370],
            label = 'Learning rate (0.1)'
        )
        plt.plot(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000, 1010, 1020, 1030, 1040, 1050, 1060, 1070, 1080, 1090, 1100, 1110, 1120, 1130, 1140, 1150, 1160, 1170, 1180, 1190, 1200, 1210, 1220, 1230, 1240, 1250, 1260, 1270, 1280, 1290, 1300, 1310, 1320, 1330, 1340, 1350, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460, 1470, 1480, 1490, 1500],
            [61101843, 56675733, 56298193, 56208774, 56167063, 56143622, 56129398, 56120161, 56113734, 56108952, 56105174, 56102036, 56099327, 56096917, 56094725, 56092700, 56090805, 56089018, 56087321, 56085702, 56084151, 56082662, 56081230, 56079849, 56078517, 56077231, 56075989, 56074787, 56073625, 56072501, 56071414, 56070361, 56069342, 56068356, 56067401, 56066477, 56065582, 56064715, 56063875, 56063062, 56062275, 56061513, 56060774, 56060059, 56059367, 56058696, 56058046, 56057417, 56056807, 56056217, 56055645, 56055091, 56054554, 56054034, 56053531, 56053043, 56052570, 56052112, 56051669, 56051239, 56050822, 56050419, 56050028, 56049649, 56049282, 56048927, 56048582, 56048248, 56047925, 56047611, 56047308, 56047013, 56046728, 56046451, 56046183, 56045923, 56045671, 56045427, 56045190, 56044961, 56044738, 56044523, 56044314, 56044111, 56043914, 56043724, 56043539, 56043360, 56043186, 56043017, 56042854, 56042695, 56042541, 56042392, 56042247, 56042106, 56041970, 56041838, 56041710, 56041585, 56041464, 56041347, 56041233, 56041122, 56041015, 56040911, 56040809, 56040711, 56040616, 56040523, 56040433, 56040346, 56040261, 56040178, 56040098, 56040020, 56039944, 56039871, 56039799, 56039729, 56039662, 56039596, 56039532, 56039470, 56039409, 56039350, 56039293, 56039237, 56039183, 56039130, 56039078, 56039028, 56038979, 56038932, 56038886, 56038841, 56038797, 56038754, 56038712, 56038671, 56038632, 56038593, 56038555, 56038518, 56038482, 56038447, 56038413, 56038379, 56038347, 56038315, 56038284],
            label = 'Learning rate (0.4)'
        )
        plt.xlabel('# Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

        plt.plot(
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400],
            [60045001, 56644393, 55739670, 55447205, 55329103, 55267852, 55229476, 55202744, 55183095, 55168209, 55156688, 55147610, 55140331, 55134396, 55129474, 55125324, 55121770, 55118678, 55115951, 55113513, 55111307, 55109289, 55107425, 55105688, 55104057, 55102515, 55101049, 55099648, 55098304, 55097008, 55095756, 55094542, 55093362, 55092214, 55091094, 55090000, 55088929, 55087881, 55086854, 55085847, 55084858, 55083887, 55082932, 55081994, 55081071, 55080163, 55079270, 55078390, 55077525, 55076672, 55075833, 55075006, 55074192, 55073390, 55072600, 55071821, 55071054, 55070298, 55069554, 55068820, 55068096, 55067384, 55066681, 55065989, 55065307, 55064634, 55063972, 55063319, 55062675, 55062041, 55061416, 55060799, 55060192, 55059593, 55059003, 55058422, 55057849, 55057284, 55056727, 55056178, 55055637],
            label = 'Learning rate (0.4)'
        )
        plt.xlabel('# Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

        plt.plot(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200],
            [62358183, 62138325, 62054240, 62010457, 61982370, 61961763, 61945365, 61931668, 61919878, 61909526, 61900304, 61892001, 61884461, 61877566, 61871226, 61865368, 61859934, 61854877, 61850158, 61845742, 61841602, 61837714, 61834058, 61830613, 61827365, 61824299, 61821402, 61818663, 61816071, 61813617, 61811292, 61809088, 61806998, 61805015, 61803133, 61801346, 61799648, 61798035, 61796502, 61795044, 61793658, 61792339, 61791084, 61789889, 61788751, 61787667, 61786635, 61785651, 61784714, 61783820, 61782967, 61782154, 61781378, 61780638, 61779931, 61779256, 61778611, 61777996, 61777407, 61776845, 61776307, 61775793, 61775302, 61774831, 61774381, 61773951, 61773538, 61773144, 61772765, 61772403, 61772056, 61771723, 61771404, 61771098, 61770805, 61770524, 61770254, 61769994, 61769746, 61769507, 61769277, 61769057, 61768845, 61768642, 61768447, 61768259, 61768078, 61767904, 61767737, 61767576, 61767421, 61767272, 61767129, 61766990, 61766857, 61766729, 61766606, 61766487, 61766372, 61766262, 61766155, 61766052, 61765953, 61765858, 61765765, 61765676, 61765590, 61765508, 61765428, 61765350, 61765276, 61765204, 61765134, 61765067, 61765002, 61764939, 61764878, 61764819, 61764763, 61764708, 61764655, 61764603, 61764554, 61764505, 61764459, 61764414, 61764370, 61764328, 61764287, 61764248, 61764210, 61764172, 61764136, 61764102, 61764068, 61764035, 61764004, 61763973, 61763943, 61763914, 61763886, 61763859, 61763833, 61763807, 61763783, 61763759, 61763735, 61763713, 61763691, 61763669, 61763649, 61763629, 61763609, 61763590, 61763572, 61763554, 61763537, 61763520, 61763503, 61763487, 61763472, 61763457, 61763442, 61763428, 61763414, 61763401, 61763388, 61763375, 61763363, 61763351, 61763339, 61763327, 61763316, 61763305, 61763295, 61763285, 61763275, 61763265, 61763256, 61763246, 61763237, 61763229, 61763220, 61763212, 61763204, 61763196, 61763188, 61763180, 61763173, 61763166, 61763159, 61763152, 61763145, 61763139, 61763133, 61763126, 61763120, 61763114, 61763109, 61763103, 61763097],
            label = 'Learning rate (0.4)'
        )
        plt.xlabel('# Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()

        plt.plot(
            [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400],
            [62866689, 62427988, 62365273, 62347705, 62341851, 62339799, 62339060, 62338783, 62338673, 62338623, 62338598, 62338582, 62338571, 62338563, 62338557, 62338552, 62338548, 62338545, 62338543, 62338541, 62338540, 62338539, 62338538, 62338537, 62338536, 62338536, 62338536, 62338535, 62338535, 62338535, 62338535, 62338535, 62338535, 62338535, 62338535, 62338535, 62338535, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534, 62338534],
            label = 'Learning rate (0.7)'
        )
        plt.xlabel('# Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()


        plt.plot(
            [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990, 1000],
            [61221270, 56408131, 55947188, 55835809, 55789433, 55765275, 55750807, 55741153, 55734156, 55728761, 55724406, 55720765, 55717637, 55714892, 55712440, 55710220, 55708186, 55706304, 55704550, 55702905, 55701354, 55699884, 55698488, 55697157, 55695885, 55694667, 55693499, 55692378, 55691300, 55690263, 55689264, 55688302, 55687373, 55686478, 55685614, 55684780, 55683974, 55683196, 55682444, 55681718, 55681015, 55680336, 55679680, 55679045, 55678431, 55677837, 55677262, 55676706, 55676169, 55675648, 55675145, 55674658, 55674186, 55673730, 55673288, 55672860, 55672447, 55672046, 55671658, 55671283, 55670919, 55670567, 55670227, 55669897, 55669578, 55669268, 55668969, 55668679, 55668398, 55668127, 55667863, 55667608, 55667361, 55667122, 55666891, 55666667, 55666449, 55666239, 55666035, 55665838, 55665647, 55665462, 55665283, 55665109, 55664941, 55664778, 55664620, 55664467, 55664319, 55664175, 55664036, 55663902, 55663771, 55663645, 55663522, 55663404, 55663289, 55663177, 55663069, 55662965, 55662863],
            label = 'Learning rate (0.4)'
        )
        plt.xlabel('# Iterations')
        plt.ylabel('Cost')
        plt.legend()
        plt.show()


    @classmethod
    def find_best_parameters(cls):

        print('Training non-normalized 0.0.1')
        cls.train_non_normalized_0_0_1()
        print()

        print('Training non-normalized 0.1')
        cls.train_non_normalized_0_1()
        print()

        print('Training non-normalized 0.4')
        cls.train_non_normalized_0_4()
        print()

        print('Training non-normalized 1')
        cls.train_non_normalized_1()
        print()

        print('Training normalized 0.0.1')
        cls.train_normalized_0_0_1()
        print()

        print('Training normalized 0.1')
        cls.train_normalized_0_1()
        print()

        print('Training normalized 0.4')
        cls.train_normalized_0_4()
        print()

        print('Training normalized 0.5')
        cls.train_normalized_0_5()
        print()

        print('Training normalized 1')
        cls.train_normalized_1()
        print()

    @classmethod
    def train_non_normalized_0_0_1(cls):
        cls.train_gradient_descent('train_start.csv', 0.01)

    @classmethod
    def train_non_normalized_0_1(cls):
        cls.train_gradient_descent('train_start.csv', 0.1)

    @classmethod
    def train_non_normalized_0_4(cls):
        cls.train_gradient_descent('train_start.csv', 0.4)

    @classmethod
    def train_non_normalized_1(cls):
        cls.train_gradient_descent('train_start.csv', 1)

    @classmethod
    def train_normalized_0_0_1(cls):
        cls.train_gradient_descent('train_normalized_x.csv', 0.01)

    @classmethod
    def train_normalized_0_1(cls):
        cls.train_gradient_descent('train_normalized_x.csv', 0.1)

    @classmethod
    def train_normalized_0_4(cls):
        cls.train_gradient_descent('train_normalized_x.csv', 0.4)

    @classmethod
    def train_normalized_0_5(cls):
        cls.train_gradient_descent('train_normalized_x.csv', 0.5)

    @classmethod
    def train_normalized_1(cls):
        cls.train_gradient_descent('train_normalized_x.csv', 1)

    @classmethod
    def train_gradient_descent(cls, training_dataset, learning_rate):

        features = [
            'bias',
            'timedelta',
            'n_tokens_title',
            'n_tokens_content',
            'n_unique_tokens',
            'n_non_stop_words',
            'n_non_stop_unique_tokens',
            'num_hrefs',
            'num_self_hrefs',
            'num_imgs',
            'num_videos',
            'average_token_length',
            'num_keywords',
            'data_channel_is_lifestyle',
            'data_channel_is_entertainment',
            'data_channel_is_bus',
            'data_channel_is_socmed',
            'data_channel_is_tech',
            'data_channel_is_world',
            'kw_min_min',
            'kw_max_min',
            'kw_avg_min',
            'kw_min_max',
            'kw_max_max',
            'kw_avg_max',
            'kw_min_avg',
            'kw_max_avg',
            'kw_avg_avg',
            'self_reference_min_shares',
            'self_reference_max_shares',
            'self_reference_avg_sharess',
            'weekday_is_monday',
            'weekday_is_tuesday',
            'weekday_is_wednesday',
            'weekday_is_thursday',
            'weekday_is_friday',
            'weekday_is_saturday',
            'weekday_is_sunday',
            'is_weekend',
            'LDA_00',
            'LDA_01',
            'LDA_02',
            'LDA_03',
            'LDA_04',
            'global_subjectivity',
            'global_sentiment_polarity',
            'global_rate_positive_words',
            'global_rate_negative_words',
            'rate_positive_words',
            'rate_negative_words',
            'avg_positive_polarity',
            'min_positive_polarity',
            'max_positive_polarity',
            'avg_negative_polarity',
            'min_negative_polarity',
            'max_negative_polarity',
            'title_subjectivity',
            'title_sentiment_polarity',
            'abs_title_subjectivity',
            'abs_title_sentiment_polarity'
        ]

        target = 'shares'

        ds = Dataset(training_dataset, features, target)
        ds.load()

        gd = GradientDescent(ds)
        gd.calculate_optimal(learning_rate)


if __name__ == '__main__':
    Activity01.run_activity()
