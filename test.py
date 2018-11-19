import pandas as pd
import numpy as np

from sklearn.model_selection import KFold

train_data = pd.read_csv('kaggle_titanic_trainclean.csv')
test_data = pd.read_csv('kaggle_titanic_test.csv')

features = train_data.columns[2:]

numeric_features = []
categorical_features = []

for dtype, feature in zip(train_data.dtypes[2:], train_data.columns[2:]):
    if dtype == object:
        #print(column)
        #print(train_data[column].describe())
        categorical_features.append(feature)
    else:
        numeric_features.append(feature)
#categorical_features

categorical_features =["Sex"]

# This way we have randomness and are able to reproduce the behaviour within this cell.
np.random.seed(13)

def impact_coding(data, feature, target):
    """
    This function does two things:
      - Directly compute the impact coded series of the feature
      - Compute the mapping to apply to test data and data to score

    Notably, the train data does not use the mapping to avoid leaking information. Instead,
    train data is computed using nested KFold
	"""
    n_folds = 20
    n_inner_folds = 10
    impact_coded = pd.Series()
    
    # Global mean of the target, applied to unknown values
    global_mean = data[target].mean()

    # This DF receives all computed means, per value of the feature.
    # Shape: (n_feature_values, n_folds * n_inner_folds)
    # Globally averaging it yields the final mapping to apply to test data
    mapping_computation_df = pd.DataFrame()

    split = 0
    kf = KFold(n_splits=n_folds, shuffle=True)

    #print("Global means per value: %s" % data.groupby(by=feature)[target].mean())

    for infold, oof in kf.split(data[feature]):
            #print("Split=%s means_per_value=%s" % (split, data.iloc[infold].groupby(by=feature)[target].mean()))
            #print("Infold=%s" % infold)

            # This dataframe has, at the end of the loop, shape=(n_feature_values, n_inner_folds)
            # It's what we will append to the global mapping_computation_df
            inner_means_df = pd.DataFrame()

            # Fallback value for this outer fold when one of the inner fold has missing value
            infold_mean = data.iloc[infold][target].mean()
            #print("Fold=%s infold=%s oof=%s infold_mean=%s" % (split, infold.sum(), oof.sum(), infold_mean))

            kf_inner = KFold(n_splits=n_inner_folds, shuffle=True)
            inner_split = 0
            for infold_inner, oof_inner in kf_inner.split(data.iloc[infold]):
                #print("infold_inner=%s" % infold_inner)
                # Actual mean per target value on the infold_inner
                infold_inner_mean = data.iloc[infold].iloc[infold_inner].groupby(by=feature)[target].mean()
                #print("  split=%s inner=%s infold_inner_mean=%s" % (split, inner_split, infold_inner_mean.to_json(orient="columns")))
                

                #impact_coded_cv = impact_coded_cv.append(data.iloc[infold].apply(
                #            lambda x: infold_inner_mean[x[feature]]
                #                      if x[feature] in infold_inner_mean.index
                #                      else infold_mean
                #            , axis=1))

                # Append the means per value to the per-innerfold DF
                inner_means_df = inner_means_df.join(pd.DataFrame(infold_inner_mean), rsuffix=inner_split, how='outer')
                inner_means_df.fillna(infold_mean, inplace=True)

                inner_split += 1

            # Now, just append all infold_inner means to the global mapping_computation_df
            # And fill with global means values that were not in the infold (so not in any of the infold_inner)
            mapping_computation_df = mapping_computation_df.join(pd.DataFrame(inner_means_df), rsuffix=split, how='outer')
            mapping_computation_df.fillna(global_mean, inplace=True)
            
            #print("split=%s inner_means_df=%s" % (split, inner_means_df.to_json(orient="columns")))

            # And actually apply the mean of all infold_inner means to the actual train data, on oof
            oof_data = data.iloc[oof]
            inner_folds_mean = inner_means_df.mean(axis=1)
            impact_coded_oof = oof_data[feature].map(inner_folds_mean).fillna(global_mean)
            impact_coded = impact_coded.append(impact_coded_oof)

            #print("split=%s inner_folds_mean=%s" % (split, inner_folds_mean.to_json(orient="columns")))

            #impact_coded_oof = oof_data.apply(lambda x: inner_means_df.loc[x[feature]].mean()
            #                          if x[feature] in inner_means_df.index
            #                          else global_mean
            #                , axis=1)
            

            split += 1

            #print(". inner_oofmean_cv shape=%s" % (inner_means_df.shape,))
            # Also populate mapping_computation_df
            #print(" infold_inner_mean_cv shape=%s" % (mapping_computation_df.shape,))
            
            #print("IC Shape before split=%s shape=%s" % (split, impact_coded.shape))

            #print("IC Shape after split=%s shape=%s" % (split, impact_coded.shape))

    mapping = mapping_computation_df.mean(axis=1)

    print("IC: %s" % impact_coded)
    print("End mapping:%s" % mapping_computation_df.mean(axis=1))
    return impact_coded, mapping, global_mean

print "GO"

# Apply the encoding to training and test data, and preserve the mapping
impact_coding_map = {}
for f in categorical_features:
    print("Impact coding for {}".format(f))
    train_data["impact_encoded_{}".format(f)], impact_coding_mapping, default_coding = impact_coding(train_data, f, "Survived")
    impact_coding_map[f] = (impact_coding_mapping, default_coding)
    mapping, default_mean = impact_coding_map[f]
    test_data["impact_encoded_{}".format(f)] = test_data.apply(lambda x: mapping[x[f]]
                                                                         if x[f] in mapping
                                                                         else default_mean
                                                               , axis=1)

#print("%s" % train_data)