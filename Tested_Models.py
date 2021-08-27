"""
Created on Mon Jul 26 23:09:02 2021

@author: Konstantinos
"""

# Test number of features
# temp_train_X = final_train_X.drop(columns=["Age", "Fare", "SibSp", "Parch"])
# temp_test_X = final_test_X.drop(columns=["Age", "Fare", "SibSp", "Parch"])
temp_train_X = final_train_X#.drop(columns=["Parch", "S"])
temp_test_X = final_test_X#.drop(columns=["Age", "Parch"])
# Models
#############################################################################################################################
# GridSearch for DecisionTreeClassifier
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# max_depth = DecisionTreeClassifier(random_state=42).fit(temp_train_X, train_Y).get_depth()+1
# parameters = {"criterion": ["gini", "entropy"], "max_depth": [x for x in range(1,max_depth)],
#               "min_samples_leaf": [x for x in range(1,20)]}
# tree = DecisionTreeClassifier(random_state=42)

# grid_search = GridSearchCV(estimator=tree, param_grid=parameters, cv=cv10, 
#                             n_jobs=-1, verbose=3, scoring="accuracy").fit(temp_train_X, train_Y)

# grid_results = pd.DataFrame(grid_search.cv_results_)
# print(grid_results.iloc[grid_results[["mean_test_score"]].idxmax()] \
#       [["params", "mean_test_score"]])
    
# classifier = DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_leaf=6, random_state=42)
# classifier.fit(temp_train_X, train_Y)
# predictions = pd.DataFrame(classifier.predict(temp_test_X))
# gender_sub = pd.concat([test_PassengerId, predictions], axis=1)
# gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export3.csv', header=["PassengerId", "Survived"], index=False)
# {'criterion': 'entropy', 'max_depth': 9, 'min_samples_leaf': 6} 0.839109 without "Age" feature
############################################################################################################################
# GridSearch for KNeighborsClassifier
# parameters = {"weights": ["uniform", "distance"], "n_neighbors": [x for x in range(5,30)], 
#               "metric": ["euclidean", "chebyshev", "minkowski"]}
# knn = KNeighborsClassifier()
# grid_search = GridSearchCV(estimator=knn, param_grid=parameters, cv=cv10, 
#                               n_jobs=-1, verbose=3, scoring="accuracy").fit(temp_train_X, train_Y)

# grid_results = pd.DataFrame(grid_search.cv_results_)
# print(grid_results.iloc[grid_results[["mean_test_score"]].idxmax()] \
#       [["params", "mean_test_score"]])
    
# classifier = KNeighborsClassifier(metric="minkowski", n_neighbors=22, weights="uniform")
# classifier.fit(temp_train_X, train_Y)
# predictions = pd.DataFrame(classifier.predict(temp_test_X))
# gender_sub = pd.concat([test_PassengerId, predictions], axis=1)
# gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export3.csv', header=["PassengerId", "Survived"], index=False)
# {'metric': 'euclidean', 'n_neighbors': 22, 'weights': 'uniform'} 0.830146 without "Age", "Fare", "Parch" features
# {'metric': 'minkowski', 'n_neighbors': 22, 'weights': 'uniform'} 0.830146 without "Age", "Fare", "Parch" features
# {'metric': 'euclidean', 'n_neighbors': 6, 'weights': 'distance'} 0.836913 without "Parch" feature but with Age, Fare transformation
###########################################################################################################################
# GridSearch for SVC
parameters = {"kernel": ["poly", "rbf"], "gamma": ["scale", "auto"], "random_state": [42]}
svc = SVC()
grid_search = GridSearchCV(estimator=svc, param_grid=parameters, cv=cv10, 
                              n_jobs=-1, verbose=3, scoring="accuracy").fit(temp_train_X, train_Y)

grid_results = pd.DataFrame(grid_search.cv_results_)
print(grid_results.iloc[grid_results[["mean_test_score"]].idxmax()] \
      [["params", "mean_test_score"]])
    
classifier = SVC(gamma="scale", kernel="poly", random_state=42)
classifier.fit(temp_train_X, train_Y)
predictions = pd.DataFrame(classifier.predict(temp_test_X))
gender_sub = pd.concat([test_PassengerId, predictions], axis=1)
gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export.csv', header=["PassengerId", "Survived"], index=False)
# {'gamma': 'scale', 'kernel': 'rbf', 'random_state': 42} 0.827886 without "Age", "Fare", "Parch" features
# {'gamma': 'scale', 'kernel': 'poly', 'random_state': 42} 0.833542 
################################################################################################################################
# cv for Naive Bayes
# nb = GaussianNB()
# all_scores = cross_validate(nb, temp_train_X, train_Y, scoring=["f1", "accuracy"], cv=cv10, n_jobs=-1, verbose=1)
# avg_accuracy = np.mean(all_scores["test_accuracy"]) # The average accuracy
# print("acc is {}".format(avg_accuracy))   # 0.8144662921348313
#################################################################################################################################
# GridSearch for RandomForest
# parameters = {"n_estimators": [25, 50, 75, 100], "criterion": ["gini", "entropy"], "random_state": [42], "n_jobs": [-1], 
#               "max_depth": [x for x in range(2,15)], "min_samples_leaf": [x for x in range(1,4)]}
# rf = RandomForestClassifier()
# grid_search = GridSearchCV(estimator=rf, param_grid=parameters, cv=cv10, 
#                                 n_jobs=-1, verbose=3, scoring="accuracy").fit(temp_train_X, train_Y)

# grid_results = pd.DataFrame(grid_search.cv_results_)
# print(grid_results.iloc[grid_results[["mean_test_score"]].idxmax()] \
#       [["params", "mean_test_score"]])

# classifier = RandomForestClassifier(criterion="gini", n_estimators=75, n_jobs=-1, random_state=42, max_depth=10)
# classifier.fit(temp_train_X, train_Y)
# predictions = pd.DataFrame(classifier.predict(temp_test_X))
# gender_sub = pd.concat([test_PassengerId, predictions], axis=1)
# gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export.csv', header=["PassengerId", "Survived"], index=False)
# {'criterion': 'gini', 'max_depth': 10, 'n_estimators': 75, 'n_jobs': -1, 'random_state': 42} 0.840258 without "Age", "Parch features"
# {'criterion': 'gini', 'max_depth': 7, 'max_features': 0.5, 'n_estimators': 25, 'n_jobs': -1, 'random_state': 42}  0.840258 without "Age", "Parch features"
##########################################################################################################################################
# GridSearch for MLP
# parameters = {"hidden_layer_sizes": [(45,45,45), (30,30,30), (10,10)], "solver": ["adam"], "max_iter": [600], "random_state": [42],
#               "learning_rate": ["invscaling", "adaptive"]}

# nn = MLPClassifier()
# grid_search = GridSearchCV(estimator=nn, param_grid=parameters, cv=cv10, 
#                                 n_jobs=-1, verbose=3, scoring="accuracy").fit(temp_train_X, train_Y)

# grid_results = pd.DataFrame(grid_search.cv_results_)
# print(grid_results.iloc[grid_results[["mean_test_score"]].idxmax()] \
#       [["params", "mean_test_score"]])

# classifier = MLPClassifier(hidden_layer_sizes=(30,30,30), solver="adam", max_iter=600, random_state=42, learning_rate="adaptive")
# classifier.fit(temp_train_X, train_Y)
# predictions = pd.DataFrame(classifier.predict(temp_test_X))
# gender_sub = pd.concat([test_PassengerId, predictions], axis=1)
# gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export.csv', header=["PassengerId", "Survived"], index=False)    
# {'hidden_layer_sizes': (45, 45, 45), 'learning_rate': 'invscaling', 'max_iter': 600, 'random_state': 42, 'solver': 'adam'} 0.836887
#########################################################################################################################################
# GridSearch for BaggingClassifier
# base_estimator = classifier = DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=2, random_state=42)
# parameters = {"base_estimator": [base_estimator], "n_estimators": [10, 20, 30, 40, 50, 100, 150], "n_jobs": [-1], "random_state": [42]}
# baggin = BaggingClassifier()
# grid_search = GridSearchCV(estimator=baggin, param_grid=parameters, cv=cv10, 
#                                 n_jobs=-1, verbose=3, scoring="accuracy").fit(temp_train_X, train_Y)

# grid_results = pd.DataFrame(grid_search.cv_results_)
# print(grid_results.iloc[grid_results[["mean_test_score"]].idxmax()] \
#       [["params", "mean_test_score"]])

# classifier = BaggingClassifier()
# classifier.fit(temp_train_X, train_Y)
# predictions = pd.DataFrame(classifier.predict(temp_test_X))
# gender_sub = pd.concat([test_PassengerId, predictions], axis=1)
# gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export.csv', header=["PassengerId", "Survived"], index=False) 
# {'base_estimator': DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42), 
# 'n_estimators': 100, 'n_jobs': -1, 'random_state': 42} 0.836887
# {'base_estimator': DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42), 
# 'n_estimators': 20, 'n_jobs': -1, 'random_state': 42} 0.840258 without "Age", "Parch"
##########################################################################################################################################
# GridSearch for StackingClassifier
# cls1 = SVC(gamma="scale", kernel="rbf", random_state=42)
# cls2 = MLPClassifier(hidden_layer_sizes=(30,30,30), solver="adam", max_iter=600, random_state=42, learning_rate="adaptive")
# cls3 = RandomForestClassifier(criterion="gini", n_estimators=75, n_jobs=-1, random_state=42, max_depth=10)
# cls_combo = [("svc", cls1), ("mlp", cls2), ("rf", cls3)]
# parameters = {"estimators": [cls_combo], "n_jobs": [-1], "passthrough": [True, False]}

# stacking = StackingClassifier(estimators=cls_combo)
# grid_search = GridSearchCV(estimator=stacking, param_grid=parameters, cv=cv10, 
#                                 n_jobs=-1, verbose=3, scoring="accuracy").fit(temp_train_X, train_Y)

# grid_results = pd.DataFrame(grid_search.cv_results_)
# print(grid_results.iloc[grid_results[["mean_test_score"]].idxmax()] \
#       [["params", "mean_test_score"]])

# classifier = StackingClassifier(estimators=cls_combo, n_jobs=-1, passthrough=True, cv=cv10)
# classifier.fit(temp_train_X, train_Y)
# predictions = pd.DataFrame(classifier.predict(temp_test_X))
# gender_sub = pd.concat([test_PassengerId, predictions], axis=1)
# gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export.csv', header=["PassengerId", "Survived"], index=False) 
#########################################################################################################################################
# Final Estimator
# temp_train_X = final_train_X.drop(columns=["Age", "Fare", "SibSp", "Parch"])
# temp_test_X = final_test_X.drop(columns=["Age", "Fare", "SibSp", "Parch"])
# #1
# temp_train_X = final_train_X.drop(columns=["Age"])
# temp_test_X = final_test_X.drop(columns=["Age"])
# cls1 = DecisionTreeClassifier(criterion="entropy", max_depth=9, min_samples_leaf=6, random_state=42)
# cls1.fit(temp_train_X, train_Y)
# predictions = pd.DataFrame(cls1.predict(temp_test_X))
# #2
# temp_train_X = final_train_X.drop(columns=["Parch"])
# temp_test_X = final_test_X.drop(columns=["Parch"])
# cls2 = KNeighborsClassifier(metric="euclidean", n_neighbors=6, weights="distance")
# cls2.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls2.predict(temp_test_X))], axis=1)
# #3
# temp_train_X = final_train_X
# temp_test_X = final_test_X
# cls3 = SVC(gamma="scale", kernel="poly", random_state=42)
# cls3.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls3.predict(temp_test_X))], axis=1)
# #4
# temp_train_X = final_train_X
# temp_test_X = final_test_X
# cls4 = GaussianNB()
# cls4.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls4.predict(temp_test_X))], axis=1)
# #5
# temp_train_X = final_train_X.drop(columns=["Age", "Parch"])
# temp_test_X = final_test_X.drop(columns=["Age", "Parch"])
# cls5 = RandomForestClassifier(criterion="gini", n_estimators=75, n_jobs=-1, random_state=42, max_depth=10)
# cls5.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls5.predict(temp_test_X))], axis=1)
# #6
# temp_train_X = final_train_X.drop(columns=["Age", "Parch"])
# temp_test_X = final_test_X.drop(columns=["Age", "Parch"])
# cls6 = RandomForestClassifier(criterion="gini", n_estimators=25, n_jobs=-1, random_state=42, max_depth=7, max_features=0.5)
# cls6.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls6.predict(temp_test_X))], axis=1)
# #7
# temp_train_X = final_train_X
# temp_test_X = final_test_X
# cls7 = MLPClassifier(hidden_layer_sizes=(45,45,45), solver="adam", max_iter=600, random_state=42, learning_rate="invscaling")
# cls7.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls7.predict(temp_test_X))], axis=1)
# #8
# temp_train_X = final_train_X.drop(columns=["Age", "Parch"])
# temp_test_X = final_test_X.drop(columns=["Age", "Parch"])
# base_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=42)
# cls8 = BaggingClassifier(base_estimator=base_estimator, n_estimators=20, n_jobs=-1, random_state=42)
# cls8.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls8.predict(temp_test_X))], axis=1)
# #9
# temp_train_X = final_train_X
# temp_test_X = final_test_X
# base_estimator = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
# cls9 = BaggingClassifier(base_estimator=base_estimator, n_estimators=100, n_jobs=-1, random_state=42)
# cls9.fit(temp_train_X, train_Y)
# predictions = pd.concat([predictions, pd.DataFrame(cls9.predict(temp_test_X))], axis=1)
# ###################################################################################################################################
# # Voting
# predictions["voting_result"] = round(predictions.mean(axis=1)).astype(int)
# voting_predictions = pd.DataFrame(predictions["voting_result"])
# gender_sub = pd.concat([test_PassengerId, voting_predictions], axis=1)
# gender_sub.to_csv(path_or_buf=r'E:\Projects\TITANIC_COMPETITION\titanic\export.csv', header=["PassengerId", "Survived"], index=False) 