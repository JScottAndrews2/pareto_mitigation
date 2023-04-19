import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fairlearn.datasets import fetch_adult
from fairlearn.reductions import DemographicParity, ErrorRate, GridSearch
from fairlearn.metrics import MetricFrame, selection_rate, count


if __name__ == '__main__':

    # Set parameters
    features = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation',
                'relationship', 'race', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    target = 'income'
    data = pd.read_csv('data/adult_census.csv')

    data = data.replace(to_replace={target: {" <=50K": 0, " >50K": 1}}, inplace=False)
    X = data.drop(columns=['sex'], inplace=False)
    Y = data[target]
    A = data['sex']

    X = pd.get_dummies(X)

    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    le = LabelEncoder()
    Y = le.fit_transform(Y)


    X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled, Y, A, test_size=0.4, random_state=123,
                                                                         stratify=Y)

    unmitigated_predictor = LogisticRegression(solver="liblinear", fit_intercept=True)
    unmitigated_predictor.fit(X_train, Y_train)

    unmit_metric_frame = MetricFrame(metrics={
                                        "accuracy": skm.accuracy_score,
                                        "selection_rate": selection_rate,
                                        "count": count,
                                    },
                                    sensitive_features=A_test,
                                    y_true=Y_test,
                                    y_pred=unmitigated_predictor.predict(X_test),
                                    )
    print(unmit_metric_frame.overall)
    print(unmit_metric_frame.by_group)

    # This is not working correctly
    # metric_frame.by_group.plot.bar(
    #     subplots=True,
    #     layout=[3, 1],
    #     legend=False,
    #     figsize=[12, 8],
    #     title="Accuracy and selection rate by group",
    # )


    # ---- Bias Mitigation ---- #

    sweep = GridSearch(
                        LogisticRegression(solver="liblinear", fit_intercept=True),
                        constraints=DemographicParity(),
                        grid_size=31,
                        )

    sweep.fit(X_train, Y_train, sensitive_features=A_train)

    predictors = sweep.predictors_

    # Here, the authors will manually call the metrics methods to calculate model results for the solution space. It is a
    # little clumsy, buts it's a product of how the package is set_up. The package does not offer an in-house solution to
    # generate the Pareto set from the produced solution space.
    errors, disparities = [], []
    for m in predictors:

        def classifier(X):
            return m.predict(X)

        error = ErrorRate()
        error.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)
        disparity = DemographicParity()
        disparity.load_data(X_train, pd.Series(Y_train), sensitive_features=A_train)

        errors.append(error.gamma(classifier)[0])
        disparities.append(disparity.gamma(classifier).max())

    all_results = pd.DataFrame({"predictor": predictors, "error": errors, "disparity": disparities})

    # This is the authorss binary non-dominated sort. There are better methods to use out there that can be drawn from other
    # packages and would support more than 2 outcome variables (i.e. many-objective)
    non_dominated = []
    for row in all_results.itertuples():
        errors_for_lower_or_eq_disparity = all_results["error"][
            all_results["disparity"] <= row.disparity
        ]
        if row.error <= errors_for_lower_or_eq_disparity.min():
            non_dominated.append(row.predictor)


    predictions = {"unmitigated": unmitigated_predictor.predict(X_test)}
    metric_frames = {"unmitigated": unmit_metric_frame}

    for i in range(len(non_dominated)):
        key = "dominant_model_{0}".format(i)
        predictions[key] = non_dominated[i].predict(X_test)

        metric_frames[key] = MetricFrame(
            metrics={
                "accuracy": skm.accuracy_score,
                "selection_rate": selection_rate,
                "count": count,
            },
            sensitive_features=A_test,
            y_true=Y_test,
            y_pred=predictions[key],
        )


    x = [metric_frame.overall["accuracy"] for metric_frame in metric_frames.values()]
    y = [
        metric_frame.difference()["selection_rate"]
        for metric_frame in metric_frames.values()
    ]

    keys = list(metric_frames.keys())
    plt.scatter(x, y)
    for i in range(len(x)):
        plt.annotate(keys[i], (x[i] + 0.0003, y[i]))
    plt.xlabel("accuracy")
    plt.ylabel("selection rate difference")
    plt.show()