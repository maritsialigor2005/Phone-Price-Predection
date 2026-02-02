import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def LogisticRegression(i):
    # 1. Load Data
    train = pd.read_csv('CleanData/CleanTrain.csv')
    test = pd.read_csv('CleanData/CleanTest.csv')
    
    # 2. Slice the Data
    trainX = train.iloc[:, 1:40]
    trainY = train['price']

    testX = test.iloc[:, 1:40]
    testY = test['price']

    # --- PART 1: Hyperparameter Tuning Loop ---
    # Define ranges to test
    # C: Smaller values = stronger regularization (simpler model), Larger = weaker regularization
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0] 
    
    # max_iter: How many times the solver tries to find the best fit
    iter_values = [100, 200, 500, 1000]

    global_best_score = 0
    global_best_params = {}

    print("\n--- Tuning Logistic Regression (C & max_iter) ---")

    # Nested loop to check every combination
    for c_val in C_values:
        for iter_val in iter_values:
            # Initialize model with current C and max_iter
            # solver='lbfgs' is standard, but you can add it explicitly if you want
            temp_model = linear_model.LogisticRegression(C=c_val, max_iter=iter_val)
            
            # 5-Fold Cross Validation
            # Note: You might see "ConvergenceWarning" in console if iter is too low, that is normal.
            scores = cross_val_score(temp_model, trainX, trainY, cv=5)
            mean_score = np.mean(scores)

            # Check if this is the new best
            if mean_score > global_best_score:
                global_best_score = mean_score
                global_best_params = {
                    'C': c_val,
                    'max_iter': iter_val
                }
                # Print only when we find a better model
                print(f"New Best Found: C={c_val}, max_iter={iter_val} -> Acc={mean_score:.4f}")

    print("---------------------------------------")
    print(f"Best Parameters: {global_best_params}")
    print(f"Best CV Score: {global_best_score*100:.2f}%")

    # --- PART 2: Final Model Training ---
    # Train the final model using the BEST params found above
    model = linear_model.LogisticRegression(
        C=global_best_params['C'], 
        max_iter=global_best_params['max_iter']
    )

    model.fit(trainX, trainY)
    prediction = model.predict(testX)

    # --- Metrics ---
    accuracy = metrics.accuracy_score(np.asarray(testY), prediction)
    precision = metrics.precision_score(np.asarray(testY), prediction)
    rec = metrics.recall_score(np.asarray(testY), prediction)
    f1 = metrics.f1_score(np.asarray(testY), prediction)

    print(f'\nFinal Accuracy on Test Set: {accuracy * 100:.2f}%')

    true_price = np.asarray(testY)[i]
    predicted_price = prediction[i]

    print('True value: ' + ("Expensive" if true_price else "Not-Expensive"))
    print('Predicted value: ' + ("Expensive" if predicted_price else "Not-Expensive"))

    # Generate Confusion Matrix
    cm = metrics.confusion_matrix(testY, prediction)
    

    def Graph():
        # Plot the Heatmap
        plt.figure(figsize=(9, 9))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, square=True)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Logistic Regression Confusion Matrix\nParams: {global_best_params}', size=15)

        plt.show()

    results = {
        "accuracy" : f"{accuracy * 100:.2f}%",
        "precision" : f"{precision * 100:.2f}%",
        "recall" : f"{rec * 100:.2f}%",
        "f1" : f"{f1 * 100:.2f}%",
        "bestC" : global_best_params['C'],
        "bestMaxIter" : global_best_params['max_iter'],
        "prediction" : predicted_price,
        "actual" : true_price,
        "graphs" : Graph
    }

    return results