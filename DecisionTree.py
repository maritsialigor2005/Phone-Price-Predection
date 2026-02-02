from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns 

def DecisionTreeModel(i):
    # 1. Load Data
    train = pd.read_csv('CleanData/CleanTrain.csv')
    test = pd.read_csv('CleanData/CleanTest.csv')

    # 2. Slice the Data
    trainX = train.iloc[:, 1:40]
    trainY = train['price']

    testX = test.iloc[:, 1:40]
    testY = test['price']

    # --- PART 1: Hyperparameter Tuning Loop ---
    # Define ranges (Note: depth starts at 1, split starts at 2)
    depth_values = range(1, 21)
    split_values = range(2, 21)
    
    global_best_score = 0
    global_best_params = {}

    print("\n--- Tuning Decision Tree (Depth & Split) ---")
    
    # Nested loop to check every combination
    for d in depth_values:
        for s in split_values:
            # Initialize model with current depth and split
            temp_model = DecisionTreeClassifier(criterion='entropy', 
                                              max_depth=d, 
                                              min_samples_split=s,
                                              random_state=42)
            
            # 5-Fold Cross Validation
            scores = cross_val_score(temp_model, trainX, trainY, cv=5)
            mean_score = np.mean(scores)

            # Optional: Print progress (Can be spammy, maybe print only improvements)
            # print(f"Depth={d}, Split={s} -> Accuracy={mean_score:.4f}")

            # Check if this is the new best
            if mean_score > global_best_score:
                global_best_score = mean_score
                global_best_params = {
                    'max_depth': d,
                    'min_samples_split': s
                }
                # Print only when we find a better model to keep console clean
                print(f"New Best Found: Depth={d}, Split={s} -> Acc={mean_score:.4f}")

    print("---------------------------------------")
    print(f"Best Parameters: {global_best_params}")
    print(f"Best CV Score: {global_best_score*100:.2f}%")

    # --- PART 2: Final Model Training ---
    # Train the final model using the BEST params found above
    model = DecisionTreeClassifier(
        criterion='entropy', 
        max_depth=global_best_params['max_depth'], 
        min_samples_split=global_best_params['min_samples_split'],
        random_state=42
    )
    
    model.fit(trainX, trainY)
    
    # Make predictions on the test set
    y_pred = model.predict(testX)
    
    # Calculate the accuracy
    accuracy = metrics.accuracy_score(testY, y_pred)
    precision = metrics.precision_score(testY, y_pred)
    rec = metrics.recall_score(testY, y_pred)
    f1 = metrics.f1_score(testY, y_pred)
    
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")

    true_price = np.asarray(testY)[i]
    predicted_price = y_pred[i]

    print('True value: ' + ("Expensive" if true_price else "Not-Expensive"))
    print('Predicted value: ' + ("Expensive" if predicted_price else "Not-Expensive"))

    # Generate Confusion Matrix Data
    cm = metrics.confusion_matrix(testY, y_pred)

    def Graph():
        # 1. Plot the decision tree (Visualizing the BEST model)
        # Note: We limit max_depth=3 for the PLOT ONLY so it remains readable
        plt.figure(figsize=(20,10))
        plot_tree(model, filled=True, feature_names=
                [f"Feature {i}" for i in range(trainX.shape[1])],
                class_names=['Class 0', 'Class 1'], rounded=True, 
                max_depth=3, fontsize=10)
        plt.title(f"Decision Tree Visualization (Best Depth: {global_best_params['max_depth']})")
        plt.show()

        # 2. Plot the Confusion Matrix 
        plt.figure(figsize=(9, 9))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, square=True)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'Decision Tree Confusion Matrix\nParams: {global_best_params}', size=15)
        plt.show()

    results = {
        "accuracy" : f"{accuracy * 100:.2f}%",
        "precision" : f"{precision * 100:.2f}%",
        "recall" : f"{rec * 100:.2f}%",
        "f1" : f"{f1 * 100:.2f}%",
        "bestMaxDepth" : global_best_params['max_depth'],
        "bestSamplesSplit" : global_best_params['min_samples_split'],
        "prediction" : predicted_price,
        "actual" : true_price,
        "graphs" : Graph
    }

    return results