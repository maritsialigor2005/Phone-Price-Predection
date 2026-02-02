import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def KNN(i):
    # 1. Load the Data
    train = pd.read_csv('CleanData/CleanTrain.csv')
    test = pd.read_csv('CleanData/CleanTest.csv')
    
    # 2. Slice the Data
    trainX = train.iloc[:, 1:40]
    trainY = train['price']

    testX = test.iloc[:, 1:40]
    testY = test['price']

    # --- PART 1: Hyperparameter Tuning Loop (Reference Style) ---
    k_values = range(1, 21)
    
    # We will test these parameters for every K
    weight_options = ['uniform', 'distance']
    metric_options = ['euclidean', 'manhattan']

    # List to store the BEST accuracy found for each K (for the graph)
    mean_accuracies = []
    
    # Variables to track the absolute best model found
    global_best_score = 0
    global_best_params = {}

    print("\n--- Cross-Validation Debug Output ---")
    
    for k in k_values:
        # Track the best score just for this specific K (to put on the graph)
        best_score_for_this_k = 0
        
        for metric in metric_options:
            for weight in weight_options:
                # Initialize model with current combination
                model = KNeighborsClassifier(n_neighbors=k, weights=weight, metric=metric)
                
                # 5-fold cross-validation (From Reference Code)
                scores = cross_val_score(model, trainX, trainY, cv=5)
                mean_score = np.mean(scores)
                
                # PRINT: Debug line
                print(f"k={k}, metric={metric}, weights={weight} -> Accuracy={mean_score:.4f}")

                # Update "Best for THIS K" (for the graph)
                if mean_score > best_score_for_this_k:
                    best_score_for_this_k = mean_score
                
                # Update "Global Best" (for the final prediction)
                if mean_score > global_best_score:
                    global_best_score = mean_score
                    global_best_params = {
                        'n_neighbors': k,
                        'weights': weight,
                        'metric': metric
                    }

        # Append the best score achieved for this K to the list
        mean_accuracies.append(best_score_for_this_k)
        
    print("---------------------------------------")
    print(f"Best Parameters Found: {global_best_params}")
    print(f"Best CV Score: {global_best_score:.4f}")

    # --- PART 2: Final Model Training ---
    # Train the final model using the GLOBAL BEST params found above
    final_model = KNeighborsClassifier(
        n_neighbors=global_best_params['n_neighbors'],
        weights=global_best_params['weights'],
        metric=global_best_params['metric']
    )
    final_model.fit(trainX, trainY)
    
    # Predict on Test set
    prediction = final_model.predict(testX)

    # --- Metrics ---
    accuracy = metrics.accuracy_score(np.asarray(testY), prediction)
    precision = metrics.precision_score(np.asarray(testY), prediction)
    rec = metrics.recall_score(np.asarray(testY), prediction)
    f1 = metrics.f1_score(np.asarray(testY), prediction)

    print(f'\nFinal Accuracy on Test Set: {accuracy * 100:.2f}%')

    # Specific Prediction Check
    true_price = np.asarray(testY)[i]
    predicted_price = prediction[i]

    print('True value: ' + ("Expensive" if true_price else "Not-Expensive"))
    print('Predicted value: ' + ("Expensive" if predicted_price else "Not-Expensive"))

    # Generate Confusion Matrix
    cm = metrics.confusion_matrix(testY, prediction)

    def Graph():
        # --- PART 3: Plotting (Reference Code Style) ---
        # Create a wider plot (Width = 10, Height = 4)
        plt.figure(figsize=(10, 4))
        
        # Plot the accuracies vs. K values
        plt.plot(k_values, mean_accuracies, marker='o', color='b')
        plt.xlabel('K Value') 
        plt.ylabel('Mean Accuracy')
        plt.title('Accuracy vs. K Value (Best Config per K)')
        
        # Set the x-ticks to display the K values clearly
        plt.xticks(k_values)
        
        # Annotate the mean accuracy values on the plot
        for idx, mean_acc in enumerate(mean_accuracies):
            plt.text(k_values[idx], mean_acc, 
                     f'{mean_acc:.3f}', 
                     ha='center', va='bottom', fontsize=8)
        
        # Show the plot
        plt.grid(True)
        plt.show()

        # Plot the Heatmap (Standard)
        plt.figure(figsize=(9, 9)) 
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, square=True)

        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(f'KNN Confusion Matrix\n{global_best_params}', size=12)

        plt.show()

    results = {
        "accuracy" : f"{accuracy * 100:.2f}%",
        "precision" : f"{precision * 100:.2f}%",
        "recall" : f"{rec * 100:.2f}%",
        "f1" : f"{f1 * 100:.2f}%",
        "bestK" : global_best_params['n_neighbors'],
        "bestWeight" : global_best_params['weights'],
        "bestMetric" : global_best_params['metric'],
        "bestCV" : f"{global_best_score* 100:.2f}%",
        "prediction" : predicted_price,
        "actual" : true_price,
        "graphs" : Graph
    }

    return results