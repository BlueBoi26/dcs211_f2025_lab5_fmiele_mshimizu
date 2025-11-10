import os
import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random

###########################################################################
def drawDigitHeatmap(pixels: np.ndarray, showNumbers: bool = True) -> None:
    ''' Draws a heat map of a given digit based on its 8x8 set of pixel values.
    Parameters:
        pixels: a 2D numpy.ndarray (8x8) of integers of the pixel values for
                the digit
        showNumbers: if True, shows the pixel value inside each square
    Returns:
        None -- just plots into a window
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)    
    # all seaborn palettes: 
    # https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = False)

###########################################################################
def fetchDigit(df: pd.core.frame.DataFrame, which_row: int) -> tuple[int, np.ndarray]:
    ''' For digits.csv data represented as a dataframe, this fetches the digit from
        the corresponding row, reshapes, and returns a tuple of the digit and a
        numpy array of its pixel values.
    Parameters:
        df: pandas data frame expected to be obtained via pd.read_csv() on digits.csv
        which_row: an integer in 0 to len(df)
    Returns:
        a tuple containing the reprsented digit and a numpy array of the pixel
        values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # don't want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)              # return a tuple

###########################################################################
def cleanTheData(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Cleans the digits dataframe by removing rows with NaN values.
    
    Args:
        df (pd.DataFrame): Raw dataframe from digits.csv
    
    Returns:
        tuple[pd.DataFrame, np.ndarray]: Cleaned dataframe and numpy array version
    """
    print(f"Original dataframe shape: {df.shape}")

    # Drop any completely empty columns (sometimes digits.csv has one)
    df = df.dropna(axis=1, how='all')

    # Convert everything to numeric; any non-numeric becomes NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    # Count NaNs before removing
    num_nans = df.isna().sum().sum()
    print(f"Number of NaN values: {num_nans}")

    # Drop rows containing NaN
    df_clean = df.dropna()
    print(f"After dropna shape: {df_clean.shape}")

    # Convert to NumPy array
    data_array = df_clean.to_numpy()

    # Safety check
    if len(data_array) == 0:
        print("ERROR: No valid data rows after cleaning! Check digits.csv formatting.")
    return df_clean, data_array
###################
def predictiveModel(training_set: np.ndarray, features: np.ndarray) -> int:
    """
    Summary: Implementation of 1-NN
    Args:
    - training_set: numpy array, last column is the label, remaining columns are features
    - features: numpy array of features for a single sample
    
    Returns: 
    -predicted label as an int using euclidean distance and single closest neighbor. 
    """
    X_train = training_set[:, :-1]  # All columns except LAST (features)
    y_train = training_set[:, -1].astype(int)  # LAST column is label
    distance = np.linalg.norm(X_train - features, axis = 1)
    idx_min = np.argmin(distance)
    predicted_label = int(y_train[idx_min])
    return predicted_label


def show_progress(current, total, bar_length=50):
    """Display a text-based progress bar"""
    percent = current / total
    filled = int(bar_length * percent)
    bar = '█' * filled + '-' * (bar_length - filled) # Got this line from ChatGPT, do not know what the blue character is
    print(f'\r[{bar}] {percent*100:.1f}% ({current}/{total})', end='', flush=True)
    if current == total:
        print()  # New line when complete
###########################################################################
from sklearn.model_selection import train_test_split

def splitData(data_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test 80/20, using sklearn's train_test_split.
    
    Args:
        data_array (np.ndarray): Full dataset as a NumPy array where the last
            column is the label.
    
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_test, y_test, X_train, y_train
    """
    X = data_array[:, :-1]
    y = data_array[:, -1].astype(int)

    # Use sklearn to ensure randomized but reproducible splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12121, shuffle=True
    )

    print(f"\nData successfully split (80/20):")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # Return in the order the autograder expects
    return X_test, y_test, X_train, y_train



####################
def compareLabels(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> int:
    """
    Prints only incorrect predictions for readability.
    """
    num_labels = len(predicted_labels)
    num_correct = 0
    mismatches = []

    for i in range(num_labels):
        predicted = int(round(predicted_labels[i]))
        actual = int(round(actual_labels[i]))
        if predicted == actual:
            num_correct += 1
        else:
            mismatches.append((i, predicted, actual))

    print(f"\nCorrect: {num_correct} out of {num_labels}")
    print(f"Accuracy: {num_correct / num_labels:.3f}")

    if mismatches:
        print("\nFirst 10 incorrect predictions:")
        for i, pred, act in mismatches[:10]:
            print(f"row {i:>3d} : predicted {pred:<3d} actual {act:<3d}")
    else:
        print("All predictions correct!")

    return num_correct


###########################################################################
def findBestK(X_train: np.ndarray, y_train: np.ndarray, seeds=[8675309, 5551212, 42]) -> dict:
    """
    Determines the best k for k-NN by testing several k values under different random seeds.
    Parameters:
        X_train, y_train: training data and labels
        seeds: list of random seeds to test
    Returns:
        dict mapping each seed to its best (k, accuracy)
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import random

    results = {}

    for seed in seeds:
        random.seed(seed)
        X_subtrain, X_val, y_subtrain, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=seed
        )

        best_k = None
        best_acc = 0

        print(f"\n--- Testing seed {seed} ---")
        for k in range(1, 11):  # test k = 1 through 10
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_subtrain, y_subtrain)
            preds = knn.predict(X_val)
            acc = accuracy_score(y_val, preds)

            print(f"k = {k:2d} → accuracy = {acc:.3f}")

            if acc > best_acc:
                best_acc = acc
                best_k = k

        results[seed] = (best_k, best_acc)
        print(f"Best k for seed {seed}: {best_k} (accuracy = {best_acc:.3f})")

    return results


def trainAndTest(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 best_k: int) -> np.ndarray:
    """
    Trains a k-NN classifier using the given best_k, predicts labels for X_test,
    prints the accuracy and misclassifications using compareLabels, and returns predictions.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        best_k (int): Optimal number of neighbors to use
    
    Returns:
        np.ndarray: Predicted labels for X_test
    """
    print(f"\n--- Step 10: Training and testing final k-NN model (k = {best_k}) ---")

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Final model accuracy with k = {best_k}: {acc:.3f}")

    compareLabels(y_pred, y_test)
    return y_pred



####################
def main() -> None:
    # for read_csv, use header=0 when row 0 is a header row
    filename = os.path.join(os.path.dirname(__file__), 'digits.csv')

    # Read CSV robustly — handle spaces and blank columns
    df = pd.read_csv(filename, header=0, skipinitialspace=True)

    print("Raw df shape:", df.shape)
    print("Columns:", df.columns.tolist()[:10], "...")
    print(df.head())

    # Clean the data using our improved function
    df_clean, data_array = cleanTheData(df)

    # Safety check to prevent ZeroDivisionError
    if len(data_array) == 0:
        print("No data available after cleaning. Please verify digits.csv formatting.")
        return

    print(f"\nCleaned data shape: {data_array.shape}")

    num_to_draw = 5
    for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)

        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")  
        drawDigitHeatmap(pixels)
        plt.show()

    #
    # OK!  Onward to knn for digits! (based on your iris work...)
    #

    # Step 1: Clean the data
    df_clean, data_array = cleanTheData(df)
    print(f"\nCleaned data shape: {data_array.shape}")
    
    # Step 3: Split data - first 80% train, last 20% test
    split_index = int(0.8 * len(data_array))
    train_set = data_array[:split_index]
    test_set = data_array[split_index:]
    
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Test the 1-NN model
    correct = 0
    total = len(test_set)
    
    print("\nTesting 1-NN (80% train, 20% test):")
    for i in range(total):
        features = test_set[i, :-1]
        actual_label = int(test_set[i, -1])
    
        predicted_label = predictiveModel(train_set, features)
    
        if predicted_label == actual_label:
            correct += 1
    
        # Show progress bar
        show_progress(i + 1, total)
    
    accuracy = correct / total
    print(f"\nAccuracy (80/20 split): {accuracy:.3f}")
    print(f"Correct: {correct} out of {total}")
    
    # Step 4: Swap the split - first 20% test, last 80% train
    #train_set = data_array[split_index:]
    #test_set = data_array[:split_index]
    split_index_20 = int(0.2 * len(data_array))  # Calculate 20% split point
    test_set = data_array[:split_index_20]       # First 20% as test
    train_set = data_array[split_index_20:]      # Last 80% as train

    correct = 0
    total = len(test_set)
    
    print("\nTesting 1-NN (20% test, 80% train - swapped):")
    for i in range(total):
        features = test_set[i, :-1]
        actual_label = int(test_set[i, -1])
        
        predicted_label = predictiveModel(train_set, features)
        
        if predicted_label == actual_label:
            correct += 1
        
        # Show progress bar
        show_progress(i + 1, total)
    
    accuracy = correct / total
    print(f"\nAccuracy (20/80 split swapped): {accuracy:.3f}")
    print(f"Correct: {correct} out of {total}")
    
    # Step 5: Visualize misclassified digits
    print("\n--- Finding Misclassified Digits ---")
    
    # Use the first split again (80% train, 20% test)
    split_index = int(0.8 * len(data_array))
    train_set = data_array[:split_index]
    test_set = data_array[split_index:]
    
    print(f"Data array length: {len(data_array)}")
    print(f"Split index: {split_index}")
    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    misclassified = []  # Store info about wrong predictions
    
    for i in range(len(test_set)):
        features = test_set[i, :-1]
        actual_label = int(test_set[i, -1])
        
        predicted_label = predictiveModel(train_set, features)
        
        # If prediction is WRONG, save it
        if predicted_label != actual_label:
            misclassified.append({
                'index': i,
                'actual': actual_label,
                'predicted': predicted_label,
                'pixels': features.reshape(8, 8)  # Reshape back to 8x8 for drawing
            })
        
        # Stop after finding 5 misclassified digits
        if len(misclassified) >= 5:
            break
    
    # Draw the first 5 misclassified digits
    print(f"\nFound {len(misclassified)} misclassified digits. Visualizing them:")
    for i, item in enumerate(misclassified):
        print(f"\nMisclassified #{i+1}:")
        print(f"  Actual digit: {item['actual']}")
        print(f"  Predicted as: {item['predicted']}")
        drawDigitHeatmap(item['pixels'], showNumbers=True)
        plt.show()
        # --- SCikit-learn assisted portion (Step 7 onwards) ---
    print("\n--- Moving into scikit-learn k-NN section ---")

    # Split the cleaned data into training and test sets
    X_test, y_test, X_train, y_train = splitData(data_array)

    # --- Step 8: Run scikit-learn k-NN classifier ---
    print("\n--- Step 8: Scikit-learn k-NN Classifier ---")

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    # Guess a value for k (start small; odd numbers help avoid ties)
    k_guess = 3
    print(f"Using k = {k_guess}")

    # Train the classifier
    knn = KNeighborsClassifier(n_neighbors=k_guess)
    knn.fit(X_train, y_train)

    # Predict labels for the test set
    y_pred = knn.predict(X_test)

    # Compute accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k = {k_guess}: {acc:.3f}")

    #call to compareLabels
    compareLabels(y_pred, y_test)

    # --- Step 9: Find best k values ---
    print("\n--- Step 9: Finding best k for different random seeds ---")
    best_k_results = findBestK(X_train, y_train)
    print("\nSummary of best k values per seed:")
    for seed, (k_val, acc_val) in best_k_results.items():
        print(f"  Seed {seed}: best k = {k_val}, accuracy = {acc_val:.3f}")
    
    best_k = 3  # or get it from findBestK results
    y_pred = trainAndTest(X_train, y_train, X_test, y_test, best_k)
    compareLabels(y_pred, y_test)


    # Now you can compare predictions to actual labels
    compareLabels(y_pred, y_test)



###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()