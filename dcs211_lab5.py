import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
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
    # Remove rows with any NaN values
    df_clean = df.dropna()
    
    # Convert to numpy array
    data_array = df_clean.to_numpy()
    
    return df_clean, data_array

###################
def predictiveModel(training_set: np.ndarray, features: np.ndarray) -> int:
    """
    Summary: Implementation of 1-NN
    Args:
    - training_set: numpy array, first column is the label, remaining colums are features
    - features: numpy array of features for a single sample
    
    Returns: 
    -predicted label as an int using euclidean distance and single closest neighbor. 
    """
    X_train = training_set[:, 1:] #all columns except first which is the label, extract training features
    y_train = training_set[:, 0].astype(int) # extract training labels
    distance = np.linalg.norm(X_train - features, axis = 1) # Euclidean distances to all training points
    idx_min = np.argmin(distance)      # index of closest training sample
    predicted_label = int(y_train[idx_min]) # label of nearest neighbor
    return predicted_label####################3
def main() -> None:
    # for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe...")

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
        # Get features (all columns except last) and actual label (last column)
        features = test_set[i, :-1]
        actual_label = int(test_set[i, -1])
        
        # Predict using 1-NN
        predicted_label = predictiveModel(train_set, features)
        
        if predicted_label == actual_label:
            correct += 1
        
        # Simple progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total}")
    
    accuracy = correct / total
    print(f"\nAccuracy (80/20 split): {accuracy:.3f}")
    print(f"Correct: {correct} out of {total}")
    
    # Step 4: Swap the split - first 20% test, last 80% train
    train_set = data_array[split_index:]
    test_set = data_array[:split_index]
    
    correct = 0
    total = len(test_set)
    
    print("\nTesting 1-NN (20% test, 80% train - swapped)...")
    for i in range(total):
        features = test_set[i, :-1]
        actual_label = int(test_set[i, -1])
        
        predicted_label = predictiveModel(train_set, features)
        
        if predicted_label == actual_label:
            correct += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total}")
    
    accuracy = correct / total
    print(f"\nAccuracy (20/80 split swapped): {accuracy:.3f}")
    print(f"Correct: {correct} out of {total}")
###############################################################################
# wrap the call to main inside this if so that _this_ file can be imported
# and used as a library, if necessary, without executing its main
if __name__ == "__main__":
    main()
