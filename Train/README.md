# VoxOracle Sentinel

## Training The Model

### Train (train.py)

This Python script is designed to train and evaluate machine learning models for image classification tasks. It has several components and functionalities, including:

1. **Environment Setup**: The script sets up the necessary environment variables for GPU usage (CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER).

2. **Data Preparation**: The script uses the `DataPreparation` class from the `Train.data_proc` module to read and process the training, validation, and test data from CSV files. The data is loaded into generators for efficient memory usage during training.

3. **Model Building**: The script uses the `ModelBuilder` class from the `Train.dense_net` module to create a DenseNet121 base model. The model can be customized with different attention mechanisms (`NormalAttentionLayer`, `GAAPAttentionLayer`) and softmax strategies (`RSoftmax`).

4. **Training and Evaluation**: The script utilizes the `TrainingManager` and `EvaluationManager` classes from the `Train.managers` module to handle the training and evaluation processes. The `TrainingManager` trains the model with the specified hyperparameters (epochs, batch size) and saves checkpoints to a specified directory. The `EvaluationManager` evaluates the trained model on the test dataset.

5. **Command-line Arguments**: The script accepts several command-line arguments to configure various aspects of the training process, such as the number of classes, input shape, checkpoint directory, log directory, and dataset paths.

To use this script, you can run it from the command line with the following arguments:

```bash
python train.py --logdir --epoch --b_size  --classes  --checkpoint_dir --train_csv --valid_csv --test_csv --in_shape --initial_sparsity_rate --softmax_type --attention_type
```

Replace the placeholders with the appropriate values for your use case.

- `--logdir`: Path to the directory where logs will be saved.
- `--epoch`: Number of training epochs (default: 100).
- `--b_size`: Batch size for training (default: 16).
- `--classes`: Number of classes in the dataset.
- `--checkpoint_dir`: Path to the directory where model checkpoints will be saved.
- `--train_csv`, `--valid_csv`, `--test_csv`: Paths to the CSV files containing the training, validation, and test datasets, respectively.
- `--in_shape`: Input shape for the model (default: '128,211').
- `--initial_sparsity_rate`: Initial sparsity rate for the R_Softmax strategy (default: 0.5).
- `--softmax_type`: Softmax strategy to use ('normal' or 'r_softmax'; default: 'normal').
- `--attention_type`: Attention mechanism to use ('normal', 'gaap', or 'no_attention'; default: 'no_attention').

After running the script with the appropriate arguments, it will start the training process and save the model checkpoints in the specified directory. Once the training is complete, the script will evaluate the trained model on the test dataset.

---

### Train Managers (managers.py)

This Python script handles the training and evaluation of machine learning models using TensorFlow and Keras. It includes two main classes: `TrainingManager` and `EvaluationManager`.

1. **TrainingManager**:
   - Responsible for setting up callbacks, compiling the model, and running the training process.
   - Initializes with the following parameters:
     - `model`: The machine learning model to be trained.
     - `checkpoint_dir`: The directory where checkpoint files will be saved.
     - `logdir`: The directory for storing logs.
     - `train_data`: The training data generator.
     - `valid_data`: The validation data generator.
     - `test_data`: The test data generator.
   - Sets up various callbacks:
     - `ModelCheckpoint`: Saves the model checkpoint with the best validation accuracy.
     - `TestSetEvaluationCallback`: Evaluates the model on the test dataset and logs the results.
     - `CustomCheckpoint`: A custom callback to save model checkpoints with specific filenames.
     - `TensorBoard`: Creates a TensorBoard log for visualizing the training process.
     - `StopAtAccuracy`: Stops the training process if a specific validation accuracy is reached.
   - The `train()` method compiles the model with the Adam optimizer, categorical cross-entropy loss, and accuracy metric, and then fits the model to the training data while using the validation data for evaluation. The method passes all the defined callbacks to the model's `fit()` method.

2. **EvaluationManager**:
   - Responsible for evaluating the trained model's performance on the test dataset.
   - Initializes with the following parameters:
     - `model`: The trained machine learning model.
     - `test_set`: The test data generator.
   - The `evaluate()` method calls the model's `evaluate()` function with the test data generator and prints the test loss and accuracy.

To use this script, you need to provide the necessary paths and data generators for training, validation, and testing. You can create instances of the `TrainingManager` and `EvaluationManager` classes, passing in the required parameters, and then call the `train()` and `evaluate()` methods, respectively.

---

### Train Handlers (train_handlers.py)

1. **CustomCheckpoint Strategy**:  
- Want to save models frequently with extra metadata like accuracy   
- Override ModelCheckpoint's on_epoch_end to trigger custom save
- Templatize path to include useful epoch and accuracy numbers

- ***Key Functions***:
  - `__init__` : Initialize path and monitor metric 
  - on_epoch_end: Called every epoch. Gets current metrics, checks if improved, formats custom path with numbers, calls save.
  - _save_model: Handles actually saving model weights to custom path

2. **TestSetEvaluationCallback Strategy**:
- Want to check metrics on test set periodically during training
- Use multiple callbacks at once in model.fit without disrupting main loop

- ***Key Functions***: 
  - `__init__`: Store test data, log directories, frequency hyperparams
  - on_epoch_end: If correct epoch, run evaluate() on test set
  - Logs using TensorBoard summary writer and text file 

3. **StopAtAccuracy Strategy**:  
- Early stop training when reach desired metric threshold 
- Clean way to encapsulate stopping logic separately  

- ***Key Functions***:
  - on_epoch_end: Check if val_acc meets threshold  
  - If yes, print message and set flag to stop model.fit gracefully

---

### Data Processing Method (data_proc.py)

This DataPreparation class inherits from Keras's Sequence class to enable loading and batching data for model training in a memory efficient way.

Some key things it does:

- Initializes with metadata like **_csv file_**, **_batch size_**, **_image dimensions_** etc. 
- Reads the csv data file 
- Encodes the text labels to integers
- Defines `__len__` and `__getitem__` to enable Sequence interface  
- Shuffles data indexes after each epoch
- Handles batch slicing and data generation/preprocessing inside `__getitem__` :
  - Loads images 
  - Preprocesses with DenseNet functions
  - One-hot encodes labels

This allows fitting a TF/Keras model seamlessly while handling data pipelines under the hood.

---

### Attention Mechanisms (attention.py)

This code defines two custom attention layers:

1. **NormalAttentionLayer**:

This layer implements a basic form of attention mechanism. It creates trainable weights `W` and biases `b`. In its forward pass (`call` method), it applies a `tanh` activation function to the dot product of the input `x` and the weights `W`, adds the bias `b`, and then applies a softmax function. This results in an attention score (`a`) for each element in the input. The output is the element-wise multiplication of the input `x` and the attention scores, effectively allowing the layer to emphasize certain features in the input.
- Implements simple feedforward attention 
- Uses a learned weight matrix W and bias b
- Computes attention scores e with tanh on dot product with input
- Applies softmax to get attention weights
- Returns input multiplied by attention weights  

- **Key Methods**:
  - build: Creates trainable weight and bias variables
  - call: Implements the attention calculations 

2. **GAAPAttentionLayer**:

This layer represents a more complex attention mechanism - the [Global Aerage Attention Pooling](https://thesai.org/Downloads/Volume13No7/Paper_96-Learning_Global_Average_Attention_Pooling_GAAP.pdf), specifically designed for processing sequences (e.g., time-series data or audio signals). It defines convolutional layers for query (`Q`), key (`K`), and value (`V`) transformations, essential components of the attention mechanism. It computes attention scores by applying a softmax function to the dot product of the query and key. These scores are then used to weigh the values. A unique aspect of this layer is the inclusion of a learnable `alpha` parameter, which is used to scale the output of the attention mechanism. This scaling factor is constrained between 0 and 1 and is learnable, allowing the model to adapt the influence of the attention mechanism during training. The layer concludes with a global average pooling operation, making it suitable for tasks that require summarizing sequential data into a fixed-size representation.

Both layers are versatile and can be integrated into various neural network architectures, enhancing their ability to focus on relevant features in the input data. They are especially useful in tasks where the identification and weighting of important input features significantly impact model performance, such as in time-series analysis, natural language processing, and complex signal processing tasks.
- Implements convolutional self-attention, known as GAAP
- Had three 1D convolution layers to get Query, Key, Value matrices
- Computes attention scores and weights like simple attention
- Returns input concatenated with the attended value vectors

- **Key Methods**: 
  - build: Defines the convolutional layers  
  - call: Passes input through conv layers to generate QKV matrices and compute attention

So in summary, this encapsulate two attention mechanisms into reusable Keras layers using standard Layer subclassing.

---

### R-Softmax Mechanisms (r_softmax.py)

VoxOracle-Sentinel introduces the innovative [R-Softmax](https://www.researchgate.net/publication/369945900_r-softmax_Generalized_Softmax_with_Controllable_Sparsity_Rate) function into its neural network architecture, a strategic move designed to tackle the inherent sparsity in audio data. In the landscape of audio classification, the sparsity challenge is characterized by the presence of only a select group of meaningful audio signals among a vast spectrum of potential inputs. Traditional Softmax functions often fall short in representing this sparse nature, leading to less discriminative models that cannot precisely distinguish between similar sounding phonetic nuances.

The R-Softmax function brings a game-changing regularization aspect to the traditional Softmax, promoting a model that is more selective and focused in its predictions. This is essential for the accurate interpretation of Persian speech commands, which require the model to discern subtle phonetic distinctions. By leveraging R-Softmax, VoxOracle-Sentinel stands out for its ability to deliver high-precision voice command detection, ensuring each command is captured with remarkable clarity, even in scenarios where data sparsity is prominent.

**Strategy**:
- R-Softmax induces sparsity in the activations through a threshold
- Learns a sparsity rate between 0 and 1
- Applies threshold based on sparsity rate 

- **Key Methods**:

  - `__init__`  Initialize desired sparsity rate
  build: 
  - Adds sparsity rate variable
  - Constrains it between 0 and 1

- **call: Performs R-Softmax calculation**
  - Sort inputs to get threshold value based on sparsity rate 
  - Calculate threshold t_r for each example
  - Compute R-Softmax formula 

So it makes the sparsity rate trainable, then implements the thresholding and re-weighting in call(). This allows the model to learn how much sparsity is useful in the activations. It can induce various levels of sparsity.

---

### Dense Net Model (dense_net.py)

The ModelBuilder class uses the **_singleton pattern_** to build Keras models for image classification. It abstracts the model architecture to enable easy customization.

The strategy used:
- Subclass DenseNet base model to inherit powerful CNN feature extraction capabilities
- Modularly add different attention mechanisms and sparse activations
- Handles weight loading, global pooling etc.   

- **Key methods**:
  - `__init__` Initialize by loading pretrained DenseNet without the classifier layers
  - **_build_model_** - Construct a full model:
    - Append attention layer, pooling, dense layers
    - Apply R-softmax or regular softmax 
    - Build the final Model with updated inputs/outputs

By keeping the base feature extractor consistent, different attention and sparsity modules can be plugged in to experiment with accuracy vs efficiency tradeoffs. This makes iterating on model architectures and hyperparameters cleaner.
