#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define numInputs 2
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4

// I - H |
//   X   O - 
// I - H |

// ============== Helper ==============
// NN
typedef struct {
    double hiddenWeights[numInputs][numHiddenNodes];
    double outputWeights[numHiddenNodes][numOutputs];
    double hiddenBias[numHiddenNodes];
    double outputBias[numOutputs];
    double hiddenLayer[numHiddenNodes];
    double outputLayer[numOutputs];
} NeuralNetwork;

// Sigmoid
double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
double dSigmoid(double x) { return x * (1.0 - x); }

// Random weights [-1,1]
double init_weight() { return ((double)rand() / RAND_MAX) * 2.0 - 1.0; }

// Knuth shuffle
void shuffle(int *array, size_t n) {
    if (n > 1) {
        for (size_t i = 0; i < n - 1; i++) {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

// Initialise weights and biases
void initialiseWeights(NeuralNetwork *nn) {
    // Hidden weights
    for (int i = 0; i < numInputs; i++) {
        for (int j = 0; j < numHiddenNodes; j++) {
            nn->hiddenWeights[i][j] = init_weight();
        }
    }
    // Hidden bias + output weights
    for (int i = 0; i < numHiddenNodes; i++) {
        nn->hiddenBias[i] = init_weight();
        for (int j = 0; j < numOutputs; j++) {
            nn->outputWeights[i][j] = init_weight();
        }
    }
    // Output bias
    for (int i = 0; i < numOutputs; i++) {
        nn->outputBias[i] = init_weight();
    }
}

// ============== Forward Pass ==============
void forwardPass(NeuralNetwork *nn, double inputs[numInputs]) {
    // Hidden layer
    for (int j = 0; j < numHiddenNodes; j++) {
        double activation = nn->hiddenBias[j];
        for (int k = 0; k < numInputs; k++) {
            activation += inputs[k] * nn->hiddenWeights[k][j];
        }
        nn->hiddenLayer[j] = sigmoid(activation);
    }

    // Output layer
    for (int j = 0; j < numOutputs; j++) {
        double activation = nn->outputBias[j];
        for (int k = 0; k < numHiddenNodes; k++) {
            activation += nn->hiddenLayer[k] * nn->outputWeights[k][j];
        }
        nn->outputLayer[j] = sigmoid(activation);
    }
}

// ============== Backpropagation ==============
void backprop(NeuralNetwork *nn, double inputs[numInputs], double targets[numOutputs], double lr) {
    double deltaOutput[numOutputs];
    double deltaHidden[numHiddenNodes];

    // Output deltas
    for (int j = 0; j < numOutputs; j++) {
        double errorOutput = targets[j] - nn->outputLayer[j];
        deltaOutput[j] = errorOutput * dSigmoid(nn->outputLayer[j]);
    }

    // Hidden deltas
    for (int j = 0; j < numHiddenNodes; j++) {
        double errorHidden = 0.0;
        for (int k = 0; k < numOutputs; k++) {
            errorHidden += deltaOutput[k] * nn->outputWeights[j][k];
        }
        deltaHidden[j] = errorHidden * dSigmoid(nn->hiddenLayer[j]);
    }

    // Update output weights and biases
    for (int j = 0; j < numOutputs; j++) {
        nn->outputBias[j] += deltaOutput[j] * lr;
        for (int k = 0; k < numHiddenNodes; k++) {
            nn->outputWeights[k][j] += nn->hiddenLayer[k] * deltaOutput[j] * lr;
        }
    }

    // Update hidden weights and biases
    for (int j = 0; j < numHiddenNodes; j++) {
        nn->hiddenBias[j] += deltaHidden[j] * lr;
        for (int k = 0; k < numInputs; k++) {
            nn->hiddenWeights[k][j] += inputs[k] * deltaHidden[j] * lr;
        }
    }
}

// ============== Printing ==============
void printResults(double inputs[numInputs], double targets[numOutputs], double outputs[numOutputs]) {
    printf("Input:%g %g  Output:%g  Expected:%g\n",
           inputs[0], inputs[1], outputs[0], targets[0]);
}

void printFinalWeights(NeuralNetwork *nn) {
    fputs("Final Hidden Weights\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numInputs; k++) {
            printf("%f ", nn->hiddenWeights[k][j]);
        }
        fputs("] ", stdout);
    }

    fputs("]\nFinal Hidden Biases\n[ ", stdout);
    for (int j = 0; j < numHiddenNodes; j++) {
        printf("%f ", nn->hiddenBias[j]);
    }

    fputs("]\nFinal Output Weights", stdout);
    for (int j = 0; j < numOutputs; j++) {
        fputs("[ ", stdout);
        for (int k = 0; k < numHiddenNodes; k++) {
            printf("%f ", nn->outputWeights[k][j]);
        }
        fputs("]\n", stdout);
    }

    fputs("Final Output Biases\n[ ", stdout);
    for (int j = 0; j < numOutputs; j++) {
        printf("%f ", nn->outputBias[j]);
    }
    fputs("]\n", stdout);
}

// ============== Evaluation ==============
void evaluateAccuracy(NeuralNetwork *nn,
                      double training_inputs[numTrainingSets][numInputs],
                      double training_outputs[numTrainingSets][numOutputs]) {
    int correct = 0;

    for (int i = 0; i < numTrainingSets; i++) {
        forwardPass(nn, training_inputs[i]);
        int predicted = (nn->outputLayer[0] > 0.5) ? 1 : 0;
        int expected  = (training_outputs[i][0] > 0.5) ? 1 : 0;

        printf("Input:%g %g  Predicted:%d  Expected:%d  (Raw: %.4f)\n",
               training_inputs[i][0], training_inputs[i][1],
               predicted, expected, nn->outputLayer[0]);

        if (predicted == expected) {
            correct++;
        }
    }

    double accuracy = (double)correct / numTrainingSets * 100.0;
    printf("\nFinal Accuracy: %.2f%% (%d/%d correct)\n",
           accuracy, correct, numTrainingSets);
}

int main(void) {
    srand(time(NULL));
    const double lr = 0.1;

    NeuralNetwork nn;
    initialiseWeights(&nn);

    // Testing XOR
    double training_inputs[numTrainingSets][numInputs] = {
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0},
        {1.0, 1.0}
    };

    double training_outputs[numTrainingSets][numOutputs] = {
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    };

    int trainingSetOrder[] = {0, 1, 2, 3};
    int numberOfEpochs = 50000;

    for (int epochs = 0; epochs < numberOfEpochs; epochs++) {
        shuffle(trainingSetOrder, numTrainingSets);

        for (int x = 0; x < numTrainingSets; x++) {
            int i = trainingSetOrder[x];

            forwardPass(&nn, training_inputs[i]);
            backprop(&nn, training_inputs[i], training_outputs[i], lr);

            // if (epochs % 1000 == 0) {
            //     printResults(training_inputs[i], training_outputs[i], nn.outputLayer);
            // }
        }
    }

    printFinalWeights(&nn);
    evaluateAccuracy(&nn, training_inputs, training_outputs);

    return 0;
}