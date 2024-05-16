package main

import (
    "encoding/csv"
    "fmt"
    "log"
    "os"
    "strconv"
	"math"
	"math/rand"
	"time"
	"os/exec"
)

const (
	numEpochs = 5000
	numIn = 4 // number of input variables
	numHidden = 4 // number of hidden neurons
	numOut = 1 // output dimension
	learningRate = 0.075
)

type HiddenLayer struct {
	Weights [][]float64
	Biases []float64
}

type OutputLayer struct {
	Weights [][]float64
	Bias float64
}

type SeriesData struct {
    Date string
	Open float64
	High float64
	Low float64
    Close float64
	DayRange float64
	PercentReturn float64
	PointDelta float64
	Signal float64
}

func sigmoid(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
    return x * (1.0 - x)
}

func forwardHidden(inputs []float64, hidden *HiddenLayer) []float64 {
	hiddenOutputs := make([]float64, numHidden)
	for i := 0; i < numHidden; i++ {
		weightedSum := hidden.Biases[i]
		for j := 0; j < numIn; j++ {
			weightedSum += inputs[j] * hidden.Weights[i][j]
		}
		hiddenOutputs[i] = sigmoid(weightedSum)
	}
	return hiddenOutputs
}

func forwardOutput(hiddenOutputs []float64, output *OutputLayer) float64 {
	weightedSum := output.Bias
	for i := 0; i < numHidden; i++ {
		weightedSum += hiddenOutputs[i] * output.Weights[0][i]
	}
	return sigmoid(weightedSum)
}

func backpropagate(inputs []float64, hidden *HiddenLayer, output *OutputLayer, target float64) {
	hiddenOutputs := forwardHidden(inputs, hidden)
	predictedOutput := forwardOutput(hiddenOutputs, output)
	predErr := target - predictedOutput
	outputDelta := predErr * sigmoidDerivative(predictedOutput)
	for i := 0; i < numHidden; i++ {
		output.Weights[0][i] += learningRate * outputDelta * hiddenOutputs[i]
	}
	output.Bias += learningRate * outputDelta
	for i := 0; i < numHidden; i++ {
		hiddenDelta := outputDelta * output.Weights[0][i] * sigmoidDerivative(hiddenOutputs[i])
		for j := 0; j < numIn; j++ {
			hidden.Weights[i][j] += learningRate * hiddenDelta * inputs[j]
		}
		hidden.Biases[i] += learningRate * hiddenDelta
	}
}

func randomWeights(rows, cols int) [][]float64 {
    weights := make([][]float64, rows)
    for i := range weights {
        weights[i] = make([]float64, cols)
        for j := range weights[i] {
            weights[i][j] = rand.Float64() - 0.5
        }
    }
    return weights
}

func xavierGlorotWeights(rows, cols int) [][]float64 {
	weights := make([][]float64, rows)
	scale := math.Sqrt(1.0 / float64(rows+cols))
	for i := range weights {
		weights[i] = make([]float64, cols)
		for j := range weights[i] {
			weights[i][j] = rand.Float64() * scale
		}
	}
	return weights
}

func randomBiases(size int) []float64 {
    biases := make([]float64, size)
    for i := range biases {
        biases[i] = rand.Float64() - 0.5
    }
    return biases
}

/*
type NeuralNetwork struct {
	numHidden int
    weightsInput [][]float64
    biasesHidden []float64
    weightsHidden [][]float64
    biasOutput float64
}

func newNN(nHidden int) *NeuralNetwork {
    rand.Seed(time.Now().UnixNano())
    nn := &NeuralNetwork{
		numHidden: nHidden,
        weightsInput: xavierGlorotWeights(nHidden, numIn),
		// weightsInput: randomWeights(numHidden, numIn),
        biasesHidden: randomBiases(nHidden),
        weightsHidden: xavierGlorotWeights(numOut, nHidden),
        // weightsHidden: randomWeights(numOut, numHidden),
		biasOutput: rand.Float64() - 0.5,
    }
    return nn
}

func (nn *NeuralNetwork) forwardHidden(inputs []float64) []float64 {
    hiddenOutputs := make([]float64, nn.numHidden)
    for i := 0; i < nn.numHidden; i++ {
        weightedSum := nn.biasesHidden[i]
        for j := 0; j < numIn; j++ {
            weightedSum += inputs[j] * nn.weightsInput[i][j]
        }
        hiddenOutputs[i] = sigmoid(weightedSum)
    }
    return hiddenOutputs
}

func (nn *NeuralNetwork) forwardOutput(hiddenOutputs []float64) float64 {
    weightedSum := nn.biasOutput
    for i := 0; i < nn.numHidden; i++ {
        weightedSum += hiddenOutputs[i] * nn.weightsHidden[0][i]
    }
    return sigmoid(weightedSum)
}

func (nn *NeuralNetwork) backpropagate(inputs []float64, target, learningRate float64) {
    hiddenOutputs := nn.forwardHidden(inputs)
    predictedOutput := nn.forwardOutput(hiddenOutputs)
    outputError := target - predictedOutput
    outputDelta := outputError * sigmoidDerivative(predictedOutput)
    for i := 0; i < nn.numHidden; i++ {
        nn.weightsHidden[0][i] += learningRate * outputDelta * hiddenOutputs[i]
    }
    nn.biasOutput += learningRate * outputDelta
    for i := 0; i < nn.numHidden; i++ {
        hiddenDelta := outputDelta * nn.weightsHidden[0][i] * sigmoidDerivative(hiddenOutputs[i])
        for j := 0; j < numIn; j++ {
            nn.weightsInput[i][j] += learningRate * hiddenDelta * inputs[j]
        }
        nn.biasesHidden[i] += learningRate * hiddenDelta
    }
} */

func parseFloat(value string) float64 {
    result, err := strconv.ParseFloat(value, 64)
    if err != nil {
        log.Fatal(err)
    }
    return result
}

func getFeatures(records [][]string) []SeriesData {
	data := make([]SeriesData, 0, len(records) - 1)
    for i := 1; i < len(records); i++ {
		closePrice := parseFloat(records[i][5])
		var nextChange float64
		var signal float64
		var percent float64
		var change float64
		if i > 2 {
			if i == len(records) - 1 {
				signal = -1.0
			} else {
				nextClose := parseFloat(records[i+1][5])
				nextChange = nextClose - closePrice
				if nextChange >= 0.0 {
					signal = 1.0
				} else {
					signal = 0.0
				}
			}
			prevClose := parseFloat(records[i-1][5])
			percent = (closePrice - prevClose) / prevClose
			change = closePrice - prevClose
		}
        entry := SeriesData{
            Date: records[i][1],
			Open: parseFloat(records[i][2]),
			High: parseFloat(records[i][3]),
			Low: parseFloat(records[i][4]),
            Close: closePrice,
			DayRange: math.Abs(parseFloat(records[i][3]) - parseFloat(records[i][4])),
			PercentReturn: percent,
			PointDelta: change,
			Signal: signal, 
        }
        data = append(data, entry) 
    }
	return data
}

func main() {
	startTime := time.Now()
	cmd := exec.Command("python", "fetch_data.py", os.Args[1])
	output, err := cmd.Output()
	if err != nil {
		fmt.Println("Error returning Python script:", err)
	}
	fmt.Println("Python script output:\n"+string(output)+"\t-> Loading python output .csv into main.go")
    file, err := os.Open("ohlc_data/"+os.Args[1]+"_tseries.csv")
    if err != nil {
        fmt.Printf("Failed to load OHLC .csv file: %v", err)
    }
    defer file.Close()
    reader := csv.NewReader(file)
    records, err := reader.ReadAll()
    if err != nil {
        fmt.Printf("Failed to read OHLC .csv file: %v", err)
    }
	/* Feature Detection; finding desired correct output data (the buy/sell signals) to train our model on */
	lastTime := time.Now()
	data := getFeatures(records)
	prepTime := time.Now().Sub(lastTime).Nanoseconds()
	fmt.Printf("\n* <main.go> Feature Detection / Preprocessing Time: %v nanoseconds (%v seconds)\n", prepTime, float64(prepTime) / 1e9)
    for i := 0; i < len(data); i++ {
		fmt.Printf("%1.0f signal on %s:\n\t OHLC: ($%.2f, $%.2f, $%.2f, $%.2f) Day Range: $%.2f Day Point Delta: $%.2f\n", data[i].Signal, data[i].Date, data[i].Open, data[i].High, data[i].Low, data[i].Close, data[i].DayRange, data[i].PointDelta)
    }
	/* Initializing the NN and training */
	trainingData := make([][]float64, len(data) - 1)
	targets := make([]float64, len(data) - 1)
	for i := 0; i < len(data) - 1; i++ {
		trainingData[i] = []float64{data[i].Close, data[i].DayRange, data[i].PointDelta, data[i].Signal}
		targets[i] = data[i+1].PointDelta
	}
	hiddenLayer := HiddenLayer{
		Weights: xavierGlorotWeights(numHidden, numIn),
		Biases: randomBiases(numHidden),
	}
	outputLayer := OutputLayer{
		Weights: xavierGlorotWeights(numOut, numHidden),
		Bias: rand.Float64() - 0.5,
	}
	/* Training */
	for epoch := 0; epoch < numEpochs; epoch++ {
		totalError := 0.0
		for i := 0; i < len(data) - 1; i++ {
			hiddenOutputs := forwardHidden(trainingData[i], &hiddenLayer)
			backpropagate(trainingData[i], &hiddenLayer, &outputLayer, targets[i])
			thisError := targets[i] - forwardOutput(hiddenOutputs, &outputLayer)
			totalError += thisError * thisError
		}
		fmt.Printf("Epoch %d Total Error: %f\n", epoch, totalError)
	}
	/* Testing */
	for i := 0; i < len(data) - 1; i++ {
		hiddenOutputs := forwardHidden(trainingData[i], &hiddenLayer)
		prediction := forwardOutput(hiddenOutputs, &outputLayer)
		fmt.Printf("Input (%s): [Close, Day Range, Point Delta, Signal] = %v; Target Point Delta: %.2f; Predicted Point Delta: %f\n", data[i].Date,trainingData[i], targets[i], prediction)
	}
	executionTime := time.Now().Sub(startTime).Nanoseconds()
	fmt.Printf("\n<main.go> Total Execution Time: %v nanoseconds (%v seconds)\n", executionTime, float64(executionTime) / 1e9)
}
/*
func oldTrain(data []SeriesData) {
	n := len(data) - 1
	learningRate := 1e-16 // 1e-08 and smaller learningRates start to converge to similar values
	for window := 2; window < 7; window++ {
		startIndex := n - window
		for numHidden := 1; numHidden < 9; numHidden++ {
			nn := newNN(numHidden)
			lastTime = time.Now()
			for epoch := 0; epoch < numEpochs; epoch++ {
				for i := startIndex; i < n; i++ {
					trainingInput := []float64{data[i-1].Close, data[i-1].DayRange, data[i-1].PointDelta, data[i-1].PercentReturn, data[i].Close, data[i].DayRange, data[i].PointDelta, data[i].Signal}
					trainingTarget := data[i+1].PointDelta
					nn.backpropagate(trainingInput, trainingTarget, learningRate)
					trainingPrediction := nn.forwardOutput(nn.forwardHidden(trainingInput))
					fmt.Printf("Training Epoch %d Prediction %d: %f\n", epoch, i, trainingPrediction) 
				}
			}
			trainingTime := time.Now().Sub(lastTime).Nanoseconds()
			fmt.Printf("\n<main.go> (window, numHidden) = (%d, %d) TRAINING TIME: %v nanoseconds (%v seconds)\n", window, numHidden, trainingTime, float64(trainingTime) / 1e9)
			testInput := []float64{data[n-1].Close, data[n-1].DayRange, data[n-1].PointDelta, data[n-1].PercentReturn, data[n].Close, data[n].DayRange, data[n].PointDelta, data[n].Signal}
			fmt.Printf("  :: (numEpochs, window, numHidden, learningRate) = (%d, %d, %d, %.e) TEST RESULTS", numEpochs, numHidden, window, learningRate)
			testPrediction := nn.forwardOutput(nn.forwardHidden(testInput))
			fmt.Printf("\n\t-> %s Next Trading Day Point Delta Prediction: $%.5f (%.5f%%)\n", os.Args[1], testPrediction, (testPrediction / data[n].Close)*100)
		}
		fmt.Println("\n------------------------------------------------------------------------------------------------")
	}
}
*/
/*
	// MA Crossover Signals Test
    shortWindow := 3
    longWindow := 5
	currentSignal := ""
	currentDate := ""
	for i := longWindow; i < len(data); i++ {
		shortAvg := 0.0
        longAvg := 0.0
        for j := i - shortWindow + 1; j <= i; j++ {
            shortAvg += data[j].Close
        }
        shortAvg /= float64(shortWindow)
        for j := i - longWindow + 1; j <= i; j++ {
            longAvg += data[j].Close
        }
        longAvg /= float64(longWindow)
		// Detect signal from moving averages
        var signal string
        if shortAvg > longAvg {
            signal = "Buy"
        } else if shortAvg < longAvg {
            signal = "Sell"
        }
        // don't repeat the same signal
        if signal != currentSignal {
            currentSignal = signal
            currentDate = data[i].Date
            fmt.Printf("%s signal at date: %s\n", signal, currentDate)
        }
	}
*/
