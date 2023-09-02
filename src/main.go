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
	learningRate = 0.01
	numEpochs = 1000
	numIn = 2
	numHidden = 5
	numOut = 1
)

type SeriesData struct {
    Date string
	Open float64
	High float64
	Low float64
    Close float64
	DayRange float64
	PercentReturn float64
	PointDelta float64
	Signal string
}

func parseFloat(value string) float64 {
    result, err := strconv.ParseFloat(value, 64)
    if err != nil {
        log.Fatal(err)
    }
    return result
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

func randomBiases(size int) []float64 {
	biases := make([]float64, size)
	for i := range biases {
		biases[i] = rand.Float64() - 0.5
	}
	return biases
}

func transpose(mat [][]float64) [][]float64 {
    rows, cols := len(mat), len(mat[0])
    transposed := make([][]float64, cols)
    for i := 0; i < cols; i++ {
        transposed[i] = make([]float64, rows)
        for j := 0; j < rows; j++ {
            transposed[i][j] = mat[j][i]
        }
    }
    return transposed
}

func add(a []float64, b []float64) []float64 {
	if len(a) != len(b) {
		log.Fatal("add() Mismatched dimensions: slices a and b")
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func sub(a []float64, b []float64) []float64 {
	if len(a) != len(b) {
		log.Fatal("sub() Mismatched dimensions: slices a and b")
	}
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] - b[i]
	}
	return result
}

func addWeights(a [][]float64, b [][]float64) [][]float64 {
    if len(a) != len(b) || len(a[0]) != len(b[0]) {
        log.Fatal("Mismatched dimensions: weight matrices a and b")
    }
    result := make([][]float64, len(a))
    for i := range a {
        result[i] = make([]float64, len(a[0]))
        for j := range a[i] {
            result[i][j] = a[i][j] + b[i][j]
        }
    }
    return result
}

func mult(a [][]float64, b [][]float64) [][]float64 {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		log.Fatal("mutl() Mismatched dimensions: matricies a and b")
	}
	result := make([][]float64, len(a))
	for i := range a {
		result[i] = make([]float64, len(a[0]))
		for j := range a[i] {
			result[i][j] = a[i][j] * b[i][j]
		}
	}
	return result
}

func dot(vec []float64, mat [][]float64) []float64 {
	if len(vec) != len(mat[0]) {
		log.Fatal("dot() Mismatched dimensions: vector and matrix")
	}
	result := make([]float64, len(mat))
	for i := range mat {
		for j := range vec {
			result[i] += vec[j] * mat[i][j]
		}
	}
	return result
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

type NeuralNetwork struct {
	inputSize int
	hiddenSize int
	outputSize int
	weightsInput [][]float64
	weightsHidden [][]float64
	biasesHidden []float64
	biasesOutput []float64
	activation func(float64) float64
	derivative func(float64) float64
}

func newNN(inputSize, hiddenSize, outputSize int) *NeuralNetwork {
	rand.Seed(time.Now().UnixNano())
	nn := &NeuralNetwork{
		inputSize: inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
		weightsInput: randomWeights(inputSize, hiddenSize),
		weightsHidden: randomWeights(hiddenSize, outputSize),
		biasesHidden: randomBiases(hiddenSize),
		biasesOutput: randomBiases(outputSize),
		activation: sigmoid,
		derivative: sigmoidDerivative,
	}
	return nn
}

func applyActivation(layer []float64, biases []float64, activation func(x float64) float64) []float64 {
	if biases != nil {
		if len(layer) != len(biases) {
			log.Fatal("Mismatched dimensions: layer and biases")
		}
		for i := range layer {
			layer[i] += biases[i]
		}
	}
	result := make([]float64, len(layer))
	for i, val := range layer {
		result[i] = activation(val)
	}
	return result
}

func (nn *NeuralNetwork) forwardPass(input []float64) []float64 {
	hiddenLayer := dot(input, nn.weightsInput)
	hiddenLayer = applyActivation(hiddenLayer, nn.biasesHidden, nn.activation)
	outputLayer := dot(hiddenLayer, nn.weightsHidden)
	outputLayer = applyActivation(outputLayer, nn.biasesOutput, nn.activation)
	return outputLayer
}

func (nn *NeuralNetwork) train(input []float64, target []float64) {
	// Forward-Pass
	hiddenLayer := dot(input, nn.weightsInput)
	hiddenLayer = applyActivation(hiddenLayer, nn.biasesHidden, nn.activation)
	outputLayer := dot(hiddenLayer, nn.weightsHidden)
	outputLayer = applyActivation(outputLayer, nn.biasesOutput, nn.activation)
	// Backpropagation
	outputError := sub(target, outputLayer)
	deltaOutput := mult(outputError, applyActivation(outputLayer, nil, nn.derivative))
	hiddenError := dot(deltaOutput, transpose(nn.weightsHidden))
	deltaHidden := mult(hiddenError, applyActivation(hiddenLayer, nil, nn.derivative))
	// Update weights and biases
	nn.weightsHidden = addWeights(nn.weightsHidden, outerProduct(hiddenLayer, deltaOutput))
	nn.biasesOutput = add(nn.biasesOutput, deltaOutput)
	nn.weightsInput = addWeights(nn.weightsInput, outerProduct(input, deltaHidden))
	nn.biasesHidden = add(nn.biasesHidden, deltaHidden)
}


func main() {
	cmd := exec.Command("python", "fetch_data.py", os.Args[1])
	output, err := cmd.Output()
	if err != nil {
		fmt.Println("Error returning Python script:", err)
	}
	fmt.Println("Python script output:\n"+string(output)+"\nLoading python output .csv into golang:")
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
    data := make([]SeriesData, 0, len(records) - 1)
    for i := 1; i < len(records); i++ {
		closePrice := parseFloat(records[i][5])
		var nextChange float64
		var signal string
		var percent float64
		var change float64
		if i > 2 {
			if i == len(records) - 1 {
				signal = "? (END OF DATA)"
			} else {
				nextClose := parseFloat(records[i+1][5])
				nextChange = nextClose - closePrice
				if nextChange >= 0.0 {
					signal = "1 (Buy)"
				} else {
					signal = "0 (Sell)"
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
    /* ML Feature Detection; finding desired correct output data (the buy/sell signals) to train our model on */
    for i := 0; i < len(data); i++ {
		fmt.Printf("%s signal at date: %s \n\t OHLC: (%f, %f, %f, %f) Day Range: %f\n", data[i].Signal, data[i].Date, data[i].Open, data[i].High, data[i].Low, data[i].Close, data[i].DayRange)
    }
	/* Initializing the NN */
	nn := newNN(numIn, numHidden, numOut)
	// input is { data[i].Close, data[i].DayRange}
	// or {data[i-1].Close, data[i].Close, data[i].DayRange} [limit training window to i := 0; i < len(data)]
	// target is {data[i+1].PointDelta}
	input := []float64{0.5, 0.7}
	target := []float64{0.8}
	for epoch := 0; epoch < numEpochs; epoch++ {
		nn.train(input, target)
	}
	predictions := nn.forwardPass(input)
	fmt.Println("Predictions:", predictions)
	/*// MA Crossover Signals
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
	// Signals based on future results
}