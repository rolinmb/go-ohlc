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

func dot(vec []float64, mat [][]float64) []float64 {
	result := make([]float64, len(mat[0]))
	for i := range mat[0] {
		for j := range vec {
			result[i] += vec[j] * mat[j][i]
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

func main() {
	rand.Seed(time.Now().UnixNano())
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
		fmt.Printf("%s signal at date: %s -> OHLC: (%f, %f, %f, %f) -> Day Range: %f\n", data[i].Signal, data[i].Date, data[i].Open, data[i].High, data[i].Low, data[i].Close, data[i].DayRange)
    }
	/* Initializing the NN */
	

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