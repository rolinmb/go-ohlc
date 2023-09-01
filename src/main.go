package main

import (
    "encoding/csv"
    "fmt"
    "log"
    "os"
    "strconv"
)

type SeriesData struct {
    Date string
    Close float64
}

func parseFloat(value string) float64 {
    result, err := strconv.ParseFloat(value, 64)
    if err != nil {
        log.Fatal(err)
    }
    return result
}

func main() {
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
        entry := SeriesData{
            Date: records[i][1],
            Close: parseFloat(records[i][5]),
        }
        data = append(data, entry) 
    }
    /* ML Feature Detection. Feature detection is the process
	   of finding desired correct output data (the buy/sell signals) to train our model on
	*/
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
    currentDate := ""
    for i := 0; i < len(data) - 1; i++ {
		nextReturn := (data[i+1].Close - data[i].Close) / data[i].Close
		var signal string
		if nextReturn > 0 {
			signal = "Buy"
		} else {
			signal = "Sell"
		}
		currentDate = data[i].Date
		fmt.Printf("%s signal at date: %s\n", signal, currentDate)
    }
}