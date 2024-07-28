// Initial chart setup
const lightTheme = {
    layout: {
        background: { type: 'solid', color: 'white' },
        textColor: 'black',
    },
    grid: {
        vertLines: {
            color: '#e1e1e1',
        },
        horzLines: {
            color: '#e1e1e1',
        },
    }
};

const darkTheme = {
    layout: {
        background: { type: 'solid', color: 'black' },
        textColor: 'white',
    },
    grid: {
        vertLines: {
            color: 'black',
        },
        horzLines: {
            color: 'black',
        },
    }
};

const chartOptions1 = {
    ...darkTheme, // Use the dark theme by default
    crosshair: {
        mode: LightweightCharts.CrosshairMode.Normal,
    },
    timeScale: {
        visible: false,
    },
    width: document.getElementById('chart').clientWidth,
    height: document.getElementById('chart').clientHeight,
};

const chartOptions2 = {
    ...darkTheme, // Use the dark theme by default
    timeScale: {
        visible: true,
    },
    width: document.getElementById('chart').clientWidth,
    height: document.getElementById('rsiChart').clientHeight,
};

const chart = LightweightCharts.createChart(document.getElementById('chart'), chartOptions1);
const candlestickSeries = chart.addCandlestickSeries();
const emaLine = chart.addLineSeries({
    color: 'blue', // Set the color for the EMA line
    lineWidth: 2
});

const rsiChart = LightweightCharts.createChart(document.getElementById('rsiChart'), chartOptions2);
const rsiLine = rsiChart.addLineSeries({
    color: 'red', // Set the color for the RSI line
    lineWidth: 2
});

let autoUpdateInterval;

// Fetch data function
function fetchData(ticker, emaPeriod, rsiPeriod) {
    fetch(`/api/data/${ticker}/${emaPeriod}/${rsiPeriod}`)
        .then(response => response.json())
        .then(data => {
            candlestickSeries.setData(data.candlestick);
            emaLine.setData(data.ema);
            rsiLine.setData(data.rsi);
        })
        .catch(error => {
            console.error('Error fetching data:', error);
        });
}

// Fetch initial data on page load
window.addEventListener('load', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const symbol = urlParams.get('symbol') || 'RELIANCE.NS';
    document.getElementById('ticker').value = symbol; // Update the symbol input value
    fetchData(symbol, 20, 14);
});

// Handle data fetching on button click
document.getElementById('fetchData').addEventListener('click', () => {
    const ticker = document.getElementById('ticker').value;
    const emaPeriod = document.getElementById('emaPeriod').value;
    const rsiPeriod = document.getElementById('rsiPeriod').value;
    fetchData(ticker, emaPeriod, rsiPeriod);
});

// Handle window resize
window.addEventListener('resize', () => {
    chart.resize(document.getElementById('chart').clientWidth, document.getElementById('chart').clientHeight);
    rsiChart.resize(document.getElementById('rsiChart').clientWidth, document.getElementById('rsiChart').clientHeight);
});

// Theme toggle functionality
document.querySelectorAll('.theme-toggle-button').forEach(button => {
    button.addEventListener('click', (event) => {
        const theme = event.currentTarget.getAttribute('data-theme');
        const themeOptions = theme === 'light' ? lightTheme : darkTheme;

        chart.applyOptions(themeOptions);
        rsiChart.applyOptions(themeOptions);
    });
});

// Sync visible logical range between charts
function syncVisibleLogicalRange(chart1, chart2) {
    chart1.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {
        chart2.timeScale().setVisibleLogicalRange(timeRange);
    });

    chart2.timeScale().subscribeVisibleLogicalRangeChange(timeRange => {
        chart1.timeScale().setVisibleLogicalRange(timeRange);
    });
}

syncVisibleLogicalRange(chart, rsiChart);

// Sync crosshair position between charts
function getCrosshairDataPoint(series, param) {
    if (!param.time) {
        return null;
    }
    const dataPoint = param.seriesData.get(series);
    return dataPoint || null;
}

function syncCrosshair(chart, series, dataPoint) {
    if (dataPoint) {
        chart.setCrosshairPosition(dataPoint.value, dataPoint.time, series);
        return;
    }
    chart.clearCrosshairPosition();
}

chart.subscribeCrosshairMove(param => {
    const dataPoint = getCrosshairDataPoint(candlestickSeries, param);
    syncCrosshair(rsiChart, rsiLine, dataPoint);
});

rsiChart.subscribeCrosshairMove(param => {
    const dataPoint = getCrosshairDataPoint(rsiLine, param);
    syncCrosshair(chart, candlestickSeries, dataPoint);
});
