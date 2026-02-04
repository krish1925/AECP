// Performance Charts and Cost Calculator

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all charts
    initSimilarityChart();
    initLatencyChart();
    initCostChart();
    initScalabilityChart();
    initCostCalculator();
});

// Semantic Similarity Comparison Chart
function initSimilarityChart() {
    const ctx = document.getElementById('similarityChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Text Serialization', 'AECP'],
            datasets: [{
                label: 'Semantic Similarity (%)',
                data: [43, 86],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(37, 99, 235, 0.8)'
                ],
                borderColor: [
                    'rgba(239, 68, 68, 1)',
                    'rgba(37, 99, 235, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + '% semantic similarity';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Semantic Similarity (%)'
                    }
                }
            }
        }
    });
}

// Latency Comparison Chart
function initLatencyChart() {
    const ctx = document.getElementById('latencyChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Text Serialization', 'AECP'],
            datasets: [{
                label: 'Transfer Latency (ms)',
                data: [150, 0.8],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(37, 99, 235, 0.8)'
                ],
                borderColor: [
                    'rgba(239, 68, 68, 1)',
                    'rgba(37, 99, 235, 1)'
                ],
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y + ' ms';
                        }
                    }
                }
            },
            scales: {
                y: {
                    type: 'logarithmic',
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return value + ' ms';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Transfer Latency (log scale)'
                    }
                }
            }
        }
    });
}

// Cost Comparison Chart
function initCostChart() {
    const ctx = document.getElementById('costChart');
    if (!ctx) return;
    
    // Default values
    const monthlyEmbeddings = 1000000;
    const highPrecisionRatio = 0.1;
    const avgTokens = 50;
    
    const costs = calculateCosts(monthlyEmbeddings, highPrecisionRatio, avgTokens);
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Always Expensive', 'AECP Cost Optimizer', 'Savings'],
            datasets: [{
                data: [costs.alwaysExpensive, costs.aecpCost, costs.savings],
                backgroundColor: [
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(37, 99, 235, 0.8)',
                    'rgba(16, 185, 129, 0.8)'
                ],
                borderColor: [
                    'rgba(239, 68, 68, 1)',
                    'rgba(37, 99, 235, 1)',
                    'rgba(16, 185, 129, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.parsed || 0;
                            return label + ': $' + value.toFixed(2);
                        }
                    }
                }
            }
        }
    });
}

// Scalability Chart
function initScalabilityChart() {
    const ctx = document.getElementById('scalabilityChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['10k', '50k', '100k', '200k', '300k'],
            datasets: [{
                label: 'Validation Similarity (%)',
                data: [94.2, 95.8, 96.5, 97.1, 97.3],
                borderColor: 'rgba(37, 99, 235, 1)',
                backgroundColor: 'rgba(37, 99, 235, 0.1)',
                tension: 0.4,
                fill: true
            }, {
                label: 'Training Similarity (%)',
                data: [92.5, 94.1, 95.2, 96.1, 95.9],
                borderColor: 'rgba(16, 185, 129, 1)',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + context.parsed.y + '%';
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 90,
                    max: 100,
                    ticks: {
                        callback: function(value) {
                            return value + '%';
                        }
                    },
                    title: {
                        display: true,
                        text: 'Similarity (%)'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Vocabulary Size'
                    }
                }
            }
        }
    });
}

// Cost Calculator
function initCostCalculator() {
    const monthlyInput = document.getElementById('monthlyEmbeddings');
    const precisionSlider = document.getElementById('highPrecisionRatio');
    const tokensInput = document.getElementById('avgTokens');
    const precisionValue = document.getElementById('precisionValue');
    
    if (!monthlyInput || !precisionSlider || !tokensInput) return;
    
    // Update precision value display
    precisionSlider.addEventListener('input', function() {
        precisionValue.textContent = this.value + '%';
        updateCosts();
    });
    
    // Update costs on input change
    [monthlyInput, tokensInput].forEach(input => {
        input.addEventListener('input', updateCosts);
    });
    
    function updateCosts() {
        const monthlyEmbeddings = parseInt(monthlyInput.value) || 1000000;
        const highPrecisionRatio = parseInt(precisionSlider.value) / 100 || 0.1;
        const avgTokens = parseInt(tokensInput.value) || 50;
        
        const costs = calculateCosts(monthlyEmbeddings, highPrecisionRatio, avgTokens);
        
        // Update display
        document.getElementById('alwaysExpensive').textContent = '$' + costs.alwaysExpensive.toFixed(2);
        document.getElementById('aecpCost').textContent = '$' + costs.aecpCost.toFixed(2);
        document.getElementById('monthlySavings').textContent = '$' + costs.savings.toFixed(2);
        document.getElementById('annualSavings').textContent = '$' + (costs.savings * 12).toFixed(2);
        
        // Update chart if it exists
        const costChartCtx = document.getElementById('costChart');
        if (costChartCtx && window.costChartInstance) {
            window.costChartInstance.data.datasets[0].data = [
                costs.alwaysExpensive,
                costs.aecpCost,
                costs.savings
            ];
            window.costChartInstance.update();
        }
    }
    
    // Initial calculation
    updateCosts();
}

// Cost calculation function
function calculateCosts(monthlyEmbeddings, highPrecisionRatio, avgTokens) {
    // Pricing (per 1M tokens)
    const cheapCostPerMillion = 0.02;  // OpenAI small
    const expensiveCostPerMillion = 0.12;  // Voyage large
    
    // Calculate tokens
    const monthlyTokens = monthlyEmbeddings * avgTokens;
    const tokensInMillions = monthlyTokens / 1000000;
    
    // Always expensive model
    const alwaysExpensive = tokensInMillions * expensiveCostPerMillion;
    
    // AECP cost optimizer
    const highPrecisionTokens = tokensInMillions * highPrecisionRatio;
    const lowPrecisionTokens = tokensInMillions * (1 - highPrecisionRatio);
    
    const aecpCost = (highPrecisionTokens * expensiveCostPerMillion) +
                     (lowPrecisionTokens * cheapCostPerMillion);
    
    const savings = alwaysExpensive - aecpCost;
    
    return {
        alwaysExpensive,
        aecpCost,
        savings
    };
}
