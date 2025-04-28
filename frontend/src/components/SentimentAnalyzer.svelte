<script lang="ts">
  import { onMount } from 'svelte';
  import { sentimentService } from '../lib/api';
  import { Doughnut } from 'svelte-chartjs';
  import { Chart, Title, Tooltip, Legend, ArcElement } from 'chart.js';
  import type { ChartData, ChartOptions } from 'chart.js';
  import '../styles/SentimentAnalyzer.css';
  
  //registers chart.js components
  Chart.register(Title, Tooltip, Legend, ArcElement);

  //chart data type definition
  type DoughnutData = ChartData<'doughnut', number[], unknown>;
  type DoughnutOptions = ChartOptions<'doughnut'>;
  
  //defines interfaces
  interface Example {
    label: string;
    text: string;
  }
  
  interface SentimentResult {
    sentiment: string;
    probabilities: Record<string, number>;
    confidence: number;
  }
  
  //state variables
  let text = '';
  let isAnalyzing = false;
  let isApiReady = false;
  let result: SentimentResult | null = null;
  let error: string | null = null;
  
  //initializes chart data with default values
  let chartData: DoughnutData = {
    labels: [],
    datasets: [{
      data: [],
      backgroundColor: []
    }]
  };
  
  let chartOptions: DoughnutOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom'
      },
      tooltip: {
        callbacks: {
          label: function(context: any) {
            return `${context.label}: ${context.raw.toFixed(1)}%`;
          }
        }
      }
    }
  };
  
  //examples
  const examples: Example[] = [
    { label: 'Positive', text: 'I had a wonderful day! Everything went perfectly.' },
    { label: 'Neutral', text: "I'm feeling okay, nothing special happened today." },
    { label: 'Negative', text: "This was the worst experience of my life. I'm very disappointed." },
    { label: 'Mixed', text: 'The movie was absolutely marvelous, but the popcorn was stale.' }
  ];
  
  //checks if API is available on mount
  onMount(async () => {
    try {
      isApiReady = await sentimentService.checkHealth();
    } catch (e) {
      console.error('Failed to connect to API:', e);
      error = 'Could not connect to the sentiment analysis service. Make sure the Python backend is running.';
    }
  });
  
  //updates chart data when result changes
  $: if (result) {
    chartData = {
      labels: Object.keys(result.probabilities),
      datasets: [{
        data: Object.values(result.probabilities).map(p => p * 100),
        backgroundColor: [
          '#dc3545', //red
          '#6c757d', //gray
          '#28a745'  //green
        ]
      }]
    };
  }
  
  //API call to analyze sentiment
  async function analyzeSentiment() {
    if (!text.trim()) {
      error = 'Please enter some text to analyze';
      return;
    }
    
    error = null;
    isAnalyzing = true;
    
    try {
      result = await sentimentService.analyze(text);
    } catch (err: any) {
      console.error('Error analyzing text:', err);
      error = err.response?.data?.detail || 'An error occurred during analysis';
      result = null;
    } finally {
      isAnalyzing = false;
    }
  }
  
  //use example text
  function useExample(example: Example) {
    text = example.text;
    analyzeSentiment();
  }
</script>

<div class="container">
  {#if !isApiReady && !error}
    <div class="loading">Connecting to sentiment analysis service...</div>
  {:else if error}
    <div class="error-container">
      <div class="error">{error}</div>
      {#if error.includes('backend is running')}
        <p>Check that your Python backend is running at http://localhost:8000</p>
      {/if}
    </div>
  {:else}
    <div class="input-section">
      <textarea 
        bind:value={text} 
        placeholder="Type or paste your text here..." 
        rows="6" 
        disabled={isAnalyzing}
      ></textarea>
      
      <button on:click={analyzeSentiment} disabled={isAnalyzing || !text.trim()}>
        {isAnalyzing ? 'Analyzing...' : 'Analyze Sentiment'}
      </button>
    </div>
    
    <div class="examples">
      <h3>Try some examples:</h3>
      <div class="example-buttons">
        {#each examples as example}
          <button class="example-btn" on:click={() => useExample(example)}>
            {example.label} Example
          </button>
        {/each}
      </div>
    </div>
    
    {#if result}
      <div class="result-container {result.sentiment.toLowerCase()}">
        <h2>Result</h2>
        
        <div class="sentiment-result">
          <h3>Sentiment: {result.sentiment}</h3>
          <p class="confidence">Confidence: {(result.confidence * 100).toFixed(1)}%</p>
        </div>
        
        <div class="chart-container">
          <Doughnut data={chartData} options={chartOptions} />
        </div>
      </div>
    {/if}
  {/if}
  
  <div class="explanation">
    <h3>How it works:</h3>
    <p>
      This application uses a machine learning model (DistilBERT) trained on 
      millions of text samples to recognize sentiment patterns. The model analyzes 
      the context and relationships between words rather than just looking for 
      specific words in isolation.
    </p>
    <p>
      The chart shows the probabilities for each sentiment category, giving you 
      insight into how confident the model is in its prediction.
    </p>
  </div>
</div>