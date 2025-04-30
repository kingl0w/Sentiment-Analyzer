<script lang="ts">
  import { onMount } from 'svelte';
  import { sentimentService } from '../lib/api';
  import Chart from 'chart.js/auto';
  import type { ChartData, ChartOptions } from 'chart.js';
  import '../styles/SentimentAnalyzer.css';

  interface Example {
    label: string;
    text: string;
  }

  interface SentimentResult {
    sentiment: string;
    probabilities: Record<string, number>;
    confidence: number;
  }

  let text = '';
  let isAnalyzing = false;
  let isApiReady = false;
  let result: SentimentResult | null = null;
  let error: string | null = null;

  let chartEl: HTMLCanvasElement;
  let chartInstance: Chart | null = null;

  const examples: Example[] = [
    { label: 'Positive', text: 'I had a wonderful day! Everything went perfectly.' },
    { label: 'Neutral', text: "I'm feeling okay, nothing special happened today." },
    { label: 'Negative', text: "This was the worst experience of my life. I'm very disappointed." },
    { label: 'Mixed', text: 'The movie was absolutely marvelous, but the popcorn was stale.' }
  ];

  onMount(async () => {
    try {
      isApiReady = await sentimentService.checkHealth();
    } catch (e) {
      console.error('Failed to connect to API:', e);
      error = 'Could not connect to the sentiment analysis service. Make sure the Python backend is running.';
    }
  });

  $: if (result && chartEl) {
    const labels = Object.keys(result.probabilities);
    const values = Object.values(result.probabilities).map(p => p * 100);

    const chartData: ChartData<'doughnut', number[], unknown> = {
      labels,
      datasets: [{
        data: values,
        backgroundColor: ['#dc3545', '#6c757d', '#28a745']
      }]
    };

    const chartOptions: ChartOptions<'doughnut'> = {
      responsive: true,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: (context: any) => `${context.label}: ${context.raw.toFixed(1)}%`
          }
        }
      }
    };

    if (chartInstance) {
      chartInstance.destroy();
    }

    chartInstance = new Chart(chartEl, {
      type: 'doughnut',
      data: chartData,
      options: chartOptions
    });
  }

  async function analyzeSentiment() {
    if (!text.trim()) {
      error = 'Please enter some text to analyze';
      return;
    }

    error = null;
    result = null;
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

    {#if isAnalyzing}
      <div class="spinner-container">
        <div class="spinner"></div>
        <p>Analyzing sentiment...</p>
      </div>
    {/if}

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
          <canvas bind:this={chartEl}></canvas>
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

