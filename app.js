document.addEventListener('DOMContentLoaded', () => {
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultContainer = document.getElementById('resultContainer');
    const sentimentIcon = document.getElementById('sentimentIcon');
    const sentimentResult = document.getElementById('sentimentResult');
    const confidenceText = document.getElementById('confidenceText');
    const confidenceIndicator = document.getElementById('confidenceIndicator');
    const additionalFeedback = document.getElementById('additionalFeedback');
    const loader = document.getElementById('loader');
    const exampleButtons = document.querySelectorAll('.example-btn');

    const API_URL = '/predict';
    const MAX_CHARS = 500;

    // Character count tracking
    textInput.addEventListener('input', () => {
        const currentLength = textInput.value.length;
        charCount.textContent = `${currentLength} / ${MAX_CHARS}`;
        
        if (currentLength > MAX_CHARS) {
            textInput.value = textInput.value.slice(0, MAX_CHARS);
            charCount.textContent = `${MAX_CHARS} / ${MAX_CHARS}`;
        }
    });

    // Example buttons functionality
    exampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            textInput.value = btn.textContent;
            analyzeBtn.click();
        });
    });

    async function analyzeSentiment(text) {
        loader.classList.remove('hidden');
        resultContainer.classList.add('hidden');

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error('Sentiment analysis failed');
            }

            const data = await response.json();
            updateResultDisplay(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze sentiment. Please try again.');
        } finally {
            loader.classList.add('hidden');
        }
    }

    function updateResultDisplay(data) {
        const confidence = data.confidence;
        const sentiment = data.sentiment;

        // Reset previous classes
        sentimentIcon.classList.remove('fa-smile', 'fa-frown', 'fa-meh', 'positive', 'negative', 'neutral');
        
        // Set sentiment display
        if (sentiment === 'Positive') {
            sentimentIcon.classList.add('fa-smile', 'positive');
            sentimentResult.textContent = 'Positive Sentiment';
            additionalFeedback.textContent = 'The text expresses a positive emotional tone.';
        } else {
            sentimentIcon.classList.add('fa-frown', 'negative');
            sentimentResult.textContent = 'Negative Sentiment';
            additionalFeedback.textContent = 'The text suggests a negative emotional undertone.';
        }

        // Confidence display
        const confidencePercentage = (confidence * 100).toFixed(2);
        confidenceText.textContent = `Confidence: ${confidencePercentage}%`;
        confidenceIndicator.style.width = `${confidencePercentage}%`;
        confidenceIndicator.style.backgroundColor = sentiment === 'Positive' ? 'var(--positive-color)' : 'var(--negative-color)';

        resultContainer.classList.remove('hidden');
    }

    analyzeBtn.addEventListener('click', () => {
        const text = textInput.value.trim();

        if (!text) {
            alert('Please enter some text to analyze');
            return;
        }

        analyzeSentiment(text);
    });
});