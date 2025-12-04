// Audio Monitoring Module
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const startListeningBtn = document.getElementById('startListening');
    const audioStatus = document.getElementById('audioStatus');
    const audioLevelMeter = document.getElementById('audioLevelMeter');
    const detectedEventList = document.getElementById('detectedEventList');
    
    // Audio context and variables
    let audioContext;
    let analyser;
    let microphone;
    let isListening = false;
    let model;
    
    // Detection thresholds
    const DETECTION_THRESHOLD = 0.78; // 78% confidence threshold
    
    // Initialize audio monitoring
    async function initAudioMonitoring() {
        try {
            // Load the TensorFlow.js speech commands model
            model = await window.speechCommands.create('BROWSER_FFT');
            await model.ensureModelLoaded();
            
            // Get the class labels from the model
            const labels = model.wordLabels();
            console.log('Model loaded with labels:', labels);
            
            // Set up audio context
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            analyser = audioContext.createAnalyser();
            
            // Set up the audio processing
            setupAudioProcessing();
            
            // Update UI
            audioStatus.textContent = 'Ready to monitor';
            startListeningBtn.disabled = false;
            
        } catch (error) {
            console.error('Error initializing audio monitoring:', error);
            audioStatus.textContent = 'Error initializing audio';
        }
    }
    
    // Set up audio processing
    function setupAudioProcessing() {
        // Get user media (microphone access)
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(function(stream) {
                // Create a media stream source
                microphone = audioContext.createMediaStreamSource(stream);
                
                // Connect the microphone to the analyser
                microphone.connect(analyser);
                
                // Set up the analyser
                analyser.fftSize = 256;
                const bufferLength = analyser.frequencyBinCount;
                const dataArray = new Uint8Array(bufferLength);
                
                // Start the analysis loop
                function analyze() {
                    if (!isListening) return;
                    
                    // Get the frequency data
                    analyser.getByteFrequencyData(dataArray);
                    
                    // Calculate the average volume
                    let sum = 0;
                    for (let i = 0; i < bufferLength; i++) {
                        sum += dataArray[i];
                    }
                    const average = sum / bufferLength;
                    
                    // Update the volume meter
                    const normalizedVolume = average / 255;
                    updateVolumeMeter(normalizedVolume);
                    
                    // Continue the analysis loop
                    requestAnimationFrame(analyze);
                }
                
                // Start the analysis
                analyze();
                
            })
            .catch(function(err) {
                console.error('Error accessing microphone:', err);
                audioStatus.textContent = 'Microphone access denied';
            });
    }
    
    // Update the volume meter
    function updateVolumeMeter(level) {
        const percentage = Math.min(100, Math.max(0, Math.round(level * 100)));
        audioLevelMeter.style.width = `${percentage}%`;
        
        // Change color based on level
        if (level > 0.7) {
            audioLevelMeter.style.backgroundColor = '#ef4444'; // Red for loud sounds
        } else if (level > 0.4) {
            audioLevelMeter.style.backgroundColor = '#f59e0b'; // Orange for medium sounds
        } else {
            audioLevelMeter.style.backgroundColor = '#10b981'; // Green for quiet sounds
        }
    }
    
    // Toggle audio monitoring
    async function toggleAudioMonitoring() {
        if (isListening) {
            // Stop listening
            if (microphone) {
                microphone.disconnect();
            }
            if (audioContext && audioContext.state !== 'closed') {
                await audioContext.close();
            }
            
            isListening = false;
            startListeningBtn.textContent = 'Start Listening';
            audioStatus.textContent = 'Monitoring stopped';
            
        } else {
            // Start listening
            try {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                
                // Get microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                microphone = audioContext.createMediaStreamSource(stream);
                microphone.connect(analyser);
                
                // Start the detection
                isListening = true;
                startListeningBtn.textContent = 'Stop Listening';
                audioStatus.textContent = 'Listening for sounds...';
                
                // Start the analysis loop
                function analyze() {
                    if (!isListening) return;
                    
                    // Get the frequency data
                    const bufferLength = analyser.frequencyBinCount;
                    const dataArray = new Uint8Array(bufferLength);
                    analyser.getByteFrequencyData(dataArray);
                    
                    // Calculate the average volume
                    let sum = 0;
                    for (let i = 0; i < bufferLength; i++) {
                        sum += dataArray[i];
                    }
                    const average = sum / bufferLength;
                    
                    // Update the volume meter
                    const normalizedVolume = average / 255;
                    updateVolumeMeter(normalizedVolume);
                    
                    // Continue the analysis loop
                    requestAnimationFrame(analyze);
                }
                
                // Start the analysis
                analyze();
                
                // Start the speech recognition
                await model.listen(result => {
                    // Get the scores for each class
                    const scores = Array.from(result.scores);
                    
                    // Get the index of the highest score
                    const maxScore = Math.max(...scores);
                    const maxIndex = scores.indexOf(maxScore);
                    const label = model.wordLabels()[maxIndex];
                    
                    // Only log detections above the threshold
                    if (maxScore >= DETECTION_THRESHOLD) {
                        console.log(`Detected: ${label} (${(maxScore * 100).toFixed(1)}%)`);
                        
                        // Add to the event list
                        addDetectedEvent(label, maxScore);
                    }
                    
                }, {
                    includeSpectrogram: true,
                    probabilityThreshold: 0.7,
                    invokeCallbackOnNoiseAndUnknown: false,
                    overlapFactor: 0.5
                });
                
            } catch (error) {
                console.error('Error starting audio monitoring:', error);
                audioStatus.textContent = 'Error: ' + error.message;
                isListening = false;
                startListeningBtn.textContent = 'Start Listening';
            }
        }
    }
    
    // Add a detected event to the list
    function addDetectedEvent(label, confidence) {
        const now = new Date();
        const timeString = now.toLocaleTimeString();
        const confidencePercent = (confidence * 100).toFixed(1);
        
        // Create the event element
        const eventElement = document.createElement('div');
        eventElement.className = 'detected-event';
        eventElement.innerHTML = `
            <div class="event-time">${timeString}</div>
            <div class="event-type">${label}</div>
            <div class="event-confidence">${confidencePercent}%</div>
        `;
        
        // Add to the top of the list
        if (detectedEventList.firstChild) {
            detectedEventList.insertBefore(eventElement, detectedEventList.firstChild);
        } else {
            detectedEventList.appendChild(eventElement);
        }
        
        // Keep only the last 10 events
        while (detectedEventList.children.length > 10) {
            detectedEventList.removeChild(detectedEventList.lastChild);
        }
    }
    
    // Initialize the audio monitoring when the page loads
    if (startListeningBtn) {
        startListeningBtn.addEventListener('click', toggleAudioMonitoring);
        initAudioMonitoring();
    }
});
