// VoiceMap Web Application JavaScript

class VoiceMapApp {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordingTimer = null;
        this.recordingDuration = 30; // 30 seconds
        this.currentTime = 30;
        this.audioBlob = null;
        
        this.init();
    }

    init() {
        this.initAnimations();
        this.initEventListeners();
        this.initTypingEffects();
        this.initScrollAnimations();
        this.initCounterAnimations();
    }

    initAnimations() {
        // Initialize AOS (Animate On Scroll)
        if (typeof AOS !== 'undefined') {
            AOS.init({
                duration: 800,
                easing: 'ease-in-out',
                once: true,
                offset: 100
            });
        }
    }

    initEventListeners() {
        // Medical history checkboxes
        const checkboxes = document.querySelectorAll('input[name="medical_history"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', this.handleMedicalHistoryChange.bind(this));
        });

        // Smooth scrolling for navigation links
        const navLinks = document.querySelectorAll('a[href^="#"]');
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });

        // Navbar scroll effect
        window.addEventListener('scroll', this.handleNavbarScroll.bind(this));
    }

    initTypingEffects() {
        // Typing effect for hero title
        const typingText = document.querySelector('.typing-text');
        if (typingText) {
            const text = typingText.textContent;
            typingText.textContent = '';
            let i = 0;
            
            const typeWriter = () => {
                if (i < text.length) {
                    typingText.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 100);
                }
            };
            
            setTimeout(typeWriter, 500);
        }

        // Typing effect for subtitle
        const typingSubtitle = document.querySelector('.typing-subtitle');
        if (typingSubtitle) {
            const text = typingSubtitle.textContent;
            typingSubtitle.textContent = '';
            let i = 0;
            
            const typeWriter = () => {
                if (i < text.length) {
                    typingSubtitle.textContent += text.charAt(i);
                    i++;
                    setTimeout(typeWriter, 50);
                }
            };
            
            setTimeout(typeWriter, 2000);
        }
    }

    initScrollAnimations() {
        // Parallax effect for hero section
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const heroSection = document.querySelector('.hero-section');
            if (heroSection) {
                heroSection.style.transform = `translateY(${scrolled * 0.5}px)`;
            }
        });
    }

    initCounterAnimations() {
        // Animate stat numbers
        const statNumbers = document.querySelectorAll('.stat-number[data-target]');
        
        const animateCounter = (element) => {
            const target = parseInt(element.getAttribute('data-target'));
            const duration = 2000;
            const step = target / (duration / 16);
            let current = 0;
            
            const timer = setInterval(() => {
                current += step;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                element.textContent = Math.floor(current);
            }, 16);
        };

        // Trigger animation when element comes into view
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateCounter(entry.target);
                    observer.unobserve(entry.target);
                }
            });
        });

        statNumbers.forEach(stat => observer.observe(stat));
    }

    handleNavbarScroll() {
        const navbar = document.querySelector('.navbar');
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(255, 255, 255, 0.98)';
            navbar.style.boxShadow = '0 4px 20px rgba(0,0,0,0.1)';
        } else {
            navbar.style.background = 'rgba(255, 255, 255, 0.95)';
            navbar.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
        }
    }

    handleMedicalHistoryChange(event) {
        const noneCheckbox = document.getElementById('none_of_above');
        const otherCheckboxes = document.querySelectorAll('input[name="medical_history"]:not(#none_of_above)');
        
        if (event.target.id === 'none_of_above') {
            // If "None of the above" is checked, uncheck all others
            if (event.target.checked) {
                otherCheckboxes.forEach(checkbox => {
                    checkbox.checked = false;
                });
            }
        } else {
            // If any other checkbox is checked, uncheck "None of the above"
            if (event.target.checked) {
                noneCheckbox.checked = false;
            }
        }
    }

    // Panel Navigation
    nextPanel() {
        const currentPanel = document.querySelector('.panel.active');
        const nextPanel = currentPanel.nextElementSibling;
        
        if (nextPanel && nextPanel.classList.contains('panel')) {
            currentPanel.classList.remove('active');
            nextPanel.classList.add('active');
            this.scrollToTop();
        }
    }

    previousPanel() {
        const currentPanel = document.querySelector('.panel.active');
        const previousPanel = currentPanel.previousElementSibling;
        
        if (previousPanel && previousPanel.classList.contains('panel')) {
            currentPanel.classList.remove('active');
            previousPanel.classList.add('active');
            this.scrollToTop();
        }
    }

    scrollToTop() {
        const screeningSection = document.getElementById('screening');
        if (screeningSection) {
            screeningSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    // Recording Functions
    async toggleRecording() {
        if (this.isRecording) {
            this.stopRecording();
        } else {
            await this.startRecording();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 22050,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });

            this.audioChunks = [];
            this.isRecording = true;
            this.currentTime = this.recordingDuration;

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                this.audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                this.updateRecordingUI();
            };

            this.mediaRecorder.start();
            this.startTimer();
            this.updateRecordingUI();

        } catch (error) {
            console.error('Error starting recording:', error);
            this.showAlert('Error', 'Could not access microphone. Please check permissions and try again.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            this.stopTimer();
            this.updateRecordingUI();
        }
    }

    startTimer() {
        this.recordingTimer = setInterval(() => {
            this.currentTime--;
            this.updateTimer();
            
            if (this.currentTime <= 0) {
                this.stopRecording();
            }
        }, 1000);
    }

    stopTimer() {
        if (this.recordingTimer) {
            clearInterval(this.recordingTimer);
            this.recordingTimer = null;
        }
    }

    updateTimer() {
        const timerElement = document.getElementById('recordingTimer');
        if (timerElement) {
            const minutes = Math.floor(this.currentTime / 60);
            const seconds = this.currentTime % 60;
            timerElement.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
    }

    updateRecordingUI() {
        const recordButton = document.getElementById('recordButton');
        const playButton = document.getElementById('playButton');
        const analyzeButton = document.getElementById('analyzeButton');
        const recordingStatus = document.getElementById('recordingStatus');
        const recordingArea = document.querySelector('.recording-area');

        if (this.isRecording) {
            recordButton.innerHTML = '<i class="fas fa-stop me-2"></i>Stop Recording';
            recordButton.className = 'btn btn-danger btn-lg';
            recordingStatus.textContent = 'Recording...';
            recordingArea.classList.add('recording');
            playButton.style.display = 'none';
            analyzeButton.style.display = 'none';
        } else {
            recordButton.innerHTML = '<i class="fas fa-microphone me-2"></i>Start Recording';
            recordButton.className = 'btn btn-primary btn-lg';
            recordingStatus.textContent = this.audioBlob ? 'Recording complete' : 'Click to start recording';
            recordingArea.classList.remove('recording');
            
            if (this.audioBlob) {
                playButton.style.display = 'inline-block';
                analyzeButton.style.display = 'inline-block';
            }
        }
    }

    playRecording() {
        if (this.audioBlob) {
            const audio = new Audio(URL.createObjectURL(this.audioBlob));
            audio.play();
        }
    }

    async analyzeRecording() {
        if (!this.audioBlob) {
            this.showAlert('Error', 'No recording available. Please record your voice first.');
            return;
        }

        // Show loading modal
        const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        loadingModal.show();

        try {
            const formData = new FormData();
            formData.append('audio', this.audioBlob, 'recording.webm');

            // Get medical history
            const medicalHistory = Array.from(document.querySelectorAll('input[name="medical_history"]:checked'))
                .map(checkbox => checkbox.value);
            formData.append('medical_history', JSON.stringify(medicalHistory));

            const response = await fetch('/record', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            loadingModal.hide();
            this.displayResults(result);

        } catch (error) {
            console.error('Error analyzing recording:', error);
            loadingModal.hide();
            this.showAlert('Error', 'Failed to analyze recording. Please try again.');
        }
    }

    displayResults(result) {
        const resultsContent = document.getElementById('resultsContent');
        
        // Determine result styling
        const isControl = result.prediction === 'Control';
        const resultClass = isControl ? 'success' : 'danger';
        const resultIcon = isControl ? 'fas fa-check-circle' : 'fas fa-exclamation-triangle';
        const resultMessage = isControl ? 
            'No significant cognitive impairment detected' : 
            'Dementia indicators detected - recommend clinical evaluation';

        resultsContent.innerHTML = `
            <div class="result-card">
                <div class="result-prediction text-${resultClass}">
                    <i class="${resultIcon} me-2"></i>
                    ${result.prediction}
                </div>
                <div class="result-confidence">
                    Confidence: ${result.confidence}%
                </div>
                
                <div class="probability-bars">
                    <div class="probability-bar control-probability">
                        <div class="probability-label">Control</div>
                        <div class="probability-value text-success">${result.control_probability}%</div>
                        <div class="progress">
                            <div class="progress-bar" style="width: ${result.control_probability}%"></div>
                        </div>
                    </div>
                    <div class="probability-bar dementia-probability">
                        <div class="probability-label">Dementia</div>
                        <div class="probability-value text-danger">${result.dementia_probability}%</div>
                        <div class="progress">
                            <div class="progress-bar" style="width: ${result.dementia_probability}%"></div>
                        </div>
                    </div>
                </div>
                
                <div class="result-interpretation">
                    <h5><i class="fas fa-lightbulb me-2"></i>Interpretation</h5>
                    <p>${resultMessage}</p>
                </div>
                
                ${result.adjustment_applied ? `
                    <div class="medical-adjustments">
                        <h6><i class="fas fa-info-circle me-2"></i>Medical History Adjustment</h6>
                        <p>Your results have been adjusted based on your medical history to account for speech-affecting conditions.</p>
                        <small>Adjustment factor: ${result.adjustment_factor}</small>
                    </div>
                ` : ''}
                
                <div class="mt-3">
                    <small class="text-muted">
                        <i class="fas fa-clock me-1"></i>
                        Analysis completed at ${result.timestamp}
                    </small>
                </div>
            </div>
        `;

        // Show results panel
        this.nextPanel();
    }

    startNewScreening() {
        // Reset form
        document.querySelectorAll('input[name="medical_history"]').forEach(checkbox => {
            checkbox.checked = false;
        });

        // Reset recording
        this.audioBlob = null;
        this.currentTime = this.recordingDuration;
        this.updateTimer();
        this.updateRecordingUI();

        // Go back to first panel
        const panels = document.querySelectorAll('.panel');
        panels.forEach(panel => panel.classList.remove('active'));
        document.getElementById('medicalHistoryPanel').classList.add('active');

        this.scrollToTop();
    }

    showAlert(title, message) {
        const alertTitle = document.getElementById('alertTitle');
        const alertBody = document.getElementById('alertBody');
        
        alertTitle.textContent = title;
        alertBody.textContent = message;
        
        const alertModal = new bootstrap.Modal(document.getElementById('alertModal'));
        alertModal.show();
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.voiceMapApp = new VoiceMapApp();
});

// Global functions for HTML onclick handlers
function nextPanel() {
    if (window.voiceMapApp) {
        window.voiceMapApp.nextPanel();
    }
}

function previousPanel() {
    if (window.voiceMapApp) {
        window.voiceMapApp.previousPanel();
    }
}

function toggleRecording() {
    if (window.voiceMapApp) {
        window.voiceMapApp.toggleRecording();
    }
}

function playRecording() {
    if (window.voiceMapApp) {
        window.voiceMapApp.playRecording();
    }
}

function analyzeRecording() {
    if (window.voiceMapApp) {
        window.voiceMapApp.analyzeRecording();
    }
}

function startNewScreening() {
    if (window.voiceMapApp) {
        window.voiceMapApp.startNewScreening();
    }
} 