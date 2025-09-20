/**
 * Lore Weaver Chatbot Frontend Script
 *
 * Handles user interactions, API communication, and UI updates for the geomythological chatbot.
 */

class LoreWeaverChatbot {
    constructor() {
        this.apiBaseUrl = 'http://localhost:8000'; // Adjust for production
        this.currentQuery = '';
        this.currentResponse = '';
        this.currentWorkflowId = null;
        this.init();
    }

    init() {
        this.bindEvents();
        this.showWelcomeMessage();
    }

    bindEvents() {
        // Send message on button click
        document.getElementById('send-button').addEventListener('click', () => {
            this.sendMessage();
        });

        // Send message on Enter key
        document.getElementById('user-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Clear chat
        document.getElementById('clear-button').addEventListener('click', () => {
            this.clearChat();
        });

        // Feedback modal
        document.getElementById('feedback-button').addEventListener('click', () => {
            this.showFeedbackModal();
        });

        // Close modal
        document.querySelector('.close').addEventListener('click', () => {
            this.hideFeedbackModal();
        });

        // Star rating
        document.querySelectorAll('.star').forEach(star => {
            star.addEventListener('click', (e) => {
                this.selectRating(e.target.dataset.rating);
            });
        });

        // Submit feedback
        document.getElementById('submit-feedback').addEventListener('click', () => {
            this.submitFeedback();
        });

        // Workflow status
        document.getElementById('workflow-status-button').addEventListener('click', () => {
            this.checkWorkflowStatus();
        });

        // Co-creation mode toggle
        document.getElementById('cocreation-mode-button').addEventListener('click', () => {
            this.toggleCoCreationMode();
        });

        // Click outside modal to close
        window.addEventListener('click', (e) => {
            const modal = document.getElementById('feedback-modal');
            if (e.target === modal) {
                this.hideFeedbackModal();
            }
        });
    }

    async sendMessage() {
        const input = document.getElementById('user-input');
        const message = input.value.trim();

        if (!message) return;

        // Store current query for feedback
        this.currentQuery = message;

        // Add user message to chat
        this.addMessage(message, 'user');

        // Clear input
        input.value = '';

        // Show loading indicator
        this.showLoading();

        // Disable send button
        this.setSendButtonState(false);

        try {
            // Send to API
            const response = await this.callAPI(message);

            // Hide loading
            this.hideLoading();

            // Add bot response
            this.addMessage(response.answer, 'bot');

            // Store response for feedback
            this.currentResponse = response.answer;

            // Show metadata if available
            if (response.metadata && Object.keys(response.metadata).length > 0) {
                this.showMetadata(response.metadata);
            }

        } catch (error) {
            this.hideLoading();
            this.addMessage(`I apologize, but I encountered an error: ${error.message}. Please try again.`, 'bot');
            console.error('API Error:', error);
        }

        // Re-enable send button
        this.setSendButtonState(true);

        // Scroll to bottom
        this.scrollToBottom();
    }

    async callAPI(message) {
        // First try the workflow API for co-creation
        try {
            const workflowResponse = await fetch(`${this.apiBaseUrl}/api/workflow/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    trigger_source: 'human',
                    human_input: message
                })
            });

            if (workflowResponse.ok) {
                const workflowResult = await workflowResponse.json();
                // Store workflow ID for later feedback
                this.currentWorkflowId = workflowResult.workflow_id;

                // Return formatted response
                return {
                    answer: `🎨 Co-creation workflow started! Workflow ID: ${workflowResult.workflow_id}. I'll begin exploring your creative prompt through autonomous discovery, artistic generation, and knowledge synthesis.`,
                    metadata: {
                        workflow_id: workflowResult.workflow_id,
                        type: 'cocreation_started'
                    }
                };
            }
        } catch (workflowError) {
            console.log('Workflow API not available, falling back to chat:', workflowError);
        }

        // Fallback to regular chat API
        const response = await fetch(`${this.apiBaseUrl}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: message,
                max_results: 5
            })
        });

        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    addMessage(content, sender) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const avatar = sender === 'bot' ? '🤖' : '👤';

        messageDiv.innerHTML = `
            <div class="message-avatar">${avatar}</div>
            <div class="message-content">
                <p>${this.formatMessage(content)}</p>
            </div>
        `;

        messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatMessage(text) {
        // Basic formatting for better readability
        return text
            .replace(/\n/g, '</p><p>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    showMetadata(metadata) {
        let metadataHtml = '<div style="margin-top: 10px; padding: 10px; background: #f0f8ff; border-radius: 8px; font-size: 0.9em;">';

        if (metadata.sources && metadata.sources.length > 0) {
            metadataHtml += `<p><strong>Sources:</strong> ${metadata.sources.join(', ')}</p>`;
        }

        if (metadata.entities && metadata.entities.length > 0) {
            metadataHtml += `<p><strong>Entities mentioned:</strong> ${metadata.entities.join(', ')}</p>`;
        }

        if (metadata.locations && metadata.locations.length > 0) {
            metadataHtml += `<p><strong>Locations:</strong> ${metadata.locations.join(', ')}</p>`;
        }

        metadataHtml += '</div>';

        // Add metadata to last bot message
        const lastBotMessage = document.querySelector('.bot-message:last-child .message-content');
        if (lastBotMessage) {
            lastBotMessage.innerHTML += metadataHtml;
        }
    }

    showLoading() {
        document.getElementById('loading-indicator').style.display = 'block';
    }

    hideLoading() {
        document.getElementById('loading-indicator').style.display = 'none';
    }

    setSendButtonState(enabled) {
        const button = document.getElementById('send-button');
        button.disabled = !enabled;
        button.textContent = enabled ? 'Send' : 'Sending...';
    }

    scrollToBottom() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    clearChat() {
        const messagesContainer = document.getElementById('chat-messages');
        messagesContainer.innerHTML = '';

        // Add welcome message back
        this.showWelcomeMessage();
    }

    showWelcomeMessage() {
        const welcomeMessage = `
            Welcome back to Lore Weaver! I'm your guide through the rich tapestry of geomythological narratives.
            Ask me about mythological entities, geographic features, or cultural concepts, and I'll weave together
            stories from our vast knowledge base.
            <br><br>
            What would you like to explore today?
        `;
        this.addMessage(welcomeMessage, 'bot');
    }

    showFeedbackModal() {
        document.getElementById('feedback-modal').style.display = 'flex';
        document.getElementById('feedback-text').value = '';
        this.resetRating();
    }

    hideFeedbackModal() {
        document.getElementById('feedback-modal').style.display = 'none';
    }

    selectRating(rating) {
        // Reset all stars
        document.querySelectorAll('.star').forEach(star => {
            star.classList.remove('selected');
        });

        // Select stars up to the clicked rating
        for (let i = 1; i <= rating; i++) {
            document.querySelector(`.star[data-rating="${i}"]`).classList.add('selected');
        }

        this.selectedRating = rating;
    }

    resetRating() {
        document.querySelectorAll('.star').forEach(star => {
            star.classList.remove('selected');
        });
        this.selectedRating = null;
    }

    async submitFeedback() {
        if (!this.selectedRating) {
            alert('Please select a rating before submitting feedback.');
            return;
        }

        const feedbackText = document.getElementById('feedback-text').value;

        try {
            // Submit to regular feedback API
            const regularFeedbackResponse = await fetch(`${this.apiBaseUrl}/feedback`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: this.currentQuery,
                    response: this.currentResponse,
                    rating: parseInt(this.selectedRating),
                    feedback: feedbackText || null
                })
            });

            // If there's an active workflow, also submit workflow feedback
            if (this.currentWorkflowId) {
                const workflowFeedbackResponse = await fetch(`${this.apiBaseUrl}/api/workflow/feedback`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        workflow_id: this.currentWorkflowId,
                        rating: parseInt(this.selectedRating),
                        feedback_text: feedbackText || null,
                        curation_decisions: null
                    })
                });

                if (workflowFeedbackResponse.ok) {
                    console.log('Workflow feedback submitted successfully');
                } else {
                    console.warn('Failed to submit workflow feedback');
                }
            }

            if (regularFeedbackResponse.ok) {
                alert('Thank you for your feedback! It helps us improve our co-creation process.');
                this.hideFeedbackModal();
            } else {
                throw new Error('Failed to submit feedback');
            }

        } catch (error) {
            alert('Failed to submit feedback. Please try again.');
            console.error('Feedback submission error:', error);
        }
    }

    async checkWorkflowStatus() {
        if (!this.currentWorkflowId) {
            alert('No active workflow to check. Start a co-creation workflow first.');
            return;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/workflow/status/${this.currentWorkflowId}`);
            const status = await response.json();

            if (response.ok) {
                const statusMessage = `
Workflow Status for ${status.workflow_id}:
• Current Stage: ${status.current_stage}
• Stages Completed: ${status.stages_completed.join(', ')}
• Success: ${status.success ? 'Yes' : 'In Progress'}
${status.completion_time ? `• Completed: ${new Date(status.completion_time).toLocaleString()}` : ''}
                `;
                alert(statusMessage);
            } else {
                alert(`Error checking workflow status: ${status.detail}`);
            }

        } catch (error) {
            alert(`Network error checking workflow status: ${error.message}`);
        }
    }

    toggleCoCreationMode() {
        const button = document.getElementById('cocreation-mode-button');
        const isActive = button.classList.contains('active');

        if (isActive) {
            // Switch to normal chat mode
            button.classList.remove('active');
            button.textContent = '🎨 Co-Creation Mode';
            document.getElementById('user-input').placeholder = "Ask about a mythological figure, place, or cultural concept...";
            document.getElementById('workflow-status-button').style.display = 'none';
            this.addMessage("Switched to normal chat mode. Ask me about mythology and culture!", 'bot');
        } else {
            // Switch to co-creation mode
            button.classList.add('active');
            button.textContent = '💬 Normal Chat Mode';
            document.getElementById('user-input').placeholder = "Describe your creative vision or question for co-creation...";
            document.getElementById('workflow-status-button').style.display = 'inline-block';
            this.addMessage("🎨 Co-Creation Mode activated! Describe your creative prompt and I'll orchestrate a complete human-AI creative workflow through autonomous discovery, artistic generation, and knowledge synthesis.", 'bot');
        }
    }

    // Utility method to check API health
    async checkHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            return response.ok;
        } catch (error) {
            console.error('Health check failed:', error);
            return false;
        }
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.loreWeaverChatbot = new LoreWeaverChatbot();

    // Check API health on load
    window.loreWeaverChatbot.checkHealth().then(healthy => {
        if (!healthy) {
            console.warn('API is not available. Please ensure the backend is running.');
        }
    });
});

// Export for potential use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LoreWeaverChatbot;
}