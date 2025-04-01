import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import deque
import os
import json

class FencingSequenceDataset(Dataset):
    """Dataset for fencing technique sequences"""
    def __init__(self, sequences, technique_to_id):
        """
        Initialize the dataset with sequences of techniques
        
        Args:
            sequences: List of technique sequences
            technique_to_id: Dict mapping technique names to IDs
        """
        self.sequences = sequences
        self.technique_to_id = technique_to_id
        self.id_to_technique = {v: k for k, v in technique_to_id.items()}
        self.data = []
        
        # Process sequences into input-output pairs
        for sequence in sequences:
            if len(sequence) < 2:
                continue
                
            # Convert techniques to IDs
            seq_ids = [technique_to_id.get(tech, 0) for tech in sequence]
            
            # Create input-output pairs (predict next technique)
            for i in range(len(seq_ids) - 1):
                # Use current technique to predict next technique
                self.data.append((seq_ids[i], seq_ids[i+1]))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_id, target_id = self.data[idx]
        
        # Convert to one-hot encoding
        input_tensor = torch.zeros(len(self.technique_to_id))
        input_tensor[input_id] = 1.0
        
        return input_tensor, target_id

class FencingSequenceLSTM(nn.Module):
    """LSTM model for fencing sequence prediction"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        """
        Initialize the LSTM model
        
        Args:
            input_size: Size of input features (number of techniques)
            hidden_size: Size of hidden state
            output_size: Size of output (number of techniques)
            num_layers: Number of LSTM layers
        """
        super(FencingSequenceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden: Optional initial hidden state
            
        Returns:
            output: Predicted next technique probabilities
            hidden: Final hidden state
        """
        # Reshape input if it's just a single technique (not a sequence)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            hidden = (h0, c0)
        
        # LSTM forward
        out, hidden = self.lstm(x, hidden)
        
        # Only take the output from the last time step
        out = self.fc(out[:, -1, :])
        
        return out, hidden

class FencingSequenceAnalyzer:
    """Analyzer for fencing technique sequences"""
    def __init__(self, model_path=None):
        """
        Initialize the sequence analyzer
        
        Args:
            model_path: Optional path to pre-trained model
        """
        self.standard_techniques = [
            'attack', 'defense', 'parry', 'riposte', 'lunge',
            'advance', 'retreat', 'feint'
        ]
        
        # Create technique mappings
        self.technique_to_id = {tech: i for i, tech in enumerate(self.standard_techniques)}
        self.id_to_technique = {i: tech for i, tech in enumerate(self.standard_techniques)}
        
        # Initialize model
        input_size = len(self.standard_techniques)
        hidden_size = 64
        output_size = len(self.standard_techniques)
        
        self.model = FencingSequenceLSTM(input_size, hidden_size, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            try:
                print(f"Loading pre-trained sequence model from {model_path}")
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Initializing new model")
        else:
            print("No pre-trained model found, initializing new model")
        
        # Sequence buffer to store recent techniques
        self.buffer_size = 10
        self.sequence_buffer = deque(maxlen=self.buffer_size)
        
        # Transition matrix to track technique transitions
        self.transition_matrix = np.zeros((len(self.standard_techniques), len(self.standard_techniques)))
        
        # Common fencing sequences (for rule-based prediction)
        self.common_sequences = {
            'attack': ['retreat', 'parry', 'riposte'],
            'parry': ['riposte', 'retreat', 'attack'],
            'riposte': ['retreat', 'advance', 'lunge'],
            'lunge': ['retreat', 'recovery', 'advance'],
            'advance': ['lunge', 'attack', 'retreat'],
            'retreat': ['parry', 'counterattack', 'advance'],
            'feint': ['attack', 'lunge', 'advance']
        }
    
    def predict_next_technique(self, current_technique):
        """
        Predict the next technique based on the current technique
        
        Args:
            current_technique: Current fencing technique
            
        Returns:
            next_technique: Most likely next technique
            probabilities: Dict of technique probabilities
        """
        # Add current technique to buffer
        self.sequence_buffer.append(current_technique)
        
        # Convert technique to ID
        tech_id = self.technique_to_id.get(current_technique, 0)
        
        # Convert to tensor
        input_tensor = torch.zeros(len(self.standard_techniques))
        input_tensor[tech_id] = 1.0
        input_tensor = input_tensor.to(self.device)
        
        # Get model prediction
        self.model.eval()
        with torch.no_grad():
            output, _ = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        # Convert probabilities to dict
        probabilities = {self.id_to_technique[i]: probs[i].item() for i in range(len(probs))}
        
        # Get most likely next technique
        next_tech_id = torch.argmax(probs).item()
        next_technique = self.id_to_technique[next_tech_id]
        
        # Update transition matrix
        self.transition_matrix[tech_id, next_tech_id] += 1
        
        return next_technique, probabilities
    
    def predict_sequence(self, initial_technique, sequence_length=5):
        """
        Predict a sequence of techniques starting from an initial technique
        
        Args:
            initial_technique: Starting technique
            sequence_length: Length of sequence to predict
            
        Returns:
            sequence: List of predicted techniques
        """
        sequence = [initial_technique]
        current_technique = initial_technique
        
        for _ in range(sequence_length - 1):
            next_technique, _ = self.predict_next_technique(current_technique)
            sequence.append(next_technique)
            current_technique = next_technique
        
        return sequence
    
    def add_observed_sequence(self, technique_sequence):
        """
        Add an observed sequence to update the model's knowledge
        
        Args:
            technique_sequence: List of techniques in temporal order
        """
        # Update transition matrix
        for i in range(len(technique_sequence) - 1):
            current_tech = technique_sequence[i]
            next_tech = technique_sequence[i + 1]
            
            # Get technique IDs
            current_id = self.technique_to_id.get(current_tech, 0)
            next_id = self.technique_to_id.get(next_tech, 0)
            
            # Update matrix
            self.transition_matrix[current_id, next_id] += 1
    
    def extract_sequences_from_timeline(self, timeline_data, min_gap=10):
        """
        Extract technique sequences from timeline data
        
        Args:
            timeline_data: List of (frame_idx, technique) tuples
            min_gap: Minimum frame gap to consider as sequence break
            
        Returns:
            sequences: List of technique sequences
        """
        if not timeline_data:
            return []
        
        sequences = []
        current_sequence = []
        prev_frame = None
        
        # Sort timeline by frame index
        sorted_timeline = sorted(timeline_data, key=lambda x: x[0])
        
        for frame_idx, technique in sorted_timeline:
            # Start new sequence if gap is too large
            if prev_frame is not None and frame_idx - prev_frame > min_gap:
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = []
            
            # Add technique to current sequence
            current_sequence.append(technique)
            prev_frame = frame_idx
        
        # Add final sequence
        if current_sequence:
            sequences.append(current_sequence)
        
        return sequences
    
    def get_transition_matrix(self):
        """
        Get the current transition matrix
        
        Returns:
            matrix: Numpy array of transition probabilities
            labels: Technique labels
        """
        # Normalize the transition matrix to get probabilities
        normalized_matrix = self.transition_matrix.copy()
        row_sums = normalized_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(normalized_matrix, row_sums, 
                                    where=row_sums!=0, out=np.zeros_like(normalized_matrix))
        
        return normalized_matrix, self.standard_techniques
    
    def train_model(self, sequence_data, epochs=100, batch_size=16, learning_rate=0.001):
        """
        Train the LSTM model on sequence data
        
        Args:
            sequence_data: List of technique sequences
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            
        Returns:
            history: Training history (losses)
        """
        # Create dataset
        dataset = FencingSequenceDataset(sequence_data, self.technique_to_id)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        losses = []
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs, _ = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return losses
    
    def save_model(self, model_path="models/sequence_model.pth"):
        """Save the trained model"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
    
    def evaluate_sequence_likelihood(self, sequence):
        """
        Evaluate how likely a sequence is based on transition probabilities
        
        Args:
            sequence: List of techniques
            
        Returns:
            likelihood: Overall likelihood score
        """
        if len(sequence) < 2:
            return 1.0
        
        # Calculate likelihood based on transition matrix
        likelihood = 1.0
        for i in range(len(sequence) - 1):
            current_tech = sequence[i]
            next_tech = sequence[i + 1]
            
            current_id = self.technique_to_id.get(current_tech, 0)
            next_id = self.technique_to_id.get(next_tech, 0)
            
            # Get transition probability
            if self.transition_matrix[current_id].sum() > 0:
                prob = self.transition_matrix[current_id, next_id] / self.transition_matrix[current_id].sum()
            else:
                # Fallback to rule-based probability
                common_next = self.common_sequences.get(current_tech, [])
                prob = 0.8 if next_tech in common_next else 0.2
            
            likelihood *= prob
        
        return likelihood
    
    def identify_signature_sequences(self, fencer_sequences, min_length=3):
        """
        Identify signature sequences that a fencer frequently uses
        
        Args:
            fencer_sequences: List of technique sequences by a fencer
            min_length: Minimum length of signature sequences
            
        Returns:
            signatures: List of signature sequences with their frequency
        """
        # Extract all subsequences of sufficient length
        all_subsequences = []
        for sequence in fencer_sequences:
            if len(sequence) < min_length:
                continue
                
            for i in range(len(sequence) - min_length + 1):
                subsequence = tuple(sequence[i:i + min_length])
                all_subsequences.append(subsequence)
        
        # Count frequency of each subsequence
        from collections import Counter
        subseq_counts = Counter(all_subsequences)
        
        # Get the most common subsequences
        signatures = [(list(subseq), count) for subseq, count in subseq_counts.most_common(5)]
        
        return signatures

# Example usage
if __name__ == "__main__":
    # Sample sequence data
    sample_sequences = [
        ['advance', 'lunge', 'retreat'],
        ['parry', 'riposte', 'retreat'],
        ['advance', 'feint', 'attack', 'retreat'],
        ['advance', 'attack', 'parry', 'riposte'],
        ['retreat', 'parry', 'riposte', 'advance'],
        ['lunge', 'retreat', 'advance', 'attack']
    ]
    
    # Initialize analyzer
    analyzer = FencingSequenceAnalyzer()
    
    # Add observed sequences
    for sequence in sample_sequences:
        analyzer.add_observed_sequence(sequence)
    
    # Train model on sample data
    print("Training sequence model...")
    history = analyzer.train_model(sample_sequences, epochs=50)
    
    # Get transition matrix
    matrix, labels = analyzer.get_transition_matrix()
    print("\nTechnique transition probabilities:")
    for i, from_tech in enumerate(labels):
        for j, to_tech in enumerate(labels):
            if matrix[i, j] > 0:
                print(f"{from_tech} → {to_tech}: {matrix[i, j]:.3f}")
    
    # Predict next technique after 'advance'
    next_tech, probs = analyzer.predict_next_technique('advance')
    print(f"\nAfter 'advance', most likely next technique: {next_tech}")
    print("Probabilities:")
    for tech, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        if prob > 0.01:
            print(f"  {tech}: {prob:.3f}")
    
    # Predict a sequence starting with 'advance'
    sequence = analyzer.predict_sequence('advance', sequence_length=5)
    print(f"\nPredicted sequence starting with 'advance': {' → '.join(sequence)}")
    
    # Evaluate likelihood of a sequence
    test_sequence = ['advance', 'lunge', 'retreat']
    likelihood = analyzer.evaluate_sequence_likelihood(test_sequence)
    print(f"\nLikelihood of sequence '{' → '.join(test_sequence)}': {likelihood:.3f}")
    
    # Identify signature sequences
    signatures = analyzer.identify_signature_sequences(sample_sequences)
    print("\nSignature sequences:")
    for sequence, count in signatures:
        print(f"  {' → '.join(sequence)}: {count} occurrences") 