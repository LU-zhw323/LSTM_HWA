import torch
import torch.nn as nn
import torch.optim as optim

class LSTM_PTB(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.5):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0  
        )
        
        
        self.dropout = nn.Dropout(dropout)
        
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        init_range = 0.1  
        
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.uniform_(param, -init_range, init_range)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    
    def forward(self, x, hidden=None):
        
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        x, new_hidden = self.lstm(x, hidden)  # (batch_size, seq_length, hidden_size)
        
        x = self.dropout(x)
        
        x = self.fc(x)  # (batch_size, seq_length, vocab_size)
        
        return x, new_hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    
    def get_embedding_component(self):
        return self.embedding
    
    def get_lstm_component(self):
        return self.lstm, self.dropout
    
    def get_output_component(self):
        return self.fc
    
    def forward_embedding_only(self, x):
        return self.embedding(x)
    
    def forward_lstm_only(self, embedded_x, hidden=None):
        lstm_out, new_hidden = self.lstm(embedded_x, hidden)
        lstm_out = self.dropout(lstm_out)
        return lstm_out, new_hidden
    
    def forward_output_only(self, lstm_out):
        return self.fc(lstm_out)



class AnalogLSTM_PTB(nn.Module):
        def __init__(self, lstm, dropout, fc):
            super().__init__()
            self.lstm = lstm
            self.dropout = dropout
            self.fc = fc
            
        def forward_lstm_only(self, embedded_x, hidden=None):
            lstm_out, new_hidden = self.lstm(embedded_x, hidden)
            lstm_out = self.dropout(lstm_out)
            return lstm_out, new_hidden
            
        def forward_output_only(self, lstm_out):
            return self.fc(lstm_out)
            
        def init_hidden(self, batch_size, device):
            h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
            c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=device)
            return (h0, c0)