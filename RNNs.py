import torch
import torch.nn as nn

# Recurrent Neural Network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.h2o(hidden)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Gate Recurrent Unit Network
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size

        # Reset gate parameters
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_activation = nn.Sigmoid()

        # Update gate parameters
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_activation = nn.Sigmoid()

        # New memory proposal parameters
        self.memory_proposal = nn.Linear(input_size + hidden_size, hidden_size)
        self.memory_activation = nn.Tanh()

        # Output layer
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # Concatenate input and hidden state
        combined = torch.cat((input, hidden), 1)

        # Calculate reset gate
        reset = self.reset_gate(combined)
        reset = self.reset_activation(reset)

        # Calculate update gate
        update = self.update_gate(combined)
        update = self.update_activation(update)

        # Calculate new memory proposal
        memory = self.memory_proposal(combined)
        memory = self.memory_activation(memory)

        # Update hidden state
        hidden = torch.mul((1 - reset), hidden) + torch.mul(reset, memory)

        # Output layer
        output = self.h2o(hidden)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# Long Short Term Memory 
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size

        # Forget gate parameters
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.forget_activation = nn.Sigmoid()

        # Input gate parameters
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_activation = nn.Sigmoid()

        # Update parameters
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.update_activation = nn.Tanh()

        # Output gate parameters
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_activation = nn.Sigmoid()

        # Output layer
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state, cell_state):
        # Concatenate input and hidden state
        combined = torch.cat((input, hidden_state), 1)

        # Forget gate
        forget = self.forget_gate(combined)
        forget = self.forget_activation(forget)

        # Input gate
        input_gate = self.input_gate(combined)
        input_gate = self.input_activation(input_gate)

        # Update gate
        update = self.update_gate(combined)
        update = self.update_activation(update)

        # Update cell state
        cell_state = torch.mul(forget, cell_state) + torch.mul(input_gate, update)

        # Output gate
        output_gate = self.output_gate(combined)
        output_gate = self.output_activation(output_gate)

        # Calculate hidden state
        hidden_state = torch.mul(output_gate, torch.tanh(cell_state))

        # Output layer
        output = self.h2o(hidden_state)
        output = self.softmax(output)

        return output, hidden_state, cell_state

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

    def initCellState(self):
        return torch.zeros(1, self.hidden_size)
