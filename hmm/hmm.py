import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p
        self.forward_table = None


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        if len(input_observation_states) == 0:
            raise ValueError("Input sequence cannot be empty")

        for obs in input_observation_states:
            if obs not in self.observation_states:
                raise ValueError(f"'{obs}' not in the list of observation states")

        # Step 1. Initialize variables
        # make a matrix to store forward probabilities for each hidden state at each step
        observation = len(input_observation_states)
        states = len(self.hidden_states)

        forward_table = np.zeros((states, observation))

        # Step 2. Calculate probabilities
        # Calculating probabilities for the first "time"
        # initial prob = P(obs @ time 1 | specific state at time 1) * P(likelihood of hidden state as first [prior_p])

        for ix in range(states):
            obs_ix = self.observation_states_dict[input_observation_states[0]]
            forward_table[ix, 0] = self.prior_p[ix] * self.emission_p[ix, obs_ix]
        
        # Calculating probabilities for the rest
        # probability of observation given a specific state * max(prev state prob * transition prob)
        for o in range(1, observation):
            for s in range(states):
                obs_ix = self.observation_states_dict[input_observation_states[o]]
                prob_obs_given_state = self.emission_p[s, obs_ix]

                # taking log to prevent underflow
                forward_table[s, o] = np.sum(forward_table[:, o-1] * self.transition_p[:, s]) * prob_obs_given_state

        # Step 3. Return final probability
        # the final probability is the sum of the forward probabilities for the last step
        # converting log back to normal for score
        score = np.sum(forward_table[:, -1])
        self.forward_table = forward_table
        
        return score

    def viterbi(self, decode_observation_states):
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states

        Note: using the resources mentioned in the README (https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/)
        Secondary note: in this function, I used the help of Claude to understand the backpointer & implement the traceback
        """
        if len(decode_observation_states) == 0:
            raise ValueError("Input sequence cannot be empty")
        for obs in decode_observation_states:
            if obs not in self.observation_states:
                raise ValueError(f"'{obs}' not in the list of observation states")       

        observation = len(decode_observation_states)
        states = len(self.hidden_states)
        
        self.forward(decode_observation_states) # to make the probability table

        # Step 1. Initialize
        backpointer = np.zeros((states, observation), dtype=int)
        
        # Step 2. Calculate Probabilities
        for t in range(1, observation):
            observation_ix = self.observation_states_dict[decode_observation_states[t]]
            
            for s in range(states):
                transitions = self.transition_p[:, s]
                emissions = self.emission_p[s][observation_ix]
                prev_prob = self.forward_table[:, t-1]
                
                probs = prev_prob * transitions
                backpointer[s][t] = np.argmax(probs)

                self.forward_table[s][t] = np.max(probs) * emissions
        
        # Step 3. Traceback
        best_path = np.zeros(observation, dtype=int)
        best_path[-1] = np.argmax(self.forward_table[:, -1])

        for t in range(observation-2, -1, -1):
            best_path[t] = backpointer[best_path[t+1]][t+1]
        
        # Step 4. Return best hidden state sequence
        return [self.hidden_states[i] for i in best_path]