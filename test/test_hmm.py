import pytest
from hmm import HiddenMarkovModel
import numpy as np

def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    hidden_states, observation_states, prior_p, transition_p, emission_p = mini_hmm.files
    observation_state_sequence, best_hidden_state_sequence = mini_input.files

    # Initialize the HMM object
    hmm = HiddenMarkovModel(
        hidden_states=mini_hmm[hidden_states], 
        observation_states=mini_hmm[observation_states], 
        prior_p=mini_hmm[prior_p], 
        transition_p=mini_hmm[transition_p], 
        emission_p=mini_hmm[emission_p]
    )

    # testing forward and viterbi algorithms with correct outputs
    forward_score = hmm.forward(mini_input[observation_state_sequence])

    assert forward_score == pytest.approx(0.0350644, 0.0001)

    viterbi_states = hmm.viterbi(mini_input[observation_state_sequence])

    assert viterbi_states == mini_input[best_hidden_state_sequence].tolist()

    #testing edge cases
    with pytest.raises(ValueError):
        hmm.forward([])
    
    with pytest.raises(ValueError):
        hmm.viterbi([])
    
    with pytest.raises(ValueError):
        hmm.forward(['snow'])
    
    with pytest.raises(ValueError):
        hmm.viterbi(['snow'])

def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm = np.load('./data/full_weather_hmm.npz')
    full_input = np.load('./data/full_weather_sequences.npz')

    hidden_states, observation_states, prior_p, transition_p, emission_p = full_hmm.files
    observation_state_sequence, best_hidden_state_sequence = full_input.files
    
    hmm = HiddenMarkovModel(
        hidden_states=full_hmm[hidden_states], 
        observation_states=full_hmm[observation_states], 
        prior_p=full_hmm[prior_p], 
        transition_p=full_hmm[transition_p], 
        emission_p=full_hmm[emission_p]
    )

    forward_score = hmm.forward(full_input[observation_state_sequence])
    assert isinstance(forward_score, float)
    assert forward_score >= 0

    viterbi_states = hmm.viterbi(full_input[observation_state_sequence])

    assert viterbi_states == full_input[best_hidden_state_sequence].tolist()


    #testing edge cases
    with pytest.raises(ValueError):
        hmm.forward([])
    
    with pytest.raises(ValueError):
        hmm.viterbi([])
    
    with pytest.raises(ValueError):
        hmm.forward(['snow'])
    
    with pytest.raises(ValueError):
        hmm.viterbi(['snow'])
