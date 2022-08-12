function CRM_MDP_TestTemporalFunctions(n)
% Active inference for cocktail paty listening
%__________________________________________________________________________
%
% Inputs:
%   n           Number to use as an identifier when saving output file
%               (useful when running the script on a cluster, for example).
%
% This routine uses active inference for Markov decision processes to
% illustrate attending to one phrase in a mixture during cocktail party
% listening. It chooses among 16 possible colour-number combinations for
% the target phrase. To identify the correct response, the agent should
% look out for the visual cue on the screen (left/right), use it to infer
% the target phrase, then report the colour-number combination from the
% target phrase. The competing phrase(s) contain different colour-number
% combinations that constitute incorrect responses. The agent receives
% feedback (correct/incorrect) about its response.
%
% Crucially, the visual cue is presented before the phrases. Precision
% changes on each epoch after the visual cue is presented, according to
% a temporal function (hz).
%
% This script simulates 5 different temporal functions.
%
% This is the script used in the following paper to simulate preparatory
% attention under different temporal functions:
% Holmes, E., Parr, T., Griffiths, T. D. & Friston, K. J. (2021). Active
% inference, selective attention, and the cocktail party problem. Neurosci.
% Biobehav. Rev. 131, 1288–1304.

% Emma Holmes


%% Setup

nLevels         = 10;
onsets          = [2, 3, 4, 6, 10];
onsetsNeural    = 10;


%% Exponential

ID  = 'CRM_MDP_Prep_Exp';

% Define probability of target occurance for each expectation level
ht      = 1 : nLevels; % time vector
base    = 0; % Minimum value
peak    = 5; % Maximum value
hz_ori  = exp(ht);
hz      = hz_ori ./ (max(hz_ori) - min(hz_ori)) * (peak - base);
hz      = hz - min(hz) + base;

% Run one simulation
CRM_MDP_SpecifyModelPrep(hz, onsets, ID, n, 0, 1, []);

% Simulate EEG response
CRM_MDP_SpecifyModelPrep(hz, onsetsNeural, sprintf('%s_Neural', ID), ...
    n, 1, [], 16);


%% Exponential CDF

ID  = 'CRM_MDP_Prep_ExpCDF';
onsets = [2, 3, 4, 6, 10];

% Define probability of target occurance for each expectation level
ht      = linspace(0, 2, nLevels); % time vector
base    = 0; % Minimum value
peak    = 5; % Maximum value
hz_ori  = cdf('Exponential', ht, 1);
hz      = hz_ori ./ (max(hz_ori) - min(hz_ori)) * (peak - base) + base;

% Run one simulation
CRM_MDP_SpecifyModelPrep(hz, onsets, ID, n, 0, 1, []);

% Simulate EEG response
CRM_MDP_SpecifyModelPrep(hz, onsetsNeural, sprintf('%s_Neural', ID), ...
    n, 1, [], 16);


%% Linear

ID  = 'CRM_MDP_Prep_Lin';
onsets = [2, 3, 4, 6, 10];

% Define probability of target occurance for each expectation level
base    = 0; % Minimum value
peak    = 5; % Maximum value
hz      = linspace(base, peak, nLevels); % time vector

% Run one simulation
CRM_MDP_SpecifyModelPrep(hz, onsets, ID, n, 0, 1, []);

% Simulate EEG response
CRM_MDP_SpecifyModelPrep(hz, onsetsNeural, sprintf('%s_Neural', ID), ...
    n, 1, [], 16);


%% Uniform (high precision)

ID  = 'CRM_MDP_Prep_UniHi';
onsets = [2, 3, 4, 6, 10];

% Define probability of target occurance for each expectation level
hz  = 4 * ones(1, nLevels);

% Run one simulation
CRM_MDP_SpecifyModelPrep(hz, onsets, ID, n, 0, 1, []);

% Simulate EEG response
CRM_MDP_SpecifyModelPrep(hz, onsetsNeural, sprintf('%s_Neural', ID), ...
    n, 1, [], 16);


%% Uniform (low precision)

ID  = 'CRM_MDP_Prep_UniHi';
onsets = [2, 3, 4, 6, 10];

% Define probability of target occurance for each expectation level
hz  = ones(1, nLevels);

% Run one simulation
CRM_MDP_SpecifyModelPrep(hz, onsets, ID, n, 0, 1, []);

% Simulate EEG response
CRM_MDP_SpecifyModelPrep(hz, onsetsNeural, sprintf('%s_Neural', ID), ...
    n, 1, [], 16);


