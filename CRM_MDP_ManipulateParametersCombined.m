function CRM_MDP_ManipulateParametersCombined(n)
% Active inference for cocktail paty listening
%__________________________________________________________________________
% 
% Inputs:
%   n           Integer that indicates which simulations to run
% 
% This script is not designed for easy re-use. It was written to run
% particular combinations of parameters on a cluster. The 'n' input
% determines which combinations of parameters are run, so they can be run
% in parallel using the same script.
%
% This is the script used in the following paper that is used to 'break'
% various parts of the optimal model and examine effects on behaviour:
% Holmes, E., Parr, T., Griffiths, T. D. & Friston, K. J. (2021). Active 
% inference, selective attention, and the cocktail party problem. Neurosci. 
% Biobehav. Rev. 131, 1288–1304.

% Emma Holmes


% Define all parameter values to simulate
Aw_prec     = linspace(0, 2, 21);
Ac_prec     = 1024;
B_prec      = 1024;
Awu_prec    = linspace(0, 2, 21);
alpha       = 1;
nreps       = 48;


% Manipulate Aw_prec and Awu_prec parameters in combination, while keeping
% the other parameters constant (run one combination at a time)
count = 8000;
for aw = 1 : length(Aw_prec)
    for awu = 1 : length(Awu_prec)
        count = count + 1;
        if (count == n)
            aw_curr = aw;
            awu_curr = awu;
        end
    end
end
CRM_MDP_SpecifyModel(Aw_prec(aw_curr), Ac_prec, B_prec, ...
    Awu_prec(awu_curr), alpha, nreps, n);
