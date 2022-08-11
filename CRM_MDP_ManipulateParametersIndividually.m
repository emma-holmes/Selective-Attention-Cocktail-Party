function CRM_MDP_ManipulateParametersIndividually(n)
% Active inference for cocktail paty listening
%__________________________________________________________________________
% 
% Inputs:
%   n           Integer that indicates which set of simulations to run
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
Aw_prec     = linspace(0, 5, 26);
Ac_prec     = linspace(0, 5, 26);
B_prec      = linspace(0, 5, 26);
Awu_prec    = linspace(0, 1024, 26);
alpha       = 1;
nreps       = 48;


% Manipulate each parameter individually, while keeping the other values
% constant
if n == 6001 % (Specified by input value)
    for aw = 1 : length(Aw_prec)
        CRM_MDP_SpecifyModel(Aw_prec(aw), 1024, 1024, 0, alpha, nreps, n);
        n = n + 1;
    end
    
elseif n == 6051
    for ac = 1 : length(Ac_prec)
        CRM_MDP_SpecifyModel(1024, Ac_prec(ac), 1024, 0, alpha, nreps, n);
        n = n + 1;
    end
    
elseif n == 6101
    for sf = 1 : length(B_prec)
        CRM_MDP_SpecifyModel(1024, 1024, B_prec(sf), 0, alpha, nreps, n);
        n = n + 1;
    end
    
elseif n == 6151
    for awu = 1 : length(Awu_prec)
        CRM_MDP_SpecifyModel(1024, 1024, 1024, Awu_prec(awu), alpha, ...
            nreps, n);
        n = n + 1;
    end
end

