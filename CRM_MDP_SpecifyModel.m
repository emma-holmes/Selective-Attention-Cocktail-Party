function [allMDPs, allAC_av, allAC] = CRM_MDP_SpecifyModel(Aw_prec, Ac_prec, B_prec, Awu_prec, alpha, nreps, num)
% Active inference for cocktail paty listening
%__________________________________________________________________________
%
% Inputs:
%   Aw_prec   	Attentional focus parameter (A-matrix values for attended
%               words).
%   Ac_prec     Spatial cue parameter (A-matrix values for spatial
%               attention / visual cue).
%   B_prec      B-matrix parameter for spatial attention state.
%   Awu_prec    Unattended words parameter (A-matrix values for unattended
%               words).
%   alpha       Precision parameter for policy selection.
%   nreps       Integer specifying number of repetitions ('trials') to run.
%   num         Number to use as an identifier when saving output file
%               (useful when running the script on a cluster, for example).
% 
% Ouptuts:
%   allMDPs     Cell array of MDP structures output from spm_MDP_VB_X.m for
%               each repetition.
%   allAC_av    Average accuracy over all repetitions.
%   allAC       Accuracy for each repetition.
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
% This is the script used in the following paper that is used to 'break'
% various parts of the optimal model and examine effects on behaviour:
% Holmes, E., Parr, T., Griffiths, T. D. & Friston, K. J. (2021). Active 
% inference, selective attention, and the cocktail party problem. Neurosci. 
% Biobehav. Rev. 131, 1288–1304.
 
% Emma Holmes
 
% set up and preliminaries
%==========================================================================
startTime   = tic;
LASTN       = maxNumCompThreads(1); 
outDir      = pwd;

% Define all possible responses
colours     = {'Red', 'Green', 'Blue', 'White'};
numbers     = {'1', '2', '3', '4'};
resps_words = [[reshape(repmat(colours, [4, 1]), 1, []); ...
    repmat(numbers, [1, 4])], {'null'; 'null'}]';
resps_IDs   = [[reshape(repmat(1:4, [4, 1]), 1, []); ...
    repmat(1:4, [1, 4])], [5; 5]]';

% Randomly select left and right colour and number words
l1_vec = [1,2,3,4];
l1 = randsample(l1_vec,1);
r1_vec = l1_vec(l1_vec~=l1);
r1 = randsample(r1_vec,1);
l2_vec = [1,2,3,4];
l2 = randsample(l2_vec,1);
r2_vec = l2_vec(l2_vec~=l2);
r2 = randsample(r2_vec,1);

% Initialise variables
allMDPs     = cell(length(Aw_prec), length(Ac_prec), length(B_prec), ...
    length(Awu_prec), length(alpha), nreps);
allAC       = zeros(length(Aw_prec), length(Ac_prec), length(B_prec), ...
    length(Awu_prec), length(alpha), nreps);
allErrsMk 	= zeros(length(Aw_prec), length(Ac_prec), length(B_prec), ...
    length(Awu_prec), length(alpha), nreps); % Masker errors
allErrsMx 	= zeros(length(Aw_prec), length(Ac_prec), length(B_prec), ...
    length(Awu_prec), length(alpha), nreps); % Mix errors
allErrsRn  	= zeros(length(Aw_prec), length(Ac_prec), length(B_prec), ...
    length(Awu_prec), length(alpha), nreps); % Random errors
sameWordFlag= zeros(length(Aw_prec), length(Ac_prec), length(B_prec), ...
    length(Awu_prec), length(alpha), nreps); % Same target and masker words
transitFlag = zeros(length(Aw_prec), length(Ac_prec), length(B_prec), ...
    length(Awu_prec), length(alpha), nreps); % Not consistent over time
rep_ind     = 6; % Dimension containing repetition index
 
% Prior beliefs about initial states: d
% --------------------------------------
d{1}    = [1 1 1 1]';       % attended colour word: {'red', 'blue', 'green', 'white'}
d{2}    = [1 1 1 1]';       % attended number word: {'1', '2', '3', '4'}
d{3}    = [1 1]';           % spatial attention: {'left', 'right'}
d{4}    = [zeros(1,16) 1]'; % response: {'red 1', 'red 2', ... 'green 1', ... 'blue 1', ... 'white 1', 'null'}

% Calculate number of states for first factor (this is used for a-matrix
% calculations below)
Aw_nEl 	= length(d{1});
Ac_nEl 	= length(d{2});
B_nEl 	= length(d{3});
Awu_nEl = Aw_nEl;

% Define number of factors and states
Nf    = numel(d);               % Nf = Number of factors
for f = 1 : Nf
    Ns(f) = numel(d{f});        % Ns = Number of states for each factor
end

% Outcomes: Left colour (r/g/b/w), right colour, left number (1/2/3/4), 
% right number, cue (l/r), accuracy (y/n)
No    = [4, 4, 4, 4, 2, 2];  	% No = Number of options for each outcome
Ng    = numel(No);              % Ng = Number of outcomes

% Initialise A and B matrices
a           = cell(1, Ng);
A           = cell(1, Ng);
B           = cell(1, Nf);

% Loop through A-precision (word) parameters
countTrials = 0;
for aw = 1 : length(Aw_prec)
    fprintf('\n\nA-prec (word) %d of %d...', aw, length(Aw_prec));

    % Attentional focus parameter  
    Aw_odv = spm_softmax([1, zeros(1, Aw_nEl-1)]', Aw_prec(aw));
    Aw_div = Aw_odv(1); % Diagonal value
    Aw_odv = Aw_odv(2); % Off-diagonal value

    % Loop through A-precision (cue) parameters
    for ac = 1 : length(Ac_prec)
        fprintf('\nA-prec (cue) %d of %d...', ac, length(Ac_prec));  	
        Ac_odv = spm_softmax([1, zeros(1, Ac_nEl-1)]', Ac_prec(ac));
        Ac_div = Ac_odv(1); % Diagonal value
        Ac_odv = Ac_odv(2); % Off-diagonal value
         
        % Loop through B-precision parameters
        for sf = 1 : length(B_prec)
            fprintf('\nB-prec %d of %d...', sf, length(B_prec));
            B_odv = spm_softmax([1, zeros(1, B_nEl-1)]', B_prec(sf));
            B_div = B_odv(1); % Diagonal value
            B_odv = B_odv(2); % Off-diagonal value

            % Loop through A-precision (unattended word) parameters
            for awu = 1 : length(Awu_prec)
                fprintf('\nA-prec (unattended word) %d of %d...', awu, length(Awu_prec));
                Awu_odv = spm_softmax([1, zeros(1, Awu_nEl-1)]', Awu_prec(awu));
                Awu_div = Awu_odv(1); % Diagonal value 
                Awu_odv = Awu_odv(2); % Off-diagonal value

                % Loop through alpha parameters
                for alph = 1 : length(alpha)
                    fprintf('\nAlpha %d of %d...', alph, length(alpha)); 

                    % Loop through trials
                    for n = 1 : nreps
                        repeat = 1;
                        while repeat    
                            fprintf('\n...trial %d of %d', n, nreps);

                            % probabilistic mapping from hidden states to outcomes: A
                            % --------------------------------------------------------

                            % Initialise A-matrices
                            for g = 1:Ng
                                a{g} = zeros([No(g), Ns]);
                                A{g} = zeros([No(g), Ns]);
                            end        

                            % Change for A{3} variable based on loop
                            a{5}    = ones([No(g), Ns]) * Ac_odv;
                            A{5}    = ones([No(g), Ns]);

                            % Change for incorrect state
                            a{6}(2,:,:,:,:) = ones([1,Ns]);
                            A{6}(2,:,:,:,:) = ones([1,Ns]);

                            for f1 = 1:Ns(1)            % f1 = attended colour
                                for f2 = 1:Ns(2)        % f2 = attended number
                                    for f3 = 1:Ns(3)    % f3 = spatial attention
                                        for f4 = 1:Ns(4)% f4 = response

                                            % A{1} = outcome 1: left colour (r/g/b/w)
                                            %========================================
                                            if (f3 == 1)        % left talker attended
                                                for o = 1 : No(1)
                                                    if (o == f1)
                                                        a{1}(o,f1,f2,f3,f4) = Aw_div;
                                                    else
                                                        a{1}(o,f1,f2,f3,f4) = Aw_odv;
                                                    end
                                                end
                                            elseif (f3 == 2)  	% left talker unattended
                                                for o = 1 : No(1)
                                                    if (o == f1)
                                                        a{1}(o,f1,f2,f3,f4) = Awu_div;
                                                    else
                                                        a{1}(o,f1,f2,f3,f4) = Awu_odv;
                                                    end
                                                end
                                            end
                                            
                                            % Ensure generative process
                                            % specifies particular outcomes
                                            A{1}(l1,f1,f2,f3,f4) = 1;


                                            % A{2} = outcome 2: right colour (r/g/b/w)
                                            %=========================================
                                            if (f3 == 2)        % right talker attended
                                                for o = 1 : No(2)
                                                    if (o == f1)
                                                        a{2}(o,f1,f2,f3,f4) = Aw_div;
                                                    else
                                                        a{2}(o,f1,f2,f3,f4) = Aw_odv;
                                                    end
                                                end
                                            elseif (f3 == 1)	% right talker unattended
                                                for o = 1 : No(2)
                                                    if (o == f1)
                                                        a{2}(o,f1,f2,f3,f4) = Awu_div;
                                                    else
                                                        a{2}(o,f1,f2,f3,f4) = Awu_odv;
                                                    end
                                                end
                                            end
                                            
                                            % Ensure generative process
                                            % specifies particular outcomes
                                            A{2}(r1,f1,f2,f3,f4) = 1;

                                            % A{3} = outcome 3: left number (1/2/3/4)
                                            %========================================
                                            if (f3 == 1)        % left talker attended
                                                for o = 1 : No(3)
                                                    if (o == f2)
                                                        a{3}(o,f1,f2,f3,f4) = Aw_div;
                                                    else
                                                        a{3}(o,f1,f2,f3,f4) = Aw_odv;
                                                    end
                                                end
                                            elseif (f3 == 2)	% left talker unattended
                                                for o = 1 : No(3)
                                                    if (o == f2)
                                                        a{3}(o,f1,f2,f3,f4) = Awu_div;
                                                    else
                                                        a{3}(o,f1,f2,f3,f4) = Awu_odv;
                                                    end
                                                end
                                            end
                                            
                                            % Ensure generative process
                                            % specifies particular outcomes
                                            A{3}(l2,f1,f2,f3,f4) = 1;


                                            % A{4} = outcome 4: right number (1/2/3/4)
                                            %=========================================
                                            if (f3 == 2)        % right talker attended
                                                for o = 1 : No(4)
                                                    if (o == f2)
                                                        a{4}(o,f1,f2,f3,f4) = Aw_div;
                                                    else
                                                        a{4}(o,f1,f2,f3,f4) = Aw_odv;
                                                    end
                                                end
                                            elseif (f3 == 1)	% right talker unattended
                                                for o = 1 : No(4)
                                                    if (o == f2)
                                                        a{4}(o,f1,f2,f3,f4) = Awu_div;
                                                    else
                                                        a{4}(o,f1,f2,f3,f4) = Awu_odv;
                                                    end
                                                end
                                            end
                                            
                                            % Ensure generative process
                                            % specifies particular outcomes
                                            A{4}(r2,f1,f2,f3,f4) = 1;

                                            % A{5} = outcome 5: visual cue (l/r/n)
                                            %=====================================
                                            a{5}(f3,f1,f2,f3,f4) = Ac_div;
                                            A{5}(f3,f1,f2,f3,f4) = 1;

                                            % A{6} = outcome 6: accuracy (correct/incorrect)
                                            %===============================================
                                            if f4 == ((f1-1)*Ns(1)+f2)
                                                % Check if response state is consistent
                                                % with attended words
                                                a{6}(1,f1,f2,f3,f4) = 1;
                                                a{6}(2,f1,f2,f3,f4) = 0;
                                                A{6}(1,f1,f2,f3,f4) = 1;
                                                A{6}(2,f1,f2,f3,f4) = 0;
                                            end
                                        end
                                    end
                                end
                            end
                            for g = 1:Ng
                                a{g} = double(a{g}) * 1024 + exp(-4);
                                A{g} = double(A{g});
                            end


                            % controlled transitions: B{f} for each factor
                            % ---------------------------------------------
                            for f = 1:Nf
                                B{f} = eye(Ns(f));
                            end

                            % B{f} for 'response' state: make dependent upon policy
                            %-------------------------------------------------------
                            B{4} = zeros(Ns(4), Ns(4), Ns(4));
                            for f4 = 1 : Ns(4)
                                B{4}(:,f4,f4) = ones(1, Ns(4));
                            end
                            B{4} = permute(B{4}, [2,1,3]);

                            % B{f} for spatial attention state: incorporate B-precision
                            B{3} = [B_div, B_odv; B_odv, B_div];

                            % Increase concentration parameters
                            for f = 1:Nf
                                B{f} = B{f} * 1024 + exp(-4);
                            end
                            

                            % allowable policies (here, specified as all actions for the trial): V
                            %----------------------------------------------------------------------
                            T           = 2;
                            Np          = Ns(end)-1;
                            V           = ones(T-1, Np, Nf); % (policy null)
                            V(1,:,end)  = 1:Np;


                            % priors (utility) over outcomes: C 
                            %-----------------------------------
                            C{1}      = zeros(No(1),T);
                            C{2}      = zeros(No(2),T);
                            C{3}      = zeros(No(3),T);
                            C{4}      = zeros(No(4),T);
                            C{5}      = zeros(No(5),T);
                            C{6}      = repmat([2; -4], [1, T]);


                            % MDP Structure
                            %==============================================
                            mdp.T = T;                      % number of moves
                            mdp.V = V;                      % allowable policies
                            mdp.a = a;                      % observation model
                            mdp.A = A;                      % observation model
                            mdp.B = B;                      % transition probabilities
                            mdp.C = C;                      % preferred outcomes
                            mdp.D = d;                      % prior over initial states

                            mdp.Aname = {'left col', 'right col', ...
                                'left num', 'right num', 'cue', 'accuracy'};
                            mdp.Bname = {'col word', 'num word', ...
                                'spatial', 'response'};

                            % Change alpha value (default = 512)
                            mdp.alpha = alpha(alph); % precision of policy selection

                            % Set time constant
                            mdp.tau = 4;


                            % Invert
                            %==============================================
                            MDP   = spm_MDP_VB_X(mdp);
                            allMDPs{aw,ac,sf,awu,alph,n} = MDP;

                            % Check outcomes
                            repeat = 0;
                            if (MDP.o(1,1) == MDP.o(2,1)) || (MDP.o(3,1) == MDP.o(4,1))
                                sameWordFlag(aw,ac,sf,awu,alph,n) = 1;
                                repeat = 1;
                                fprintf(' (same word flag, repeating...)');%\nMDP.o(1,1) = %.d, MDP.o(2,1) = %d, \nMDP.o(3,1) = %d, MDP.o(4,1) = %d', ...
%                                     MDP.o(1,1), MDP.o(2,1), ...
%                                     MDP.o(3,1), MDP.o(4,1)); 
%                                     % ^ Uncomment the above to print more detail
                            else
                                sameWordFlag(aw,ac,sf,awu,alph,n) = 0;
                            end
                            if (MDP.o(1,1) ~= MDP.o(1,2)) ...
                                    || (MDP.o(2,1) ~= MDP.o(2,2))  ...
                                    || (MDP.o(3,1) ~= MDP.o(3,2))  ...
                                    || (MDP.o(4,1) ~= MDP.o(4,2))  ...
                                    || (MDP.o(5,1) ~= MDP.o(5,2))
                                transitFlag(aw,ac,sf,awu,alph,n) = 1;
                                repeat = 1;
                                fprintf(' (transit flag, repeating...)');%\nMDP.o(1,1) = %.d, MDP.o(1,2) = %.d, \nMDP.o(2,1) = %d, MDP.o(2,2) = %.d, \nMDP.o(3,1) = %d, MDP.o(3,2) = %.d, \nMDP.o(4,1) = %d, MDP.o(4,2) = %.d, \nMDP.o(5,1) = %d, MDP.o(5,2) = %.d))', ...
%                                     MDP.o(1,1), MDP.o(1,2), ...
%                                     MDP.o(2,1), MDP.o(2,2), ...
%                                     MDP.o(3,1), MDP.o(3,2), ...
%                                     MDP.o(4,1), MDP.o(4,2), ...
%                                     MDP.o(5,1), MDP.o(5,2));
%                                 if (MDP.o(1,1) ~= MDP.o(1,2))
%                                     fprintf('\nMismatch in MDP.o(1,1) and MDP.o(1,2)');
%                                 end
%                                 if (MDP.o(2,1) ~= MDP.o(2,2))
%                                     fprintf('\nMismatch in MDP.o(2,1) and MDP.o(2,2)');
%                                 end
%                                 if (MDP.o(3,1) ~= MDP.o(3,2))
%                                     fprintf('\nMismatch in MDP.o(3,1) and MDP.o(3,2)');
%                                 end
%                                 if (MDP.o(4,1) ~= MDP.o(4,2))
%                                     fprintf('\nMismatch in MDP.o(4,1) and MDP.o(4,2)');
%                                 end
%                                 if (MDP.o(5,1) ~= MDP.o(5,2))
%                                     fprintf('\nMismatch in MDP.o(5,1) and MDP.o(5,2)');
%                                 end
                                % ^ Uncomment the above to print more detail
                            else
                                transitFlag(aw,ac,sf,awu,alph,n) = 0;
                            end

                            if ~repeat
                                % Calculate accuracy for this trial               
                                allAC(aw,ac,sf,awu,alph,n) 	= (resps_IDs(MDP.u(4,1),1) == MDP.o(MDP.o(5,1),1)) ...
                                    && (resps_IDs(MDP.u(4,1),2) == MDP.o(MDP.o(5,1)+2,1)); % Time 1
                                if ~allAC(aw,ac,sf,awu,alph,n)
                                    if (resps_IDs(MDP.u(4,1),1) == MDP.o((setdiff([1,2], MDP.o(5,1))),1)) ...
                                            && (resps_IDs(MDP.u(4,1),2) == MDP.o((setdiff([1,2], MDP.o(5,1)))+2,1))
                                        allErrsMk(aw,ac,sf,awu,alph,n) 	= 1;
                                    elseif ((resps_IDs(MDP.u(4,1),1) == MDP.o(MDP.o(5,1),1)) ...
                                            && (resps_IDs(MDP.u(4,1),2) == MDP.o((setdiff([1,2], MDP.o(5,1)))+2,1))) ...
                                            || ((resps_IDs(MDP.u(4,1),1) == MDP.o((setdiff([1,2], MDP.o(5,1))),1)) ...
                                            && (resps_IDs(MDP.u(4,1),2) == MDP.o(MDP.o(5,1)+2,1)))
                                        allErrsMx(aw,ac,sf,awu,alph,n) 	= 1;
                                    else
                                        allErrsRn(aw,ac,sf,awu,alph,n) 	= 1;
                                    end
                                end
                            end
                                
                            % Keep running total of number of trials
                            countTrials = countTrials + 1;
                        end
                    end
                end
            end
        end
    end
end

% Calculate average and standard error
allAC_av    = mean(allAC, rep_ind);
allAC_pc    = 100 * allAC_av;

% Calculate average and standard error - after removing trials with flags
allAC_noFlag= allAC;
allAC_noFlag(logical(sameWordFlag) | logical(transitFlag))	= NaN;
nFlags = sameWordFlag | transitFlag;
nFlags = sum(nFlags, rep_ind);
allAC_av_noFlag    = nanmean(allAC_noFlag, rep_ind);
allAC_pc_noFlag    = 100 * allAC_av_noFlag;

% Calculate error rates
allNErrs    = sum(allAC==0, rep_ind);
allErrsMk_av= sum(allErrsMk, rep_ind) ./ allNErrs;
allErrsMk_p = 100 * allErrsMk_av;
allErrsRn_av= sum(allErrsRn, rep_ind) ./ allNErrs;
allErrsRn_p = 100 * allErrsRn_av;
allErrsMx_av= sum(allErrsMx, rep_ind) ./ allNErrs;
allErrsMx_p = 100 * allErrsMx_av;

% Calculate error rates - after removing trials with flags
allNErrs_noFlag    = sum(allAC_noFlag==0, rep_ind);
allErrsMk_noFlag= allErrsMk;
allErrsMk_noFlag(logical(sameWordFlag) | logical(transitFlag))	= NaN;
allErrsRn_noFlag= allErrsRn;
allErrsRn_noFlag(logical(sameWordFlag) | logical(transitFlag))	= NaN;
allErrsMx_noFlag= allErrsMx;
allErrsMx_noFlag(logical(sameWordFlag) | logical(transitFlag))	= NaN;
allErrsMk_av_noFlag= sum(allErrsMk_noFlag, rep_ind) ./ allNErrs_noFlag;
allErrsMk_p_noFlag = 100 * allErrsMk_av_noFlag;
allErrsRn_av_noFlag= sum(allErrsRn_noFlag, rep_ind) ./ allNErrs_noFlag;
allErrsRn_p_noFlag = 100 * allErrsRn_av_noFlag;
allErrsMx_av_noFlag= sum(allErrsMx_noFlag, rep_ind) ./ allNErrs_noFlag;
allErrsMx_p_noFlag = 100 * allErrsMx_av_noFlag;

% Save variables
date_str    = datestr(now, 'yyyymmdd-HHSS');
script_str  = 'CRM_MDP_SpecifyModel';
filename    = fullfile(outDir, sprintf('%s_outputs_%d_%dreps', ...
    script_str, num, nreps));
fprintf('\n\nSaving output to file: %s\n', filename');
save(filename);

% Print time taken to run
duration = toc(startTime) / 60;
fprintf('\n...Script finished (time taken = %.2f minutes).\n', duration);
