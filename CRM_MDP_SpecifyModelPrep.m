function CRM_MDP_SpecifyModelPrep(hz, onsets, ID, n, disp_plots, alpha, tau)
% Active inference for cocktail paty listening
%__________________________________________________________________________
%
% Inputs:
%   hz          10-element vector defining the temporal function (i.e., 
%               probability of target occurance for each expectation
%               level).
%   onsets      Vector specifying all cue-target intervals (i.e., epoch 
%               numbers to present phrases) to simulate under the model.
%   ID          String specifying your name for this model, which is used
%               when saving output files.
%   n           Number to use as an identifier when saving output file
%               (useful when running the script on a cluster, for example).
%   disp_plots  Boolean specifying whether or not to plot graphs.
%   alpha       Precision of action selection. Pass empty array to use 
%               default value.
%   tau         Time constant (determines spread of simulated ERPs). Pass 
%               empty array to use default value.
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
% a temporal function (hz). This script loops through different 
% cue-target intervals and saves an output file for each interval. The
% temporal function is an input to the script.
%
% This is the script used in the following paper to simulate preparatory
% attention:
% Holmes, E., Parr, T., Griffiths, T. D. & Friston, K. J. (2021). Active 
% inference, selective attention, and the cocktail party problem. Neurosci. 
% Biobehav. Rev. 131, 1288–1304.
 
% Emma Holmes
 
% set up and preliminaries
%==========================================================================
fprintf('\n\nInitialising...');
startTime	= tic;
LASTN   	= maxNumCompThreads(1);
outDir      = pwd;

% Use defauls in case not entered into script
if nargin < 6
    alpha = [];
end
if nargin < 7
    tau = [];
end

% Define all possible responses
colours     = {'Red', 'Green', 'Blue', 'White'};
numbers     = {'1', '2', '3', '4'};
resps_words = [[reshape(repmat(colours, [4, 1]), 1, []); ...
    repmat(numbers, [1, 4])], {'null'; 'null'}]';
resps_IDs   = [[reshape(repmat(1:4, [4, 1]), 1, []); ...
    repmat(1:4, [1, 4])], [5; 5]]';

% Loop through onsets
for t = 1:length(onsets) 
    fprintf('\n\nRunning onset %d of %d...', t, length(onsets)); 
    talkerOnset     = onsets(t);
    trialTime       = tic;
    
    % Prior beliefs about initial states: d
    % --------------------------------------
    d{1} = [1 0 0 0 0 0 0 0 0 0]';% expect target: scale from 1-10
    d{2} = [.25 .25 .25 .25]';	% content of attended 1: {'red', 'green', 'blue', 'white'}
    d{3} = [.25 .25 .25 .25]';	% content of attended 2: {'1', '2', '3', '4'}
    d{4} = [.5 .5]';            % attention:    {'left', 'right'}
    d{5} = [zeros(1,16) 1]';  	% response:     {'red 1', 'red 2', ... 'green 1', ... 'blue 1', ... 'white 1', 'null'}
    
    % probabilistic mapping from hidden states to outcomes: A
    % --------------------------------------------------------
    % Determine number of options per state
    Nf    = numel(d);               % Nf = Number of states
    for f = 1:Nf
        Ns(f) = numel(d{f});        % Ns = Number of options for each state
    end
    
    % Outcomes: Left talker word 1 (r/g/b/w/n), right talker word 1, left 
    % talker word 2 (1/2/3/4/n), right talker word 2, cue (l/r), feedback
    No    = [5, 5, 5, 5, 2, 3];   	% No = Number of options for each outcome
    Ng    = numel(No);              % Ng = Number of outcomes
    
    % Initialise A-matrices
    for g = 1:Ng
        A{g} = zeros([No(g),Ns]);
        a{g} = zeros([No(g),Ns]);
    end
    A{6}(2,:,:,:,:,1:16) = ones([1,10,4,4,2,16]);	% incorrect starts with matrix of ones
    a{6}(2,:,:,:,:,1:16) = ones([1,10,4,4,2,16]);	% incorrect starts with matrix of ones
    
    % set left and right colour and number words randomly
    l1_vec = [1,2,3,4];
    l1 = randsample(l1_vec,1);
    r1_vec = l1_vec(l1_vec~=l1);
    r1 = randsample(r1_vec,1);
    l2_vec = [1,2,3,4];
    l2 = randsample(l2_vec,1);
    r2_vec = l2_vec(l2_vec~=l2);
    r2 = randsample(r2_vec,1);
    
    for f1 = 1:Ns(1)                % f1 = expect target
        for f2 = 1:Ns(2)            % f2 = attended content word 1
            for f3 = 1:Ns(3)        % f3 = attended content word 2
                for f4 = 1:Ns(4)    % f4 = attention
                    for f5 = 1:Ns(5)% f5 = response
                        
                        % Calculate time-dependent values
                        Aw_nEl 	= No(1);
                        Aw_odv  = spm_softmax([1, zeros(1, Aw_nEl-1)]', hz(f1));
                        Aw_div  = Aw_odv(1);
                        Aw_odv  = Aw_odv(2);
                        
                        
                        % A{1} = outcome 1: left talker word 1 (r/g/b/w/n)
                        %=================================================
                        if (f4 == 1)  % only if left talker attended
                            for g1 = 1:No(1)
                                if (g1 == f2)
                                    a{1}(g1,f1,f2,f3,f4,f5) = Aw_div;
                                else
                                    a{1}(g1,f1,f2,f3,f4,f5) = Aw_odv;
                                end
                            end
                        else
                            for g1 = 1:No(1)
                                a{1}(g1,f1,f2,f3,f4,f5) = 1 / No(1);
                            end
                        end
                        
                        % Always infinite precision for actual states of the world
                        if (f1 >= talkerOnset) % only if time window 5                            
                            A{1}(l1,f1,f2,f3,f4,f5) = 1;
                        else                            
                            A{1}(5,f1,f2,f3,f4,f5) = 1;                            
                        end
                        
                        
                        % A{2} = outcome 2: right talker word 1 (r/g/b/w/n)
                        %==================================================
                        if (f4 == 2)  % only if right talker attended
                            for g2 = 1:No(2)
                                if (g2 == f2)
                                    a{2}(g2,f1,f2,f3,f4,f5) = Aw_div;
                                else
                                    a{2}(g2,f1,f2,f3,f4,f5) = Aw_odv;
                                end
                            end
                        else
                            for g2 = 1:No(2)
                                a{2}(g2,f1,f2,f3,f4,f5) = 1 / No(2);
                            end
                        end
                        
                        % Always infinite precision for actual states of the world
                        if (f1 >= talkerOnset) % only if time window 5                            
                            A{2}(r1,f1,f2,f3,f4,f5) = 1;                            
                        else
                            A{2}(5,f1,f2,f3,f4,f5) = 1;                            
                        end
                        
                        
                        % A{3} = outcome 3: left talker word 2 (1/2/3/4/n)
                        %=================================================
                        if (f4 == 1)  % only if left talker attended
                            for g1 = 1:No(3)
                                if (g1 == f3)
                                    a{3}(g1,f1,f2,f3,f4,f5) = Aw_div;
                                else
                                    a{3}(g1,f1,f2,f3,f4,f5) = Aw_odv;
                                end
                            end
                        else
                            for g1 = 1:No(3)
                                a{3}(g1,f1,f2,f3,f4,f5) = 1 / No(3);
                            end
                        end
                        
                        % Always infinite precision for actual states of the world
                        if (f1 >= talkerOnset) % only if time window 5                            
                            A{3}(l2,f1,f2,f3,f4,f5) = 1;                            
                        else
                            A{3}(5,f1,f2,f3,f4,f5) = 1;
                        end
                        
                        
                        % A{4} = outcome 4: right talker word 2 (1/2/3/4/n)
                        %==================================================
                        if (f4 == 2)  % only if right talker attended
                            for g2 = 1:No(4)
                                if (g2 == f3)
                                    a{4}(g2,f1,f2,f3,f4,f5) = Aw_div;
                                else
                                    a{4}(g2,f1,f2,f3,f4,f5) = Aw_odv;
                                end
                            end
                        else
                            for g2 = 1:No(4)
                                a{4}(g2,f1,f2,f3,f4,f5) = 1 / No(4);
                            end
                        end
                        
                        % Always infinite precision for actual states of the world
                        if (f1 >= talkerOnset) % only if time window 5                            
                            A{4}(r2,f1,f2,f3,f4,f5) = 1;                            
                        else                            
                            A{4}(5,f1,f2,f3,f4,f5) = 1;                            
                        end
                        
                        
                        % A{5} = outcome 5: visual cue (l/r)
                        %===================================
                        a{5}(f4,f1,f2,f3,f4,f5) = 1;
                        A{5}(f4,f1,f2,f3,f4,f5) = 1;
                        
                        
                        % A{6} = outcome 6: feedback (correct/incorrect/no response)
                        %===========================================================
                        if (f2 == resps_IDs(f5,1)) && (f3 == resps_IDs(f5,2))
                            a{6}(1,f1,f2,f3,f4,f5) = 1; % correct state
                            a{6}(2,f1,f2,f3,f4,f5) = 0; % incorrect state
                        end
                        a{6}(3,f1,f2,f3,f4,17) = 1;
                        
                        if (f4 == 1)
                            if (resps_IDs(f5,1) == l1) && (resps_IDs(f5,2) == l2)
                                A{6}(1,f1,f2,f3,f4,f5) = 1;
                                A{6}(2,f1,f2,f3,f4,f5) = 0;
                            end
                        elseif (f4 == 2)
                            if (resps_IDs(f5,1) == r1) && (resps_IDs(f5,2) == r2)
                                A{6}(1,f1,f2,f3,f4,f5) = 1;
                                A{6}(2,f1,f2,f3,f4,f5) = 0;
                            end
                        end                        
                        A{6}(3,f1,f2,f3,f4,17) = 1;
                        
                    end
                end
            end
        end
    end
    
    for g = 1:Ng
        a{g} = double(a{g})* 1024;
        A{g} = double(A{g});
    end
    
    
    % controlled transitions: B{f} for each factor
    %----------------------------------------------
    for f = 1:Nf
        B{f} = eye(Ns(f));
    end
    
    
    % B{f} for 'when' state: move to next time point
    %------------------------------------------------
    B{1} = zeros(Ns(1));
    B{1}(2,1) = 1;
    B{1}(3,2) = 1;
    B{1}(4,3) = 1;
    B{1}(5,4) = 1;
    B{1}(6,5) = 1;
    B{1}(7,6) = 1;
    B{1}(8,7) = 1;
    B{1}(9,8) = 1;
    B{1}(10,9) = 1;
    B{1}(10,10) = 1;   


    % B{f} for 'response' state: make dependent upon action
    %-------------------------------------------------------
    B{5} = zeros(Ns(5), Ns(5), Ns(5));
    for f5 = 1:Ns(5)
        B{5}(:,:,f5) = eye(Ns(5));
        B{5}(Ns(5),Ns(5),f5) = 0;
        B{5}(f5,Ns(5),f5) = 1;
    end
    
    
    % priors (utility) over outcomes: C 
    %-----------------------------------
    T         = 14; % Time
    C{1}      = zeros(No(1),T);
    C{2}      = zeros(No(2),T);
    C{3}      = zeros(No(3),T);
    C{4}      = zeros(No(4),T);
    C{5}      = zeros(No(5),T);
    C{6}      = repmat([2; -4; -3], [1, T]);
    
    
    % allowable policies (here, specified as all actions for the trial): V
    %----------------------------------------------------------------------
    Np          = (T-1)*(Ns(5)-1);
    V = ones(T-1, Np, Nf);
    V(:,:,Nf)    = Ns(5);
    for index = 1:T-1
        for x =  (T-index):(T-1)
            V(x,1+(Ns(5)-1)*(index-1):(Ns(5)-1)*index,Nf)  = 1:16;
        end
    end
    
    %% MDP Structure
    %======================================================================
    mdp.T = T;                      % number of moves
    mdp.V = V;                      % allowable policies
    mdp.a = a;
    mdp.A = A;                      % observation model
    mdp.B = B;                      % transition probabilities
    mdp.C = C;                      % preferred outcomes
    mdp.D = d;                      % prior over initial states
    
    mdp.Aname = {'left 1','right 1','left 2','right 2',...
        'cue','feedback'}; % outcome labels
    mdp.Bname = {'expect target','content 1','content 2',...
        'attention','response'}; % state labels
    
    % Check whether to change default alpha and tau
    if ~isempty(alpha)
        mdp.alpha = alpha;
    end
    if ~isempty(tau)
        mdp.tau = tau;
    end
    
    % Run a single trial
    fprintf('\nInverting model...');
    MDP   = spm_MDP_VB_X(mdp);
    
    % Plotting
    %----------
    if disp_plots
        % Plot behaviour
        spm_figure('GetWin','Figure 1'); clf
        spm_MDP_VB_trial(MDP,[1,2,3,4,5],[1,2,3,4,5,6]);
        subplot(3,2,3)
        set(gcf, 'PaperPositionMode', 'auto');    
        filename    = fullfile(outDir, sprintf('%s_PLOT_behaviour_%d_%d', ...
            ID, talkerOnset, n));
        %print(gcf, filename, '-dtiff', '-r600');

        % Plot neural
        spm_figure('GetWin','Figure 2'); clf
        [~,~,LFP_plot_x,LFP_plot_y] = spm_MDP_VB_LFP_edit(MDP,[5;5],1,0,...
            sprintf('%d_tau%d',n,mdp.tau));     
        [~, peak_idx]   = findpeaks(LFP_plot_y);
        peak_idx        = peak_idx([1:2:23,24]);
        peak_vals       = LFP_plot_y(peak_idx); 
        peak_times      = LFP_plot_x(peak_idx);
        figure; 
        plot(LFP_plot_x, LFP_plot_y, 'LineWidth', 1.5, 'Color', [.5, .5, .5]);
        hold on; 
        try
            fitobject       = fit(peak_times',peak_vals,'cubicinterp');
            hC = plot(fitobject, 'r-'); 
            set(hC, 'LineWidth', 1.5); 
        end
        format_graph(gca, 'Time (seconds)', 'Amplitude (a.u.)'); 
        hold on; 
        set(gcf, 'Position', [4, 254, 985, 325]); 
        ylim([-0.01, 0.025]); 
        legend('off'); 
        xvals = get(gca, 'xlim'); 
        yvals = get(gca, 'ylim'); 
        hLine1 = plot(xvals, [0, 0], 'k--', 'LineWidth', 1.5); 
        hLine2 = plot([2.25, 2.25], yvals, 'k--', 'LineWidth', 1.5); 
        uistack([hLine1, hLine2], 'bottom'); 
        set(gcf, 'PaperPositionMode', 'auto');
        filename    = fullfile(outDir, sprintf('%s_PLOT_ERP_%d_%d', ...
            ID, talkerOnset, n));
        %print(gcf, filename, '-dtiff', '-r600'); 
    end
    
    % Save all variables    
    filename    = fullfile(outDir, sprintf('%s_outputs_%d_%d', ...
        ID, talkerOnset, n));
    fprintf('\nSaving outputs to file: %s', filename);
    save(filename);
    
    % Print time taken
    duration = toc(trialTime) / 60;
    fprintf('\n...finished (time taken = %.2f minutes)', duration);
end

% Print time taken to run
duration = toc(startTime) / 60;
fprintf('\n...Script finished (time taken = %.2f minutes).\n', duration);