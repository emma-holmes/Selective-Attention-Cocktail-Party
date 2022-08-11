function [allMDPs, allAC_av, allAC] = CRM_MDP_OptimalModel(nreps)
% Active inference for cocktail paty listening
%__________________________________________________________________________
%
% Inputs:
%   nreps       Integer specifying number of repetitions ('trials') to run.
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
% This is the initial model used in the following paper that produces
% perfect performance as a proof of principle:
% Holmes, E., Parr, T., Griffiths, T. D. & Friston, K. J. (2021). Active 
% inference, selective attention, and the cocktail party problem. Neurosci. 
% Biobehav. Rev. 131, 1288–1304.
 
% Emma Holmes
 
% set up and preliminaries
%==========================================================================
startTime = tic;

% Define all possible responses
colours     = {'Red', 'Green', 'Blue', 'White'};
numbers     = {'1', '2', '3', '4'};
resps_words = [[reshape(repmat(colours, [4, 1]), 1, []); ...
    repmat(numbers, [1, 4])], {'Null'; 'Null'}]';
resps_IDs   = [[reshape(repmat(1:4, [4, 1]), 1, []); ...
    repmat(1:4, [1, 4])], [5; 5]]';

% Initialise variables
allMDPs     = cell(1, 1, 1, nreps);
allAC       = zeros(1, 1, 1, nreps);
allErrsMk 	= zeros(1, 1, 1, nreps); % Masker errors
allErrsMx 	= zeros(1, 1, 1, nreps); % Mix errors
allErrsRn  	= zeros(1, 1, 1, nreps); % Random errors
sameWordFlag= zeros(1, 1, 1, nreps); % Same target and masker words
transitFlag = zeros(1, 1, 1, nreps); % Identifies trials not consistent over time
 
% Prior beliefs about initial states (in terms of counts_: D and d
d{1}    = [1 1 1 1]';       % attended colour word: {'red', 'blue', 'green', 'white'}
d{2}    = [1 1 1 1]';       % attended number word: {'1', '2', '3', '4'}
d{3}    = [1 1]';           % spatial attention: {'left', 'right'}
d{4}    = [zeros(1,16) 1]'; % response: {'red 1', 'red 2', ... 'green 1', ... 'blue 1', ... 'white 1', 'null'}

% Define number of factors and states
Nf      = numel(d);         % Nf = Number of factors
for f = 1 : Nf
    Ns(f) = numel(d{f}); 	% Ns = Number of states for each factor
end

% Outcomes: Left colour (r/g/b/w), right colour, left number (1/2/3/4), 
% right number, cue (l/r), accuracy (y/n)
No      = [4, 4, 4, 4, 2, 2];  	% No = Number of options for each outcome
Ng      = numel(No);        	% Ng = Number of outcomes

% Initialise A and B matrices
A   	= cell(1, Ng);
B   	= cell(1, Nf);

% Attentional focus parameter (A-matrix values for attended words)
Aw_odv	= 0; 	% Off-diagonal value
Aw_div	= 1; 	% Diagonal value

% Spatial cue parameter (A-matrix values for spatial attention / visual
% cue)
Ac_odv	= 0; 	% Off-diagonal value
Ac_div	= 1; 	% Diagonal value

% B-matrix parameter for spatial attention state
B_odv	= 0; 	% Off-diagonal value
B_div	= 1; 	% Diagonal value

% Loop through trials
for n = 1 : nreps
    fprintf('\n...trial %d of %d', n, nreps);
    
    % probabilistic mapping from hidden states to outcomes: A
    %--------------------------------------------------------------
    
    % Initialise A-matrices
    for g = 1:Ng
        A{g} = zeros([No(g), Ns]);
    end
    
    % Change for A{3} variable based on loop
    A{5}    = ones([No(g), Ns]) * Ac_odv;
    
    % Change for incorrect state
    A{6}(2,:,:,:,:) = ones([1,Ns]);
    
    for f1 = 1:Ns(1)            % f1 = attended colour
        for f2 = 1:Ns(2)        % f2 = attended number
            for f3 = 1:Ns(3)    % f3 = spatial attention
                for f4 = 1:Ns(4)% f4 = response
                    
                    % A{1} = outcome 1: left colour (r/g/b/w)
                    %==================================================
                    if (f3 == 1)        % left talker attended
                        for o = 1 : No(1)
                            if (o == f1)
                                A{1}(o,f1,f2,f3,f4) = Aw_div;
                            else
                                A{1}(o,f1,f2,f3,f4) = Aw_odv;
                            end
                        end
                    elseif (f3 == 2)	% left talker unattended
                        for o = 1 : No(1)
                            A{1}(o,f1,f2,f3,f4) = 1 / No(1);
                        end
                    end
                    
                    
                    % A{2} = outcome 2: right colour (r/g/b/w)
                    %==================================================
                    if (f3 == 2)        % right talker attended
                        for o = 1 : No(2)
                            if (o == f1)
                                A{2}(o,f1,f2,f3,f4) = Aw_div;
                            else
                                A{2}(o,f1,f2,f3,f4) = Aw_odv;
                            end
                        end
                    elseif (f3 == 1) 	% right talker unattended
                        for o = 1 : No(2)
                            A{2}(o,f1,f2,f3,f4) = 1 / No(2);
                        end
                    end
                    
                    % A{3} = outcome 3: left number (1/2/3/4)
                    %==================================================
                    if (f3 == 1)        % left talker attended
                        for o = 1 : No(3)
                            if (o == f2)
                                A{3}(o,f1,f2,f3,f4) = Aw_div;
                            else
                                A{3}(o,f1,f2,f3,f4) = Aw_odv;
                            end
                        end
                    elseif (f3 == 2)	% left talker unattended
                        for o = 1 : No(3)
                            A{3}(o,f1,f2,f3,f4) = 1 / No(3);
                        end
                    end
                    
                    
                    % A{4} = outcome 4: right number (1/2/3/4)
                    %==================================================
                    if (f3 == 2)        % right talker attended
                        for o = 1 : No(4)
                            if (o == f2)
                                A{4}(o,f1,f2,f3,f4) = Aw_div;
                            else
                                A{4}(o,f1,f2,f3,f4) = Aw_odv;
                            end
                        end
                    elseif (f3 == 1) 	% right talker unattended
                        for o = 1 : No(4)
                            A{4}(o,f1,f2,f3,f4) = 1 / No(4);
                        end
                    end
                    
                    % A{5} = outcome 5: visual cue (l/r/n)
                    %==================================================
                    A{5}(f3,f1,f2,f3,f4) = Ac_div;
                    
                    % A{6} = outcome 6: accuracy (correct/incorrect)
                    %==================================================
                    if f4 == ((f1-1)*Ns(1)+f2)
                        % Check if response state is consistent
                        % with attended words
                        A{6}(1,f1,f2,f3,f4) = 1;
                        A{6}(2,f1,f2,f3,f4) = 0;
                    end
                end
            end
        end
    end
    for g = 1:Ng
        A{g} = double(A{g}) * 1024 + exp(-4);
    end
    
    
    % controlled transitions: B{f} for each factor
    %--------------------------------------------------------------
    for f = 1:Nf
        B{f} = eye(Ns(f));
    end
    
    % B{f} for 'response' state: make dependent upon policy
    %--------------------------------------------------------------
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
    
    % allowable policies (here, specified as the next action) U
    %--------------------------------------------------------------
    T           = 2;
    Np          = Ns(end)-1;
    V           = ones(T-1, Np, Nf); % (policy null)
    V(1,:,end)  = 1:Np;
    
    
    % priors: (utility) C over outcomes
    %--------------------------------------------------------------
    C{1}      = zeros(No(1),T);
    C{2}      = zeros(No(2),T);
    C{3}      = zeros(No(3),T);
    C{4}      = zeros(No(4),T);
    C{5}      = zeros(No(5),T);
    C{6}      = repmat([2; -4], [1, T]);
    
    
    % MDP Structure
    %==============================================================
    mdp.T = T;                      % number of moves
    mdp.V = V;                      % allowable policies
    mdp.A = A;                      % observation model
    mdp.B = B;                      % transition probabilities
    mdp.C = C;                      % preferred outcomes
    mdp.D = d;                      % prior over initial states
    
    mdp.Aname = {'Left Colour', 'Right Colour', 'Left Number', ...
        'Right Number', 'Visual cue', 'Accuracy'};
    mdp.Bname = {'Target colour', 'Target Number', 'Spatial Attention', ...
        'Response'};   
    
    % Define time constant
    mdp.tau = 4;
    
    
    % Invert
    %==============================================================
    MDP   = spm_MDP_VB_X(mdp);
    allMDPs{1,1,1,n} = MDP;
    
    % Calculate accuracy for this trial
    allAC(1,1,1,n) 	= (resps_IDs(MDP.u(4,1),1) == MDP.o(MDP.o(5,1),1)) ...
        && (resps_IDs(MDP.u(4,1),2) == MDP.o(MDP.o(5,1)+2,1));
    if ~allAC(1,1,1,n)
        if (resps_IDs(MDP.u(4,1),1) == MDP.o((setdiff([1,2], MDP.o(5,1))),1)) ...
                && (resps_IDs(MDP.u(4,1),2) == ...
                MDP.o((setdiff([1,2], MDP.o(5,1)))+2,1))
            allErrsMk(aw,ac,sf,n) 	= 1;
        elseif ((resps_IDs(MDP.u(4,1),1) == MDP.o(MDP.o(5,1),1)) ...
                && (resps_IDs(MDP.u(4,1),2) == ...
                MDP.o((setdiff([1,2], MDP.o(5,1)))+2,1))) ...
                || ((resps_IDs(MDP.u(4,1),1) == ...
                MDP.o((setdiff([1,2], MDP.o(5,1))),1)) ...
                && (resps_IDs(MDP.u(4,1),2) == MDP.o(MDP.o(5,1)+2,1)))
            allErrsMx(1,1,1,n) 	= 1;
        else
            allErrsRn(1,1,1,n) 	= 1;
        end
    end
    
    % Check outcomes
    if (MDP.o(1,1) == MDP.o(2,1)) || (MDP.o(3,1) == MDP.o(4,1))
        sameWordFlag(1,1,1,n) = 1;
    end
    if (MDP.o(1,1) ~= MDP.o(1,2)) || (MDP.o(2,1) ~= MDP.o(2,2))  ...
            || (MDP.o(3,1) ~= MDP.o(3,2))  ...
            || (MDP.o(4,1) ~= MDP.o(4,2))  ...
            || (MDP.o(5,1) ~= MDP.o(5,2))
        transitFlag(1,1,1,n) = 1;
    end
end

% Calculate average and standard error
allAC_av    = mean(allAC, 4);
allAC_pc    = 100 * allAC_av;
allAC       = squeeze(allAC);

% Calculate error rates
allNErrs    = sum(allAC==0, 4);
allErrsMk_av= sum(allErrsMk, 4) ./ allNErrs;
allErrsMk_p = 100 * allErrsMk_av;
allErrsRn_av= sum(allErrsRn, 4) ./ allNErrs;
allErrsRn_p = 100 * allErrsRn_av;
allErrsMx_av= sum(allErrsMx, 4) ./ allNErrs;
allErrsMx_p = 100 * allErrsMx_av;

% Save variables
%date_str   = datestr(now, 'yyyymmdd-HHSS');
%script_str = 'CRM_MDP_OptimalModel';
%save(sprintf('%s_%s_outputs', script_str, date_str));

% Plotting
if nreps == 1

    % Summary plot: show belief updates (and behaviour)
    %--------------------------------------------------------------------------
    spm_figure('GetWin','Figure 1'); clf
    spm_MDP_VB_trial(MDP);
    subplot(3,2,3)
    set(gcf, 'PaperPositionMode', 'auto');
    %print(gcf, sprintf('%s_%s_fig1', script_str, date_str), '-dtiff', '-r600');

    % Summary plot: illustrate phase-precession and responses
    %--------------------------------------------------------------------------
    spm_figure('GetWin','Figure 2'); clf
    spm_MDP_VB_LFP(MDP,[],1);
    set(gcf, 'PaperPositionMode', 'auto');
    %print(gcf, sprintf('%s_%s_fig2', script_str, date_str), '-dtiff', '-r600');

    % Plot hidden states in their own figures
    %--------------------------------------------------------------------------
    % State 1
    s1_name = 'Target Colour';
    s1_vals = colours;
    % State 2
    s2_name = 'Target Number';
    s2_vals = numbers;
    % State 3
    s3_name = 'Spatial Attention';
    s3_vals = {'Left', 'Right'};
    % State 4
    s4_name = 'Response';
    s4_vals = strcat(resps_words(:,1), '-', resps_words(:,2));
    s4_vals{end} = 'Null';
    
    % Plot for each state
    allSt = 1:4;
    allNames = {s1_name, s2_name, s3_name, s4_name};
    allVals = {s1_vals, s2_vals, s3_vals, s4_vals};
    allFigHeights = [255, 255, 187, 680];
    for s = 1 : length(allSt)
        st = allSt(s);
        name = allNames{s};
        vals = allVals{s};

        figure; image(64*(1 - MDP.X{st}))
        colormap('gray')
        title(sprintf('%s', name));
        hold on;
        plot(MDP.s(st,:),'.c','MarkerSize',16)
        legend('True state values', 'Location', 'EastOutside')
        format_graph(gca, 'Time step', '')
        set(gca, 'Box', 'on', 'TickLength', [0, 0.01]);
        xticks(1:3)
        yticks(1:length(vals))
        set(gcf, 'Position', [10, 50, 578, allFigHeights(s)]);
        yticklabels(vals)
        set(gcf, 'PaperPositionMode', 'auto');
        %print(gcf, sprintf('%s_%s_s%d', script_str, date_str, st), ...
        %    '-dtiff', '-r600');
    end

    % Plot policy decisions in a separate figure
    figure; image(64*(1 - squeeze(MDP.P(1,1,1,:,:))))
    colormap('gray')
    title('Action');
    hold on;
    plot(MDP.u(4,:),'.c','MarkerSize',16)
    legend('True state values', 'Location', 'EastOutside')
    format_graph(gca, 'Time step', '')
    set(gca, 'Box', 'on', 'TickLength', [0, 0.01]);
    xticks(1:3)
    yticks(1:length(vals))
    set(gcf, 'Position', [10, 50, 495, 680]);
    yticklabels(s4_vals)
    set(gcf, 'PaperPositionMode', 'auto');
    %print(gcf, sprintf('%s_%s_actions', script_str, date_str), ...
    %    '-dtiff', '-r600');
end

% Print time taken to run
duration = toc(startTime) / 60;
fprintf('\n\n...Script finished (time taken = %.2f minutes).\n', duration);
