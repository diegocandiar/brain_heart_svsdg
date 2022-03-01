function [S2B, P2B, B2S, B2P, tH2B, tB2H] = model_SVSDG(EEG_comp, IBI, SAI, PAI, FS, time, wind)
% The function model_SVSDG computes the coupling coefficients between EEG
% in a defined frequency band with the estimators of symathovagal activity
% in both directions.
% The inputs are:
% EEG_comp: Matrix ChannelxTime containing the time-varying power
% IBI: evenly interpolated interbeat intervals in seconds
% SAI: evenly interpolated sympathetic activity index
% PAI: evenly interpolated sympathetic activity index
% FS: sample frequency
% time: the time array that fits all inputs
% wind: time window in seconds in which the model is estimated (wind = 15)
% The outputs are:
% S2B: time-varying coupling coefficients from SAI to EEG band
% P2B: time-varying coupling coefficients from PAI to EEG band
% B2S: time-varying coupling coefficients from EEG band to SAI
% B2P: time-varying coupling coefficients from EEG band to PAI
% tH2B: time array of the coefficients from heart to brain
% tB2H: time array of the coefficients from brain to heart

% Author: Diego Candia-Rivera (diego.candia.r@ug.uchile.cl)
% Please cite: https://doi.org/10.1016/j.neuroimage.2022.119023


[Nch,Nt] = size(EEG_comp);
if (Nt==1)&&(Nch~=1)
    EEG_comp = EEG_comp';
    [Nch,Nt] = size(EEG_comp);
end

%% HRV model
% select time window
ss = wind*FS;

% step
sc = 1;

% final samples
nt = ceil((length(time)-ss)/sc);

% reallocate memory
CS = zeros(1,nt);
CP = zeros(1,nt);
TM = zeros(1,nt);
HR = zeros(1,nt);

% regression
for i = 1:nt
    ix1 = ((i-1)*sc+1);
    ix2 = (((i-1)*sc)+ss);
    ixm = floor(mean(ix1:ix2));
    x = [SAI(ix1:ix2)' PAI(ix1:ix2)'];
    y = 1/(IBI(ix1:ix2)' - 1/mean(IBI(ix1:ix2)));
    [beta,~,~] = glmfit(x,y,'normal', 'constant', 'off');
    TM(i) = time(ixm);
    CS(i) = beta(1);
    CP(i) = beta(2);
    HR(i) = mean(IBI(ix1:ix2));
end


% interpolation
CS = interp1(TM, CS, time, 'spline');
CP = interp1(TM, CP, time, 'spline');

%% RUN MODEL

if Nch > 1
    parfor ch = 1:Nch % PARALLEL COMPUTING
        [S2B(ch,:), P2B(ch,:), B2S(ch,:), B2P(ch,:)] = SVSDG(EEG_comp(ch,:), SAI, PAI, CS, CP, wind);
    end
else
    [S2B, P2B, B2S, B2P] = SVSDG(EEG_comp, SAI, PAI, CS, CP, wind);
end

%% TIME VECTOR DEFINITION

tH2B = time(floor(wind/2) +1 : end-ceil(wind/2)) + wind/2;
tB2H = time(wind+1 : end-wind) + wind;

end

function [SAI_to_EEG, PAI_to_EEG, EEG_to_SAI, EEG_to_PAI] = SVSDG(EEG_ch, HRV_S, HRV_P, CSr, CPr, window)

    Nt = length(EEG_ch);
    %% First time window is calculated separately
    for i = 1:window
        arx_data = iddata(EEG_ch(i:i+window)', HRV_S(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]);                                                 
        SAI_to_EEG(i) = model_eegP.B(2);

        arx_data = iddata(EEG_ch(i:i+window)', HRV_P(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]);                                                 
        PAI_to_EEG(i) = model_eegP.B(2);
        
        pow_eeg(1,i) = mean(EEG_ch(i:i+window));                                 
    end
    
    
    for i = window+1:min([length(CPr),Nt-window, length(HRV_S)-window])

        %% Heart to brain estimation
        arx_data = iddata(EEG_ch(i:i+window)', HRV_S(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]); 
        SAI_to_EEG(i) = model_eegP.B(2); 

        arx_data = iddata(EEG_ch(i:i+window)', HRV_P(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]); 
        PAI_to_EEG(i) = model_eegP.B(2); 
        
        pow_eeg(1,i) = mean(EEG_ch(i:i+window));

        %% Brain to heart estimation
        if i-window <= length(CPr)-window-1
            EEG_to_PAI(i-window) = mean((CPr(i-window:i))./pow_eeg(i-window:i));
            EEG_to_SAI(i-window) = mean((CSr(i-window:i))./pow_eeg(i-window:i));
        else

            EEG_to_PAI(i-window) = EEG_to_PAI(i-window-1);
            EEG_to_SAI(i-window) = EEG_to_SAI(i-window-1);
        end
    end

end
