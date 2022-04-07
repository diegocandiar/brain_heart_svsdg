function [LF2B, HF2B, B2LF, B2HF, tH2B, tB2H] = model_LFHF_SDG(EEG_comp, IBI, t_IBI, Fs, time, wind, opt_freq, opt_norm)
% The function model_LFHF_SDG computes the coupling coefficients between EEG
% in a defined frequency band with the spectral estimators of symathovagal activity
% in both directions.
% The inputs are:
% -EEG_comp: Matrix ChannelxTime containing the time-varying power
% -IBI: NON interpolated interbeat intervals in seconds
% -t_IBI: time of each IBI in seconds
% -FS: EEG sample frequency
% -time: the time array for EEG data
% -wind: time window in seconds in which the model is estimated (wind = 15)
% -opt_freq: if 'y', the central LF and HF are estimated from Burg
% periodogram, else f_lf = 0.1 and f_hf = 0.25
% -opt_norm: of 'y', the LF HF series are set to unitary STD and
% EEG=sqrt(EEG) to ease model convergence, else the model use of real values
%
% The outputs are:
% -LF2B: time-varying coupling coefficients from LF to EEG band
% -HF2B: time-varying coupling coefficients from HF to EEG band
% -B2LF: time-varying coupling coefficients from EEG band to LF
% -B2HF: time-varying coupling coefficients from EEG band to HF
% -tH2B: time array of the coefficients from heart to brain
% -tB2H: time array of the coefficients from brain to heart

% Author: Diego Candia-Rivera (diego.candia.r@ug.uchile.cl)

%% Check if one-channel EEG data has inverted dimension

[Nch,Nt] = size(EEG_comp);
if Nt==1
    EEG_comp = EEG_comp';
    [Nch,Nt] = size(EEG_comp);
end

%% Estimate LF and HF central frequencies
Frr = 4;
t_temp = t_IBI(1) : 1/Frr : t_IBI(end);
ibi_int = interp1(t_IBI, IBI, t_temp, 'spline'); 
ibi_int = detrend(ibi_int);

if strcmp(opt_freq, 'y')
    nfft =1024;
    order_burg = 30;
    [Pxx,F] = pburg(ibi_int, order_burg, nfft, Frr);
    ix_lf = find(F >= 0.04 & F<= 0.15);
    ix_hf = find(F >= 0.15 & F<= 0.4);
    F_lf = F(ix_lf);
    F_hf = F(ix_hf);
    P_lf = Pxx(ix_lf);
    P_hf = Pxx(ix_hf);
    
    [~, ix] = max(P_lf);
    f_lf = F_lf(ix);
    
    [~, ix] = max(P_hf);
    f_hf = F_hf(ix);
else
    f_lf = 0.1;
    f_hf = 0.25;
end

w_lf = 2*pi*f_lf;
w_hf = 2*pi*f_hf;


%% compute LF and HF time course
% Main parameters to set
Fbins = pow2(12); 
v0 = 0.03; 
tau0 = 0.06; 
lambda = 0.3; 

% Wigner-Ville distribution
ibi_x = hilbert(ibi_int-mean(ibi_int));
ibi_x = ibi_x(:);
n_ibi = 1:length(ibi_x);

[xrow,xcol] = size(ibi_x);
[trow,tcol] = size(n_ibi);

wvx= zeros (Fbins,tcol);  
for icol = 1:tcol
    ti = n_ibi(icol); 
    taumax = min([ti-1,xrow-ti,round(Fbins/2)-1]);
    tau = -taumax:taumax; 
    indices = rem(Fbins+tau,Fbins)+1;
    wvx(indices,icol) = ibi_x(ti+tau,1) .* conj(ibi_x(ti-tau,xcol));
    tau = round(Fbins/2); 
    if ti<=(xrow-tau) && ti>=(tau+1)
        wvx(tau+1,icol) = 0.5 * (ibi_x(ti+tau,1) * conj(ibi_x(ti-tau,xcol))  + ...
                               ibi_x(ti-tau,1) * conj(ibi_x(ti+tau,xcol))) ;
    end
end

wvx= fft(wvx); 
if xcol==1
    wvx=real(wvx); 
end

% Ambiguity function
AF = fft(ifft(wvx).'); clear wvx

if rem(size(AF,2),512)~=0 && rem(size(AF,2),1024)~=0 && rem(size(AF,2),4096)~=0  && rem(size(AF,2),256)~=0
    AF=AF.';
end

[Crow,Ccol] = size(AF);
dy = (-Ccol/2 : Ccol/2-1) / (Ccol/2);

if Crow/2- fix(Crow/2)==0
    dx = (-Crow/2:Crow/2-1) / (Crow/2);
else
    dx = (-Crow/2-1/2:Crow/2-3/2) / (Crow/2);
end

[x,y] = meshgrid(dy,dx);
tau1 = x/tau0;
v1 = y/v0;

beta = 1;
gamma = 1;
alfa = 0;
r = 0;
mu =  (tau1.^2.*(v1.^2).^alfa + (tau1.^2).^alfa .* v1.^2 +2.*r*((tau1.*v1).^beta).^gamma);
clear tau1 v1 x y

% Compute LF HF
t_wind = exp(-pi*( mu.^2).^lambda);
t_wind = t_wind/max(max(t_wind));
Asmooth = (AF.*fftshift(t_wind)).';
TFR_RR = real(ifft(fft(Asmooth).').');

f = linspace(0,Frr/2,Fbins);
[~,LF_ix] = find(f>=0.04 & f<=0.15);
[~,HF_ix] = find(f>=0.15 & f<=0.4);

LF = trapz(TFR_RR(LF_ix,:));
HF = trapz(TFR_RR(HF_ix,:));

%% HRV model based on Poincare plot

ss = wind*Fs;
sc = 1;
nt = ceil((length(time)-ss)/sc);

Cs = zeros(1,nt);
Cp = zeros(1,nt);
TM = zeros(1,nt);


for i = 1:nt
    ix1 = (i-1)*sc + 1;
    ix2 = ix1 + ss - 1;
    ixm = floor(mean(ix1:ix2));   
%     ixm = ix1;
    t1 = time(ix1);
    t2 = time(ix2);
    ix = find(t_IBI >= t1 & t_IBI<= t2);

    
    mu_ibi = mean(IBI(ix));
    mu_hr = 1/mu_ibi;  
    
    G = sin(w_hf/(2*mu_hr))-sin(w_lf/(2*mu_hr)); 
    
    M_11 = sin(w_hf/(2*mu_hr))*w_lf*mu_hr/(sin(w_lf/(2*mu_hr))*4);
    M_12 = -sqrt(2)*w_lf*mu_hr/(8*sin(w_lf/(2*mu_hr)));
    M_21 = -sin(w_lf/(2*mu_hr))*w_hf*mu_hr/(sin(w_hf/(2*mu_hr))*4);
    M_22 = sqrt(2)*w_hf*mu_hr/(8*sin(w_hf/(2*mu_hr)));
    M = [M_11, M_12; M_21, M_22];
    L = max(IBI(ix))-min(IBI(ix));         
    W = sqrt(2)*max(abs(IBI(ix(2:end))-IBI(ix(1:end-1))));
    C = 1/G*M*[L; W];
    Cs(i) = C(1);   Cp(i) = C(2);
    TM(i) = time(ixm);
end

%% normalization 
if strcmp(opt_norm,'y')
    Cs = Cs/std(Cs);  Cp = Cp/std(Cp);
    EEG_comp = sqrt(EEG_comp);
end

%% interpolation (edges are extended to avoid extrapolation)
Cs = interp1([time(1) TM time(end)], [Cs(1) Cs Cs(end)], time, 'spline');
Cp = interp1([time(1) TM time(end)], [Cp(1) Cp Cp(end)], time, 'spline');

LF = interp1([time(1) t_temp time(end)], [LF(1) LF LF(end)], time, 'spline');
HF = interp1([time(1) t_temp time(end)], [HF(1) HF HF(end)], time, 'spline');

%% RUN MODEL

parfor ch = 1:Nch % Consider use parallel computing here
    [LF2B(ch,:), HF2B(ch,:), B2LF(ch,:), B2HF(ch,:)] = SDG(EEG_comp(ch,:), LF, HF, Cs, Cp, wind);
end

%% TIME VECTOR DEFINITION
% at the beginning of the window
tH2B = time(1 : end-wind);
tB2H = time(wind+1 : end-wind);

% in the middle of the window
% tH2B = time(floor(wind/2) +1 : end-ceil(wind/2));
% tB2H = time(wind+1 : end-wind) + wind/2;

end

function [LF_to_EEG, HF_to_EEG, EEG_to_LF, EEG_to_HF] = SDG(EEG_ch, HRV_LF, HRV_HF, Cs_i, Cp_i, window)

    Nt = length(EEG_ch);
    %% First time window is calculated separately
    for i = 1 : window
        arx_data = iddata(EEG_ch(i:i+window)', HRV_LF(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]);                                                 
        LF_to_EEG(i) = model_eegP.B(2);

        arx_data = iddata(EEG_ch(i:i+window)', HRV_HF(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]);                                                 
        HF_to_EEG(i) = model_eegP.B(2);
        
        pow_eeg(1,i) = mean(EEG_ch(i:i+window));                                 
    end
    
    
    for i = window+1:min([length(Cp_i),Nt-window, length(HRV_LF)-window])

        %% Heart to brain estimation
        arx_data = iddata(EEG_ch(i:i+window)', HRV_LF(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]); 
        LF_to_EEG(i) = model_eegP.B(2); 

        arx_data = iddata(EEG_ch(i:i+window)', HRV_HF(i:i+window)',1); 
        model_eegP = arx(arx_data,[1 1 1]); 
        HF_to_EEG(i) = model_eegP.B(2); 
        
        pow_eeg(1,i) = mean(EEG_ch(i:i+window));

        %% Brain to heart estimation
        if i-window <= length(Cp_i)-window-1
            EEG_to_HF(i-window) = mean((Cp_i(i-window:i))./pow_eeg(i-window:i));
            EEG_to_LF(i-window) = mean((Cs_i(i-window:i))./pow_eeg(i-window:i));
        else

            EEG_to_HF(i-window) = EEG_to_HF(i-window-1);
            EEG_to_LF(i-window) = EEG_to_LF(i-window-1);
        end
    end

end
