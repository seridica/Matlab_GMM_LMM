function obj = fit(X,k,varargin)
% Not intended to be called directly. Use FITGMDIST to fit a GMDISTRIBUTION.
%
%   See also FITGMDIST.

%   Copyright 2008-2016 The MathWorks, Inc.

s = length(X);
X_data = []; % Aggregate all of the data not separated by subject
n = 0;
d = 0;
for i=1:s
    [ni,d] = size(X{i});
    n = n + ni;
    X_data = [X_data; X{i}];
end

if nargin > 2
    [varargin{:}] = convertStringsToChars(varargin{:});
end

if nargin < 2
    error(message('stats:gmdistribution:TooFewInputs'));
end

checkdata(X_data);

if ~isscalar(k) || ~isnumeric(k) || ~isfinite(k) ...
         || k<1 || k~=round(k)
    error(message('stats:gmdistribution:BadK'));
end

%remove NaNs from X
wasnan = any(isnan(X_data),2);
hadNaNs = any(wasnan);
if hadNaNs
    error(message('stats:gmdistribution:MissingData'));
end

[n, d] = size(X_data);
if n <= d
    error(message('stats:gmdistribution:TooFewN'));
end

if n <= k
    error(message('stats:gmdistribution:TooManyClusters'));
end

% parse input and error check
pnames = {      'start' 'replicates'  'covariancetype' 'sharedcovariance'  'regularizationvalue'  'options' 'probabilitytolerance'};
dflts =  {      'plus'           1      'full'            false             0                     []         1e-8};
[start,reps, CovType,SharedCov, RegV, options,probtol,setflags,extraArgs] ...
    = internal.stats.parseArgs(pnames, dflts, varargin{:});

if  ~setflags.covariancetype || ~setflags.regularizationvalue
    pnames = {'covtype','regularize' };
    dflts = {'full'      zeros(1,1,'like',X_data)};
    [CovType_oldinput,RegV_oldinput] = internal.stats.parseArgs(pnames, dflts, extraArgs{:});
    if ~setflags.covariancetype
        %if 'covariancetype' is not used, accept 'covtype' for backward compatibility
        CovType = CovType_oldinput;
    end
    if ~setflags.regularizationvalue
        RegV =  RegV_oldinput;
    end
end

options = statset(statset('gmdistribution'),options);

if ~isnumeric(reps) || ~isscalar(reps) || round(reps) ~= reps || reps < 1
    error(message('stats:gmdistribution:BadReps'));
end

if ~isnumeric(probtol) || ~isscalar(probtol)|| probtol >1e-6 || probtol<0
    error(message('stats:gmdistribution:BadProbTol'));
end


if ~isnumeric(RegV) || ~isscalar(RegV) || RegV < 0
    error(message('stats:gmdistribution:InvalidReg'));
end

varX = var(X_data) + RegV;
I = find(varX < eps(max(varX)));
if ~isempty(I)
    error(message('stats:gmdistribution:ZeroVariance', num2str( I )));
end

if ischar(CovType)
    covNames = {'diagonal','full'};
    i = find(strncmpi(CovType,covNames,length(CovType)));
    if isempty(i)
        error(message('stats:gmdistribution:UnknownCovType', CovType));
    end
    CovType = i;
else
    error(message('stats:gmdistribution:InvalidCovType'));
end

if ~islogical(SharedCov)
    error(message('stats:gmdistribution:InvalidSharedCov'));
end

options.Display = find(strncmpi(options.Display, {'off','notify','final','iter'},...
    length(options.Display))) - 1;

try
    [S,NlogL,optimInfo] =...
        gmcluster(X,k,start,reps,CovType,SharedCov,RegV,options,probtol);

    % Store results in object
    obj = gmdistribution_lmm;
    obj.NDimensions = d;
    obj.NComponents = k;
    obj.NSubjects = s;
    obj.PComponents = S.PComponents;
    obj.mu = S.mu;
    obj.Sigma = S.Sigma;
    obj.Converged = optimInfo.Converged;
    obj.Iters = optimInfo.Iters;
    obj.NlogL = NlogL;
    obj.SharedCov = SharedCov;
    obj.RegV = RegV;
    obj.ProbabilityTolerance = probtol;
    obj.Sigma_eff = S.Sigma_eff;
    obj.bhi = S.bhi;
    if CovType == 1
        obj.CovType = 'diagonal';
        if SharedCov
            nParam = obj.NDimensions;
        else
            nParam = obj.NDimensions * k;
        end
    else
        obj.CovType = 'full';
        if SharedCov
            nParam = obj.NDimensions * (obj.NDimensions+1)/2;
        else
            nParam = k*obj.NDimensions * (obj.NDimensions+1)/2;
        end

    end
    nParam = nParam + k-1 + k * obj.NDimensions + s*k*obj.NDimensions;
    obj.BIC = 2*NlogL + nParam*log(n);
    obj.AIC = 2*NlogL + 2*nParam;

catch ME
    rethrow(ME) ;
end
