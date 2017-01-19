

function grad = B_MVDR_eigenVector(X, curr_layer, beamform_layer, after_power_layer)
%Implementation of bp of eigenVector basd beamforming.

% X is the multichannel complex spectrum inputs
[D,C,T,N] = size(X);
% weight is the beamforming weight
weight = reshape(curr_layer.a, D,C,N);
phi_s = curr_layer.phi_s;
phi_n = curr_layer.phi_n;

if isfield(curr_layer, 'noiseCovL2')
    noiseCovL2 = curr_layer.noiseCovL2;
else
    noiseCovL2 = 0;  % add noiseCovRegularization*\lambda*I to noise covariance, where \lambda is the maximum eigenvalue
end

% Y is the beamforming's output
Y = beamform_layer.a;
% future_grad is the gradient of the power specturm of beamformed signal
future_grad = after_power_layer.grad;
future_layers{1} = beamform_layer;
future_grad2 = GetFutureGrad(future_layers, curr_layer);
grad = GetGradUtt(X,Y,phi_s, phi_n, weight,  future_grad, future_grad2, noiseCovL2);
end

%%
function grad = GetGradUtt(X,Y,phi_s, phi_n, weight, future_grad, future_grad2, noiseCovL2)
[D,C,T,N] = size(X);
u = zeros(C,1);
u(1) = 1;
future_grad2 = reshape(future_grad2, 257,C);
dw_dd = zeros(C, D); % intermediate derivative, 
grad_phi_s = zeros(C,C,D);
grad_phi_n = zeros(C,C,D);

for f=1:D
    phi_s_f = phi_s{1, 1, f};  %speech covariance matrix of freq bin f
    phi_n_inv = inv(phi_n{1,1,f} + 1e-3*eye(C));
    [u, v] = eig(phi_s_f);
    d = u(:, end); % this is the steering vector
    scale = d'*phi_n_inv*d;
    %calculate the derivative between beamforming weight and steering
    %vector
    for ii = 1:C
        tmp = zeros(C, 1);
        tmp(ii, 1) = d'*phi_n_inv(:, ii);
        dw_dd(:, f) = dw_dd(:, f) + future_grad2(f, ii) * (phi_n_inv(ii, :).' - weight(f,ii)*tmp)/scale;
        for jj = 1:C
            dd_dR = zeros(C, C);
            dd_dR(jj, :) = ones(1, C) ./ (v(end, end) * conj(phi_s_f(jj, :)));
            grad_phi_s(:, :, f) = grad_phi_s(:, :, f) + dw_dd(jj, f)*dd_dR;
        end
    end
    
    
    for ii = 1:C
        tmp = zeros(C, C);
        tmp(ii, :) = d.';
        dw_dR_inv = (tmp  - weight(f, ii) * d*d')/scale;
        dw_dR = -phi_n_inv' * dw_dR_inv * phi_n_inv';
        grad_phi_n(:, :, f) = grad_phi_n(:, :, f) + future_grad2(f, ii) * dw_dR/scale;
    end
    
%     for ii = 1:C
%         grad_phi_s(:,:,f) = grad_phi_s(:,:,f) + future_grad2(f,ii) *  phi_n_inv(ii,:).' * u' /lambda{1,1,f};
%     end
%     for ii = 1:C
%         tmp = zeros(C,C);
%         tmp(ii,:) = u'*phi_s{1,1,f}';
%         grad_phi_n(:,:,f) = grad_phi_n(:,:,f) - conj(future_grad2(f,ii)) *  phi_n_inv' * tmp * phi_n_inv' / lambda{1,1,f};
%     end
%     for ii = 1:C
%         grad_phi_s(:,:,f) = grad_phi_s(:,:,f) - future_grad2(f,ii) * weight(f,ii) * phi_n_inv/lambda{1,1,f};
%         grad_phi_n(:,:,f) = grad_phi_n(:,:,f) + conj(future_grad2(f,ii)) * conj(weight(f,ii)) * (phi_n_inv * phi_s{1,1,f} * phi_n_inv)' / lambda{1,1,f};
%     end
end

grad = conj([reshape(grad_phi_s,C*C*D,1); reshape(grad_phi_n,C*C*D,1)]);
end

