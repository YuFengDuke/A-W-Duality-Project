clear;
load('MNIST_B_result.mat');

para_name = '$B$';
res = MNIST_B_result;
figure;
thresholds = plot_loss(res.loss1, res.sigma_g, res.para, para_name);
 
figure;
plot_sigma_g(res.sigma_g, thresholds, res.para, para_name);

figure;
plot_sigma_w(res.sigma_w, thresholds, res.para, para_name);
