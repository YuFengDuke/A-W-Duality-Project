function thresholds = plot_loss(loss, sigma_g, para, para_name)


lw = 3;
ms = 14;
fs = 36;

mean_loss = mean(loss, 3);
mean_sigma_g = mean(sigma_g, 3);

thresholds = zeros(size(para));
for i = 1:size(para,2)
    cum_sum_sharpness = cumsum(mean_sigma_g(i,:).^2) / sum(mean_sigma_g(i,:).^2);
    threshold = find(cum_sum_sharpness>0.6,1);
    thresholds(i) =  10;
end

mean_total_loss = sum(mean_loss,2);
mean_sharp_loss = zeros(size(para));
mean_flat_loss = zeros(size(para));

for i = 1:size(para,2)
    mean_sharp_loss(i) = sum(mean_loss(i,1:thresholds(i)));
    mean_flat_loss(i) = sum(mean_loss(i,thresholds(i)+1:end));
end

std_sharp_loss = zeros(size(para));
std_flat_loss = zeros(size(para));
for i = 1:size(para,2)
    sharp_loss = loss(i, 1:thresholds(i), :);
    flat_loss = loss(i, thresholds(i)+1:end, :);
    sum_sharp_loss = sum(sharp_loss, 2);
    sum_flat_loss = sum(flat_loss, 2);
    std_sharp_loss(i) = std(sum_sharp_loss, 0, 3);
    std_flat_loss(i) = std(sum_flat_loss, 0, 3);
end

sum_loss = sum(loss, 2);
std_sum_loss = std(sum_loss,0,3);

plot(para, mean_total_loss,'ks', 'LineWidth', 3, 'MarkerSize', ms, 'DisplayName','$\Delta L$');
hold on;
plot(para, mean_sharp_loss,'b-', 'LineWidth', 3, 'MarkerSize', ms, 'DisplayName','$\Delta L_s$');
plot(para, mean_flat_loss,'g-', 'LineWidth', 3, 'MarkerSize', ms, 'DisplayName','$\Delta L_f$');


h = errorbar(para, mean_total_loss, std_sum_loss, '.r', 'LineWidth', 2);
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
h = errorbar(para, mean_sharp_loss, std_sharp_loss, '.r', 'LineWidth', 2);
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
h = errorbar(para, mean_flat_loss, std_flat_loss, '.r', 'LineWidth', 2);
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

xlim([min(para),max(para)]);
ylim([0,max(sum(mean_loss,2))+0.1]);
l = legend;
l.set('Interpreter','latex');
xlabel(para_name,'Interpreter','latex');
ylabel('$\Delta L$','Interpreter','latex');
set(gca,'FontSize',fs,'FontName','Times New Roman');
set(gcf, 'position', [0 0 800 800]);