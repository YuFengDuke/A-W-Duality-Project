function [] = plot_sigma_g(data, thresholds, para, para_name)

mean_data = mean(data,3);
std_data = std(data,0,3);

lw = 2;
fs = 36;

for i = 1:size(para,2)
    if strcmp(para_name, '$H$') == 1
        plot(mean_data(i,1:para(i) * 784),'DisplayName',[para_name,'=',num2str(para(i))],'LineWidth',lw);
    else
        plot(mean_data(i,:),'DisplayName',[para_name,'=',num2str(para(i))],'LineWidth',lw);
    end
    hold on;
    h = errorbar(1,  mean_data(i,1), std_data(i,1),'r', 'LineWidth', 2);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
end
hold on;
for i = 1:size(para,2)
    h = plot(thresholds(i), mean_data(i,thresholds(i)), 'r.', 'LineWidth',lw, 'MarkerSize', 24);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
end

% xticks([1,10,100,1000]);
set(gca,'Xscale','log');
% xlim([5e-1,1000]);

l = legend('FontSize',24);
l.set('Interpreter','latex');
xlabel('$n$','Interpreter','latex');
ylabel('$\sigma_{g,n}$','Interpreter','latex');
set(gca,'FontSize',fs,'FontName','Times New Roman');
set(gcf, 'position', [0 0 800 800]);


% mean_data = mean(data.^2,3);
% sum_mean_data = sum(mean_data,2);
% sum_mean_data = sum_mean_data / max(sum_mean_data);
% axes('Position',[0.5 0.6 0.3 0.3]);
% plot(para, sum_mean_data,'LineWidth',2);
% ylim([0,1]);
% yticks([0,1]);
% xlabel(para_name,'Interpreter','latex');
% ylabel('$\tilde{S}_g$','Interpreter','latex');
% hYLabel = get(gca,'YLabel');
% set(hYLabel,'rotation',0,'VerticalAlignment','middle')
% set(gca,'FontSize',28,'FontName','Times New Roman');
% set(gcf, 'position', [0 0 800 800]);