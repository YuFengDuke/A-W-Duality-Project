function [] = plot_sigma_w(data, thresholds, para, para_name)

mean_data = mean(data,3);
sum_data = sum(data,2);
std_data = std(sum_data,0,3);
mean_sum_data = mean(sum_data, 3);

lw = 2;
fs = 36;

N = size(mean_data, 2);
for i = 1:size(para,2)
    cumsum_data = cumsum(mean_data(i,:));
    if strcmp(para_name, '$H$') == 1
        plot(cumsum_data(1:para(i)*784),'DisplayName',[para_name,'=',num2str(para(i))],'LineWidth',lw);
        hold on;
        h = errorbar(para(i)*784,  mean_sum_data(i), std_data(i),'r', 'LineWidth', 2);
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    else
        plot(cumsum_data,'DisplayName',[para_name,'=',num2str(para(i))],'LineWidth',lw);
        hold on;
        h = errorbar(N,  mean_sum_data(i), std_data(i),'r', 'LineWidth', 2);
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
end
hold on;
for i = 1:size(para,2)
    cumsum_data = cumsum(mean_data(i,:));
    h = plot(thresholds(i), cumsum_data(thresholds(i)), 'r.', 'LineWidth',lw, 'MarkerSize', 24);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
end
% xticks([1,10,100,1000]);
set(gca,'Xscale','log');
% xlim([1,1e3]);
l = legend('FontSize',24);
l.set('Interpreter','latex');
xlabel('$n$','Interpreter','latex');
ylabel('$S_{w,n}$','Interpreter','latex');
set(gca,'FontSize',fs,'FontName','Times New Roman');
set(gcf, 'position', [0 0 800 800]);


if strcmp(para_name, '$\beta$') == 1 || strcmp(para_name, '$s$') == 1 
    figure;
    cumsum_data = cumsum(mean_data,2);
    for i = 1:size(para,2)
        if strcmp(para_name, '$s$') == 1
            plot(cumsum_data(i,:)./cumsum_data(1,:),'DisplayName',[para_name,'=',num2str(para(i))],'LineWidth',lw);
        else
            plot(cumsum_data(i,:)./cumsum_data(end,:),'DisplayName',[para_name,'=',num2str(para(i))],'LineWidth',lw);
        end
        hold on;
    end
    xticks([1,10,100,1000]);
    set(gca,'Xscale','log');
    xlim([1,1e3]);
    l = legend('FontSize',24);
    l.set('Interpreter','latex');
    xlabel('$n$','Interpreter','latex');
    ylabel('$\tilde{S}_{w,n}$','Interpreter','latex');
    set(gca,'FontSize',fs,'FontName','Times New Roman');
    set(gcf, 'position', [0 0 800 800]);
end

% mean_data = mean(data.^2,3);
% sum_mean_data = sum(mean_data,2);
% sum_mean_data = sum_mean_data / max(sum_mean_data);
% 
% axes('Position',[0.3 0.6 0.3 0.3]);
% plot(para, sum_mean_data,'LineWidth',2);
% ylim([0,1]);
% yticks([0,1]);
% xlabel(para_name,'Interpreter','latex');
% ylabel('$\tilde{S}_w$','Interpreter','latex');
% hYLabel = get(gca,'YLabel');
% set(hYLabel,'rotation',0,'VerticalAlignment','middle')
% set(gca,'FontSize',28,'FontName','Times New Roman');
% set(gcf, 'position', [0 0 800 800]);