function [] = plot_c(data, para, para_name)

mean_data = mean(data,3);

lw = 2;
fs = 36;


for i = 1:size(para,2)
    plot(mean_data(i,:),'DisplayName',[para_name,'=',num2str(para(i))],'LineWidth',lw);
    hold on;
end


xlim([1,900]);
ylim([0,0.5]);
l = legend('FontSize',24);
l.set('Interpreter','latex');
xlabel('$n$','Interpreter','latex');
ylabel('$c_n$','Interpreter','latex');
set(gca,'FontSize',fs,'FontName','Times New Roman');
set(gcf, 'position', [0 0 900 800]);