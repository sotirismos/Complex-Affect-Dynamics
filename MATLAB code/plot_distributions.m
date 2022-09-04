% import .MAT file
self = load('C:\Users\sotir\Documents\thesis\affect_dynamics\results\self\initAll.mat');
partner = load('C:\Users\sotir\Documents\thesis\affect_dynamics\results\partner\initAll.mat');
external = load('C:\Users\sotir\Documents\thesis\affect_dynamics\results\external\initAll.mat');

nbins = 20;
%x1 = histogram(self.SAll{1}(:,10), nbins, 'FaceColor', 'r');
%hold on
x2 = histogram(partner.SAll{1}(:,10), nbins, 'FaceColor', 'c');
hold on
x3 = histogram(external.SAll{1}(:,10), nbins, 'FaceColor', 'y');
xlabel('AR NA')
ylabel('Frequency')
title('Distribution of AR NA-partner vs external')
