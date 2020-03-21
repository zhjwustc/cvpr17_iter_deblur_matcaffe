function Solver = usePreTrainedModel2(Solver)
%%%%%% use it before 1st iteration in train
w1 = Solver.Solver_.net.get_weights();

w2 = load('./pretrainedModel/exp102_4_iter2_weights_iter_122000.mat'); w2 = w2.weights;
w1 = w2;

% for i=1:25
%     w1(i).weights = w2(i).weights;
% end
% 
% 
% for i=26:41
%     w1(i+2).weights = w2(i).weights;
% end
% 
% 
% 
% for i=42:44
%     w1(i+18).weights = w2(i).weights;
% end


Solver.Solver_.net.set_weights(w1);
