function Solver = usePreTrainedModelFromMatconvnet3(Solver)
%%%%%% use it before 1st iteration in train
w1 = Solver.Solver_.net.get_weights();
w2 = load('./pretrainedModel/net-epoch-1_newLambda2_3_4_noise1.mat');

w1(1).weights{1} = w2.net.params(25).value;
w1(1).weights{2} = reshape(w2.net.params(26).value,1,1,1,length(w2.net.params(26).value));
w1(2).weights{1} = w2.net.params(27).value;
w1(2).weights{2} = reshape(w2.net.params(28).value,1,1,1,length(w2.net.params(28).value));

w1(3).weights{1} = w2.net.params(29).value;
w1(3).weights{2} = reshape(w2.net.params(30).value,1,1,1,length(w2.net.params(30).value));
w1(4).weights{1} = w2.net.params(31).value;
w1(4).weights{2} = reshape(w2.net.params(32).value,1,1,1,length(w2.net.params(32).value));

w1(5).weights{1} = w2.net.params(33).value;
w1(5).weights{2} = reshape(w2.net.params(34).value,1,1,1,length(w2.net.params(34).value));
w1(6).weights{1} = w2.net.params(35).value;
w1(6).weights{2} = reshape(w2.net.params(36).value,1,1,1,length(w2.net.params(36).value));

w1(7).weights{1} = w2.net.params(25).value;
w1(7).weights{2} = reshape(w2.net.params(26).value,1,1,1,length(w2.net.params(26).value));
w1(8).weights{1} = w2.net.params(27).value;
w1(8).weights{2} = reshape(w2.net.params(28).value,1,1,1,length(w2.net.params(28).value));

w1(9).weights{1} = w2.net.params(29).value;
w1(9).weights{2} = reshape(w2.net.params(30).value,1,1,1,length(w2.net.params(30).value));
w1(10).weights{1} = w2.net.params(31).value;
w1(10).weights{2} = reshape(w2.net.params(32).value,1,1,1,length(w2.net.params(32).value));

w1(11).weights{1} = w2.net.params(33).value;
w1(11).weights{2} = reshape(w2.net.params(34).value,1,1,1,length(w2.net.params(34).value));
w1(12).weights{1} = w2.net.params(35).value;
w1(12).weights{2} = reshape(w2.net.params(36).value,1,1,1,length(w2.net.params(36).value));

Solver.Solver_.net.set_weights(w1);