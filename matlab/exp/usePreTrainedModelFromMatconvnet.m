function Solver = usePreTrainedModelFromMatconvnet(Solver)
%%%%%% use it before 1st iteration in train
w1 = Solver.Solver_.net.get_weights();
w2 = load('./pretrainedModel/net-epoch-1_newLambda2_3_4_noise1.mat');

w1(1).weights{1} = w2.net.params(1).value;
w1(1).weights{2} = reshape(w2.net.params(2).value,1,1,1,length(w2.net.params(2).value));
w1(2).weights{1} = w2.net.params(3).value;
w1(2).weights{2} = reshape(w2.net.params(4).value,1,1,1,length(w2.net.params(4).value));

w1(3).weights{1} = w2.net.params(5).value;
w1(3).weights{2} = reshape(w2.net.params(6).value,1,1,1,length(w2.net.params(6).value));
w1(4).weights{1} = w2.net.params(7).value;
w1(4).weights{2} = reshape(w2.net.params(8).value,1,1,1,length(w2.net.params(8).value));

w1(5).weights{1} = w2.net.params(9).value;
w1(5).weights{2} = reshape(w2.net.params(10).value,1,1,1,length(w2.net.params(10).value));
w1(6).weights{1} = w2.net.params(11).value;
w1(6).weights{2} = reshape(w2.net.params(12).value,1,1,1,length(w2.net.params(12).value));

w1(7).weights{1} = w2.net.params(1).value;
w1(7).weights{2} = reshape(w2.net.params(2).value,1,1,1,length(w2.net.params(2).value));
w1(8).weights{1} = w2.net.params(3).value;
w1(8).weights{2} = reshape(w2.net.params(4).value,1,1,1,length(w2.net.params(4).value));

w1(9).weights{1} = w2.net.params(5).value;
w1(9).weights{2} = reshape(w2.net.params(6).value,1,1,1,length(w2.net.params(6).value));
w1(10).weights{1} = w2.net.params(7).value;
w1(10).weights{2} = reshape(w2.net.params(8).value,1,1,1,length(w2.net.params(8).value));

w1(11).weights{1} = w2.net.params(9).value;
w1(11).weights{2} = reshape(w2.net.params(10).value,1,1,1,length(w2.net.params(10).value));
w1(12).weights{1} = w2.net.params(11).value;
w1(12).weights{2} = reshape(w2.net.params(12).value,1,1,1,length(w2.net.params(12).value));

Solver.Solver_.net.set_weights(w1);