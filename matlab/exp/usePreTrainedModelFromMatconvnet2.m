function Solver = usePreTrainedModelFromMatconvnet2(Solver)
%%%%%% use it before 1st iteration in train
w1 = Solver.Solver_.net.get_weights();
w2 = load('./pretrainedModel/net-epoch-1_newLambda2_3_4_noise1.mat');

w1(1).weights{1} = w2.net.params(13).value;
w1(1).weights{2} = reshape(w2.net.params(14).value,1,1,1,length(w2.net.params(14).value));
w1(2).weights{1} = w2.net.params(15).value;
w1(2).weights{2} = reshape(w2.net.params(16).value,1,1,1,length(w2.net.params(16).value));

w1(3).weights{1} = w2.net.params(17).value;
w1(3).weights{2} = reshape(w2.net.params(18).value,1,1,1,length(w2.net.params(18).value));
w1(4).weights{1} = w2.net.params(19).value;
w1(4).weights{2} = reshape(w2.net.params(20).value,1,1,1,length(w2.net.params(20).value));

w1(5).weights{1} = w2.net.params(21).value;
w1(5).weights{2} = reshape(w2.net.params(22).value,1,1,1,length(w2.net.params(22).value));
w1(6).weights{1} = w2.net.params(23).value;
w1(6).weights{2} = reshape(w2.net.params(24).value,1,1,1,length(w2.net.params(24).value));

w1(7).weights{1} = w2.net.params(13).value;
w1(7).weights{2} = reshape(w2.net.params(14).value,1,1,1,length(w2.net.params(14).value));
w1(8).weights{1} = w2.net.params(15).value;
w1(8).weights{2} = reshape(w2.net.params(16).value,1,1,1,length(w2.net.params(16).value));

w1(9).weights{1} = w2.net.params(17).value;
w1(9).weights{2} = reshape(w2.net.params(18).value,1,1,1,length(w2.net.params(18).value));
w1(10).weights{1} = w2.net.params(19).value;
w1(10).weights{2} = reshape(w2.net.params(20).value,1,1,1,length(w2.net.params(20).value));

w1(11).weights{1} = w2.net.params(21).value;
w1(11).weights{2} = reshape(w2.net.params(22).value,1,1,1,length(w2.net.params(22).value));
w1(12).weights{1} = w2.net.params(23).value;
w1(12).weights{2} = reshape(w2.net.params(24).value,1,1,1,length(w2.net.params(24).value));

Solver.Solver_.net.set_weights(w1);