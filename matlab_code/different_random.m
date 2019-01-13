clear all;

rng(10, 'Twister');
n = 10^9;
% list = 1:52;
% list = rand(1000,52);
prob = zeros(37, 1);
scores = [];
for i = 1:1000
    list = (0:51);
    list = list(randperm(length(list))) ;
    table = Table(list);
    score = table.calculate_score + 1;
    prob(score) = prob(score) + 1;
    % scores = [scores, score];
    % disp(i)
end

disp(prob)


% card = Card(12);
% card.show()
% disp(card)


