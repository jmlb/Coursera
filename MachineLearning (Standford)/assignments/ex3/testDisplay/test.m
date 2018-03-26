% Gray Image
colormap(gray);
pad=1;
val = -0.5*ones(10, 10);
% Display Image
h = imagesc(val, [-1 1]);

% Do not show axis
axis image off

drawnow;