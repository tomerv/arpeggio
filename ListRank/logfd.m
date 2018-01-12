function f=logfd(x)
if abs(x) > 30
    f = 0;
else
    f=exp(x)./((1+exp(x)).^2);
end